#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, time, json, argparse, subprocess
from multiprocessing import Process, Manager, Queue
import imageio_ffmpeg
import socket, contextlib

# ───────────── argparse ─────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--max_video_processes", type=int, default=1)
parser.add_argument("--max_queue_size", type=int, default=20)
parser.add_argument("--skip_conversion", action="store_true", help="Skip .mov to .mp4 conversion")
args = parser.parse_args()

# ───────────── constants ─────────────
VBENCH_DIMS = ["motion_smoothness", "dynamic_degree"]
TMP_DIR = "./tmp"
SENTINEL = ("__DONE__", None, None, None, None)

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

OUT_FILE = os.path.join(args.output_path, "output.txt")
DBG_FILE = os.path.join(args.output_path, "debug.txt")

def get_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# ───────────── producer ─────────────
def convert_to_mp4_worker(task_list, q: Queue, n_consumer: int):
    """
    task_list 裡每筆是 (orig_path, video_id, video_url)
    轉檔完成後送進 queue → (orig_path, video_id, video_url, mp4_path, conv_time)
    """
    try:
        for vpath, vid, vurl in task_list:
            mp4_path, conv_t = vpath, 0.0  # 預設值（非 .mov 或已快取）

            # 如果開啟了 --skip_conversion，直接使用原始檔案
            if args.skip_conversion:
                print(f"[SKIP_CONVERT] {vpath} - Using original file", flush=True)
                q.put((vpath, vid, vurl, vpath, conv_t))
                continue

            if vpath.lower().endswith(".mov"):
                mp4_path = os.path.join(TMP_DIR, f"{os.path.splitext(vpath.replace('/','_'))[0]}.mp4")
                if not os.path.exists(mp4_path):
                    tic = time.time()
                    try:
                        subprocess.run(
                            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", vpath,
                             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                             "-c:a", "aac", "-b:a", "128k", mp4_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        conv_t = time.time() - tic
                        print(f"[CONVERT] ✓ {mp4_path} {conv_t:.2f}s", flush=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[CONVERT] ✗ {vpath}\n{e.stderr.decode()}", flush=True)
                        mp4_path = None
            q.put((vpath, vid, vurl, mp4_path, conv_t))
    finally:
        for _ in range(n_consumer):
            q.put(SENTINEL)

def run_vbench(mp4, dim, odir):
    """return (score, elapsed, err_msg, stderr_text)"""
    cmd = [
        "python", "./evaluate_safe.py",
        "--videos_path", mp4,
        "--dimension", dim,
        "--mode", "custom_input",
        "--output_path", odir,
    ]

    env = os.environ.copy()  # 使用默認環境變量

    tic = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    et = time.time() - tic
    err_txt = result.stderr.strip()[:500]  # 前 500 字即可，太長寫檔

    if result.returncode != 0:
        return -1, et, f"CLI_FAIL(rc={result.returncode})", err_txt

    json_files = [f for f in os.listdir(odir) if f.endswith("_eval_results.json")]
    if not json_files:
        return -1, et, "NO_JSON", err_txt

    latest = os.path.join(odir, max(json_files, key=lambda f: os.path.getmtime(os.path.join(odir, f))))
    try:
        with open(latest) as jf:
            j = json.load(jf)
        val = j.get(dim, -1)
        if isinstance(val, list):
            val = val[1][0].get("video_results", -1) if dim == "dynamic_degree" \
                  else val[1][0].get("video_results", -1)
        score = float(val) if val != -1 else -1
    except Exception as e:
        return -1, et, f"PARSE_ERR:{e}", err_txt

    return score, et, "", err_txt

# ───────────── consumer ─────────────
def consumer(q: Queue, results, dbg, total_tasks):
    def log(m): print(m, file=open(dbg, "a"), flush=True)
    processed_count = 0  # 記錄已處理的影片數量
    start_time = time.time()  # 記錄處理開始時間

    while True:
        vpath, vid, vurl, mp4, _ = q.get()
        if vpath == "__DONE__":
            break
        if not mp4:
            log(f"{vid}\tconvert_failed")
            results.append({"videoid": vid, "Imgurl": vurl,
                            "motion_smoothness": -1, "dynamic_degree": -1})
            continue

        row = {"videoid": vid, "Imgurl": vurl}
        base_out = os.path.join(args.output_path, "evaluate_result")
        video_start_time = time.time()  # 記錄單部影片處理開始時間
        for dim in VBENCH_DIMS:
            odir = os.path.join(base_out, dim, vid)  # 每影片專屬子目錄
            os.makedirs(odir, exist_ok=True)
            score, elapsed, err_tag, stderr_txt = run_vbench(mp4, dim, odir)
            row[dim] = score
            if err_tag:  # 不論何種錯誤都寫 tag + 簡短 stderr
                log(f"{vid}\t{dim}\t{err_tag}\t{stderr_txt}")
            else:
                log(f"{vid}\t{dim}:{score}\t{elapsed:.2f}s")
        video_elapsed_time = time.time() - video_start_time  # 單部影片處理時間
        results.append(row)

        # 即時將結果寫入 OUT_FILE
        with open(OUT_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, ["videoid", "Imgurl"] + VBENCH_DIMS, delimiter="\t")
            writer.writerow(row)

        # 刪除暫存的 mp4 檔案
        if mp4.startswith(TMP_DIR) and os.path.exists(mp4):
            os.remove(mp4)
            log(f"[CLEAN] {mp4}")

        # 更新進度
        processed_count += 1
        print(f"[PROGRESS] Processed {processed_count}/{total_tasks} videos: {vid}", flush=True)
        print(f"[RESULT] {vid} - motion_smoothness: {row['motion_smoothness']}, dynamic_degree: {row['dynamic_degree']}", flush=True)
        print(f"[TIME] {vid} - Processing time: {video_elapsed_time:.2f}s", flush=True)

    # 顯示總處理時間
    total_elapsed_time = time.time() - start_time
    print(f"[DONE] All videos processed in {total_elapsed_time:.2f}s", flush=True)

# ───────────── main ─────────────
def main():
    tasks, skipped = [], []
    processed_videos = set()

    # 檢查 OUT_FILE 是否已存在，並讀取已處理的檔案
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                processed_videos.add(row[0])  # 使用第一列作為 videoid
        print(f"[MAIN] Resuming from {OUT_FILE}, already processed {len(processed_videos)} videos")

    # 讀取 input_tsv，過濾掉已處理的檔案
    with open(args.input_tsv) as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) < 4: continue
            vpath, vid, vq, vurl = r[:4]
            try:
                vq = float(vq)
            except ValueError:
                continue
            if vq > .3 or not os.path.exists(vpath):
                skipped.append({"videoid": vid, "Imgurl": vurl,
                                "motion_smoothness": -1, "dynamic_degree": -1})
                continue
            if vid in processed_videos:
                print(f"[SKIP] {vid} already processed, skipping", flush=True)
                continue
            tasks.append((vpath, vid, vurl))
    print(f"[MAIN] tasks={len(tasks)} skipped={len(skipped)}")

    with Manager() as m:
        q = m.Queue(args.max_queue_size)
        results = m.list(skipped)
        open(DBG_FILE, "w").close()

        prod = Process(target=convert_to_mp4_worker, args=(tasks, q, args.max_video_processes))
        prod.start()
        workers = [Process(target=consumer, args=(q, results, DBG_FILE, len(tasks)))
                   for _ in range(args.max_video_processes)]
        for w in workers: w.start()

        prod.join()
        for w in workers: w.join()

        # 持續將結果寫入 OUT_FILE
        with open(OUT_FILE, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows([[row["videoid"], row["Imgurl"], row["motion_smoothness"], row["dynamic_degree"]] for row in results])
    
    print(f"[DONE] → {OUT_FILE}\n[DONE] debug → {DBG_FILE}")

if __name__ == "__main__":
    main()
