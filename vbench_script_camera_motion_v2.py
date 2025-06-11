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

# ====== 先放在檔案前半段 (import 之後) 的共用設定 ======
BATCH_SIZE = 10                                           # 一批幾支影片
import uuid, tempfile

# --------------------------------------------------------
# 把「一批影片」丟給 VBench 的小工具
# --------------------------------------------------------
def run_vbench_batch(batch_rows, odir, batch_idx):
    """
    batch_rows : [(mp4_path, video_id, url), ...]  長度 <= BATCH_SIZE
    odir       : 這批輸出目錄
    batch_idx  : 第幾批 (1-based)
    return     : dict {video_id: predict_type 或 error_tag}
    """
    # (1) 生成暫存 .tsv，evaluate_i2v 現在能直接吃
    batch_tsv = os.path.join(odir, f"batch_{batch_idx:03d}.tsv")
    with open(batch_tsv, "w") as f:
        for mp4, vid, url in batch_rows:
            print("\t".join([mp4, vid, "", "", url]), file=f)

    # (2) 呼叫 evaluate_i2v
    cmd = [
        "python", "../VBench/evaluate_i2v.py",
        "--videos_path", batch_tsv,
        "--mode", "custom_input",
        "--dimension", "camera_motion",
        "--output_path", odir,
    ]
    env = os.environ.copy()
    env["HUB_NO_GIT"] = "1"              # 關掉 git ping → 更快
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)

    # (3) 找到最新 _eval_results.json
    jfile = max(
        (p for p in os.listdir(odir) if p.endswith("_eval_results.json")),
        key=lambda p: os.path.getmtime(os.path.join(odir, p))
    )
    with open(os.path.join(odir, jfile)) as jf:
        res = json.load(jf)["camera_motion"][2]          # list[dict]

    # (4) 轉成 {basename: predict_str}
    name2pred = {
        os.path.basename(r["video_path"]):
        ";".join(r.get("predict_type", []))
        for r in res
    }
    # (5) 回到 {video_id: predict}
    out = {}
    for mp4, vid, _ in batch_rows:
        out[vid] = name2pred.get(os.path.basename(mp4), "PARSE_FAIL")
    return out

def run_vbench(mp4, odir):
    """return (predict_type, elapsed, err_msg, stderr_text)"""
    cmd = [
        #"python", "./vbench_cust/evaluate_i2v.py",
        "python", "../VBench/evaluate_i2v.py",
        "--videos_path", mp4,
        "--mode", "custom_input",
        "--dimension", "camera_motion",
        "--output_path", odir,
    ]

    env = os.environ.copy()  # 使用默認環境變量

    tic = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    et = time.time() - tic
    err_txt = result.stderr.strip()[:5000]  # 前 500 字即可，太長寫檔

    if result.returncode != 0:
        return None, et, f"CLI_FAIL(rc={result.returncode})", err_txt

    json_files = [f for f in os.listdir(odir) if f.endswith("_eval_results.json")]
    if not json_files:
        return None, et, "NO_JSON", err_txt

    latest = os.path.join(odir, max(json_files, key=lambda f: os.path.getmtime(os.path.join(odir, f))))
    try:
        with open(latest) as jf:
            j = json.load(jf)
        predict_types = j["camera_motion"][2][0].get("predict_type", [])
        predict_types_str = ";".join(predict_types)  # 將 predict_type 用分號分隔
    except Exception as e:
        return None, et, f"PARSE_ERR:{e}", err_txt

    return predict_types_str, et, "", err_txt

# --------------------------------------------------------
# NEW consumer —— 每 50 片 flush 一批
# --------------------------------------------------------
def consumer(q: Queue, results, dbg, total_tasks):
    def log(msg): print(msg, file=open(dbg, "a"), flush=True)

    bucket       = []        # 暫存 (mp4, vid, url)
    batch_idx    = 0
    processed    = 0
    start_time   = time.time()

    def flush_batch(final=False):
        nonlocal bucket, batch_idx, processed
        if not bucket:
            return
        batch_idx += 1
        odir = os.path.join(
            args.output_path, "evaluate_result",
            f"camera_motion_batch_{batch_idx:03d}"
        )
        os.makedirs(odir, exist_ok=True)

        try:
            preds = run_vbench_batch(bucket, odir, batch_idx)
        except subprocess.CalledProcessError as e:
            preds = {vid: f"BATCH_FAIL({e.returncode})"
                     for _, vid, _ in bucket}
            log(f"[BATCH_FAIL] batch_{batch_idx:03d} rc={e.returncode}")

        # 把批次結果寫回每支影片
        for mp4, vid, url in bucket:
            pred = preds.get(vid, "unknown")
            row  = {"videoid": vid, "Imgurl": url, "camera_motion": pred}
            results.append(row)

            # 立即 append 至 OUT_FILE 方便 resume
            with open(OUT_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f,
                         ["videoid", "Imgurl", "camera_motion"],
                         delimiter="\t")
                writer.writerow(row)

            log(f"{vid}\tcamera_motion:{pred}")
            processed += 1
            print(f"[PROGRESS] Processed {processed}/{total_tasks} videos: {vid}",
                  flush=True)

            # 若 mp4 是暫存檔就刪掉
            if mp4.startswith(TMP_DIR) and os.path.exists(mp4):
                try:
                    os.remove(mp4)
                    log(f"[CLEAN] {mp4}")
                except Exception as e:
                    log(f"[CLEAN_FAIL] {mp4}\t{e}")

        bucket = []    # 清空

    # ---------------- 主迴圈 ----------------
    while True:
        vpath, vid, vurl, mp4, _ = q.get()
        if vpath == "__DONE__":
            flush_batch(final=True)      # 清理殘餘未滿 50 片的 bucket
            break

        if not mp4:
            log(f"{vid}\tconvert_failed")
            results.append({"videoid": vid,
                            "Imgurl": vurl,
                            "camera_motion": "convert_failed"})
            # 即時寫檔
            with open(OUT_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f,
                         ["videoid", "Imgurl", "camera_motion"],
                         delimiter="\t")
                writer.writerow(results[-1])
            processed += 1
            continue

        bucket.append((mp4, vid, vurl))
        if len(bucket) >= BATCH_SIZE:
            flush_batch()

    # ---------------- 收尾 ----------------
    total_elapsed = time.time() - start_time
    print(f"[DONE] All videos processed in {total_elapsed:.2f}s", flush=True)

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
            if len(r) < 5: continue  # 確保有足夠的欄位
            vpath, vid, _, _, vurl = r[:5]  # 忽略 motion_smoothness 和 dynamic_degree
            if not os.path.exists(vpath):
                skipped.append({"videoid": vid, "Imgurl": vurl, "camera_motion": "skipped"})
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
            writer.writerows([[row["videoid"], row["Imgurl"], row["camera_motion"]] for row in results])
    
    print(f"[DONE] → {OUT_FILE}\n[DONE] debug → {DBG_FILE}")

if __name__ == "__main__":
    main()