#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, time, json, argparse, subprocess
from multiprocessing import Process, Manager, Queue
import imageio_ffmpeg

# ───────────────────────── argparse ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--max_video_processes", type=int, default=4)
parser.add_argument("--max_queue_size", type=int, default=20)
args = parser.parse_args()

# ───────────────────────── constants ────────────────────────
VBENCH_DIMS = ["motion_smoothness", "dynamic_degree"]
TMP_DIR     = "./tmp"
SENTINEL    = ("__DONE__", None, None, None, None)   # queue item 長度

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

OUT_FILE = os.path.join(args.output_path, "output.txt")
DBG_FILE = os.path.join(args.output_path, "debug.txt")

# ──────────────────── producer: convert mov → mp4 ────────────────────
def convert_to_mp4_worker(task_list, q: Queue, n_consumers: int):
    """
    task_list: (orig_path, video_id, video_url)
    出隊列:   (orig_path, video_id, video_url, mp4_path, conv_time)
    """
    try:
        for vpath, vid, vurl in task_list:
            mp4_path, conv_t = vpath, 0.0
            if vpath.lower().endswith(".mov"):
                videoid = os.path.splitext(vpath.replace("/", "_"))[0]
                mp4_path = os.path.join(TMP_DIR, f"{videoid}.mp4")

                if not os.path.exists(mp4_path):
                    tic = time.time()
                    try:
                        subprocess.run(
                            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", vpath,
                             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                             "-c:a", "aac", "-b:a", "128k", mp4_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                        )
                        conv_t = time.time() - tic
                        print(f"[CONVERT]  ✓ {mp4_path}  {conv_t:.2f}s", flush=True)
                    except subprocess.CalledProcessError as e:
                        err = e.stderr.decode("utf-8", errors="ignore")
                        print(f"[CONVERT]  ✗ {vpath}\n{err}", flush=True)
                        mp4_path = None
                else:
                    print(f"[CONVERT]  cache-hit {mp4_path}", flush=True)
            else:
                print(f"[CONVERT]  pass {vpath}", flush=True)

            q.put((vpath, vid, vurl, mp4_path, conv_t))
    finally:
        for _ in range(n_consumers):
            q.put(SENTINEL)

# ────────────────── helper: run one vbench dim ───────────────────────
def run_vbench(mp4_path, dim, odir):
    """
    回傳 (score, elapsed_time)
    """
    tic = time.time()
    try:
        subprocess.run(
            ["vbench", "evaluate",
             "--ngpus", "1",
             "--videos_path", mp4_path,
             "--dimension", dim,
             "--mode", "custom_input",
             "--output_path", odir],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        print(f"[VBENCH]   ✗ {mp4_path} ({dim})\n{err}", flush=True)
        return -1, -1.0

    et = time.time() - tic

    # 讀最新 *_eval_results.json
    json_files = [f for f in os.listdir(odir) if f.endswith("_eval_results.json")]
    if not json_files:
        return -1, et
    json_files.sort(key=lambda f: os.path.getmtime(os.path.join(odir, f)), reverse=True)
    latest = os.path.join(odir, json_files[0])

    try:
        with open(latest, "r") as jf:
            j = json.load(jf)
        score = j.get(dim, -1)
        if isinstance(score, list):
            if dim == "dynamic_degree":
                vres = score[1][0].get("video_results", -1)
                score = float(vres) if vres != -1 else -1
            else:
                score = score[1][0].get("video_results", -1)
        else:
            score = float(score) if score != -1 else -1
    except Exception as e:
        print(f"[VBENCH]   ✗ parse {latest}: {e}", flush=True)
        score = -1
    return score, et

# ───────────────────── consumer: process video ───────────────────────
def process_video_worker(q: Queue, results, dbg_file_path):
    def log(msg: str):
        with open(dbg_file_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    while True:
        vpath, vid, vurl, mp4, conv_t = q.get()
        if vpath == "__DONE__":
            break
        if not mp4:
            log(f"{vid}\tconvert_failed")
            results.append({"videoid": vid, "Imgurl": vurl,
                            "motion_smoothness": -1, "dynamic_degree": -1})
            continue

        out_root = os.path.join(args.output_path, "evaluate_result")
        os.makedirs(out_root, exist_ok=True)
        row = {"videoid": vid, "Imgurl": vurl}

        for dim in VBENCH_DIMS:
            odir = os.path.join(out_root, dim)
            os.makedirs(odir, exist_ok=True)
            score, elapsed = run_vbench(mp4, dim, odir)
            row[dim] = score
            log(f"{vid}\t{dim}:{score}\t{elapsed:.2f}s")

        results.append(row)

        if mp4.startswith(TMP_DIR) and os.path.exists(mp4):
            os.remove(mp4)
            log(f"[CLEAN ]  {mp4}")

# ─────────────────────────── main ───────────────────────────
def main():
    tasks, skip_rows = [], []
    with open(args.input_tsv, newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4:
                continue
            vpath, vid, vq, vurl = row[:4]
            try:
                vq = float(vq)
            except ValueError:
                continue
            if vq > 0.3 or not os.path.exists(vpath):
                skip_rows.append({"videoid": vid, "Imgurl": vurl,
                                  "motion_smoothness": -1, "dynamic_degree": -1})
                continue
            tasks.append((vpath, vid, vurl))

    print(f"[MAIN   ]  tasks={len(tasks)} skipped={len(skip_rows)}")

    with Manager() as m:
        q = m.Queue(maxsize=args.max_queue_size)
        results = m.list(skip_rows)

        open(DBG_FILE, "w").close()   # 清空 debug

        prod = Process(target=convert_to_mp4_worker, args=(tasks, q, args.max_video_processes))
        prod.start()

        workers = []
        for _ in range(args.max_video_processes):
            p = Process(target=process_video_worker, args=(q, results, DBG_FILE))
            p.start()
            workers.append(p)

        prod.join()
        for p in workers:
            p.join()

        rows = list(results)
        fieldnames = ["videoid", "Imgurl"] + VBENCH_DIMS
        with open(OUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    print(f"[DONE   ]  rows={len(rows)}  →  {OUT_FILE}")
    print(f"[DONE   ]  debug → {DBG_FILE}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
