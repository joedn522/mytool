#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, time, argparse, subprocess
from multiprocessing import Process, Manager, Queue
import imageio_ffmpeg

# ───────────────────────── argparse ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True)
parser.add_argument("--output_path", required=True)
parser.add_argument("--max_video_processes", type=int, default=2)
parser.add_argument("--max_queue_size", type=int, default=20)
args = parser.parse_args()

# ───────────────────────── constants ────────────────────────
VBENCH_DIMS = ["motion_smoothness", "dynamic_degree"]
TMP_DIR     = "./tmp"
SENTINEL    = ("__DONE__", None, None)

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

OUT_FILE  = os.path.join(args.output_path, "output.txt")
DBG_FILE  = os.path.join(args.output_path, "debug.txt")

# ──────────────────── producer: convert mov → mp4 ────────────────────
def convert_to_mp4_worker(video_list, q: Queue, n_consumers: int):
    try:
        for vpath in video_list:
            print(f"[CONVERT]  ⇢ {vpath}", flush=True)

            if vpath.lower().endswith(".mov"):
                mp4_path = os.path.join(TMP_DIR,
                                        os.path.basename(vpath) + ".mp4")

                if not os.path.exists(mp4_path):
                    tic = time.time()
                    try:
                        subprocess.run(
                            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", vpath,
                             "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                             "-c:a", "aac", "-b:a", "128k", mp4_path],
                            check=True,
                            stdout=subprocess.DEVNULL,        # 不需擷取
                            stderr=subprocess.PIPE,           # 只拿錯誤
                        )
                        toc = time.time()
                        q.put((vpath, mp4_path, toc - tic))
                        print(f"[CONVERT]  ✓ {mp4_path}  {toc - tic:.2f}s",
                              flush=True)
                    except subprocess.CalledProcessError as e:
                        err = e.stderr.decode("utf-8", errors="ignore")
                        print(f"[CONVERT]  ✗ {vpath}\n{err}", flush=True)
                        q.put((vpath, None, None))
                else:
                    q.put((vpath, mp4_path, 0.0))
                    print(f"[CONVERT]  cache-hit {mp4_path}", flush=True)
            else:
                # 非 .mov 直接給 consumer
                q.put((vpath, vpath, 0.0))
                print(f"[CONVERT]  pass {vpath}", flush=True)

    finally:
        # 無論發生什麼事，都保證送出終止訊號
        for _ in range(n_consumers):
            q.put(SENTINEL)

# ────────────────── helper: run one vbench dim ───────────────────────
def run_vbench(mp4_path, dim, out_dir):
    tic = time.time()
    try:
        subprocess.run(
            ["vbench", "evaluate",
             "--ngpus", "1",
             "--videos_path", mp4_path,
             "--dimension", dim,
             "--mode", "custom_input",
             "--output_path", out_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return time.time() - tic
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        print(f"[VBENCH]   ✗ {mp4_path} ({dim})\n{err}", flush=True)
        return -1.0

# ───────────────────── consumer: process video ───────────────────────
def process_video_worker(q: Queue, results, dbg):
    while True:
        vpath, mp4, conv_t = q.get()      # blocking
        if vpath == "__DONE__":
            break
        if not mp4:                       # 轉檔失敗
            dbg.append(f"{vpath}\tconvert_failed")
            continue

        row = {"videoid": os.path.basename(vpath), "Imgurl": vpath}
        out_root = os.path.join(args.output_path, "evaluate_result")
        os.makedirs(out_root, exist_ok=True)

        times = {}
        for dim in VBENCH_DIMS:
            odir = os.path.join(out_root, dim)
            os.makedirs(odir, exist_ok=True)
            times[dim] = run_vbench(mp4, dim, odir)
            row[dim]   = times[dim]

        results.append(row)
        dbg.append(f"{vpath}\tconv:{conv_t:.2f}s\t"
                   f"motion:{times['motion_smoothness']:.2f}s\t"
                   f"dynamic:{times['dynamic_degree']:.2f}s")

        if mp4.startswith(TMP_DIR) and os.path.exists(mp4):
            os.remove(mp4)
            print(f"[CLEAN ]  {mp4}", flush=True)

# ─────────────────────────── main ───────────────────────────
def main():
    # 1. load list
    with open(args.input_tsv, newline="") as f:
        vids = [r[0] for r in csv.reader(f, delimiter="\t") if r]
    print(f"[MAIN   ]  loaded {len(vids)} videos")

    with Manager() as m:
        q       = m.Queue(maxsize=args.max_queue_size)
        results = m.list()
        dbg     = m.list()

        prod = Process(target=convert_to_mp4_worker,
                       args=(vids, q, args.max_video_processes))
        prod.start()

        workers = []
        for _ in range(args.max_video_processes):
            p = Process(target=process_video_worker, args=(q, results, dbg))
            p.start()
            workers.append(p)

        prod.join()
        for p in workers:
            p.join()

        # 2. write output
        fieldnames = ["videoid", "Imgurl"] + VBENCH_DIMS
        with open(OUT_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames, delimiter="\t").writerows(results)
        with open(DBG_FILE, "w") as f:
            f.write("\n".join(dbg))

    print(f"[DONE   ]  rows={len(results)}  →  {OUT_FILE}")
    print(f"[DONE   ]  debug → {DBG_FILE}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
