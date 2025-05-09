#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import subprocess
import argparse
from multiprocessing import Process, Manager, Queue
import imageio_ffmpeg

###############################################################################
# 參數解析
###############################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True,
                    help="Input TSV file：第一欄為影片路徑")
parser.add_argument("--output_path", required=True,
                    help="輸出資料夾 (e.g., /vc_data/.../Output1)")
parser.add_argument("--max_video_processes", type=int, default=2,
                    help="處理 vbench 的並行 consumer 數")
parser.add_argument("--max_queue_size", type=int, default=20,
                    help="Queue 最大長度（生產端背壓）")
args = parser.parse_args()

###############################################################################
# 全域常量
###############################################################################
VBENCH_DIMS   = ["motion_smoothness", "dynamic_degree"]
TMP_DIR       = "./tmp"
SENTINEL      = ("__DONE__", None, None)   # queue 終止訊號

os.makedirs(args.output_path, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

OUTPUT_FILE   = os.path.join(args.output_path, "output.txt")
DEBUG_FILE    = os.path.join(args.output_path, "debug.txt")

###############################################################################
# 轉檔 Worker
###############################################################################
def convert_to_mp4_worker(video_list, q: Queue, num_consumers: int):
    """
    將 .mov 轉 mp4；其餘格式直接送入 queue。
    每支影片轉檔完成 (或直接可用) 後：
        q.put((原路徑, mp4路徑, 轉檔耗時))
    最後送出 num_consumers 個 SENTINEL。
    """
    for vpath in video_list:
        print(f"[CONVERT]  start  {vpath}", flush=True)

        # .mov 需要轉檔；其他格式直接丟給後續流程
        if vpath.lower().endswith(".mov"):
            mp4_path = os.path.join(TMP_DIR,
                                    os.path.basename(vpath) + ".mp4")

            if not os.path.exists(mp4_path):
                tic = time.time()
                try:
                    proc = subprocess.run(
                        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", vpath,
                         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                         "-c:a", "aac", "-b:a", "128k", mp4_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    toc = time.time()
                    q.put((vpath, mp4_path, toc - tic))
                    print(f"[CONVERT]  ✓ {vpath} -> {mp4_path} "
                          f"{toc - tic:.2f}s", flush=True)
                except subprocess.CalledProcessError as e:
                    # 失敗仍送進 queue，讓 consumer 記 log 後略過
                    q.put((vpath, None, None))
                    print(f"[CONVERT]  ✗ {vpath} ffmpeg error\n{e.stderr}",
                          flush=True)
            else:
                # 已經轉過
                q.put((vpath, mp4_path, 0.0))
                print(f"[CONVERT]  skip (cached) {mp4_path}", flush=True)
        else:
            # 不需轉檔
            q.put((vpath, vpath, 0.0))
            print(f"[CONVERT]  pass-through {vpath}", flush=True)

    # 轉檔全部結束，送出終止訊號
    for _ in range(num_consumers):
        q.put(SENTINEL)

###############################################################################
# vbench 單一 dimension
###############################################################################
def run_vbench_dimension(mp4_path: str, dim: str, output_dir: str) -> float:
    tic = time.time()
    try:
        subprocess.run(
            ["vbench", "evaluate",
             "--ngpus", "1",
             "--videos_path", mp4_path,
             "--dimension", dim,
             "--mode", "custom_input",
             "--output_path", output_dir],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return time.time() - tic
    except subprocess.CalledProcessError as e:
        print(f"[VBENCH]   ✗ {mp4_path} ({dim})\n{e.stderr}", flush=True)
        return -1.0

###############################################################################
# Consumer Worker
###############################################################################
def process_video_worker(q: Queue, results, debug_logs):
    """
    循環取 queue，碰到 SENTINEL 就結束。
    """
    while True:
        video_path, mp4_path, conv_time = q.get()  # blocking
        if video_path == "__DONE__":
            break

        if not mp4_path:          # 轉檔失敗
            debug_logs.append(f"{video_path}\tconversion_failed")
            continue

        print(f"[PROCESS]  {video_path}", flush=True)

        score_row = {"videoid": os.path.basename(video_path),
                     "Imgurl":  video_path}

        # 依序跑兩個 dimension
        dim_times = {}
        out_base  = os.path.join(args.output_path, "evaluate_result")
        os.makedirs(out_base, exist_ok=True)

        for dim in VBENCH_DIMS:
            dim_dir            = os.path.join(out_base, dim)
            os.makedirs(dim_dir, exist_ok=True)
            dim_times[dim]     = run_vbench_dimension(mp4_path, dim, dim_dir)
            score_row[dim]     = dim_times[dim]

        results.append(score_row)
        debug_logs.append(
            f"{video_path}\tconv:{conv_time:.2f}s\t"
            f"motion:{dim_times['motion_smoothness']:.2f}s\t"
            f"dynamic:{dim_times['dynamic_degree']:.2f}s"
        )

        # 若是暫存檔，刪掉
        if mp4_path.startswith(TMP_DIR) and os.path.exists(mp4_path):
            os.remove(mp4_path)
            print(f"[CLEAN ]  remove {mp4_path}", flush=True)

###############################################################################
# 主程式
###############################################################################
def main():
    # 1. 讀取 TSV
    with open(args.input_tsv, "r") as f:
        reader      = csv.reader(f, delimiter="\t")
        video_list  = [row[0] for row in reader if row]  # 取第一欄
    print(f"[MAIN   ]  loaded {len(video_list)} videos")

    # 2. 建立共享結構
    with Manager() as m:
        q          = m.Queue(maxsize=args.max_queue_size)
        results    = m.list()
        debug_logs = m.list()

        # 3. 啟動轉檔 (Producer)
        producer = Process(target=convert_to_mp4_worker,
                           args=(video_list, q, args.max_video_processes))
        producer.start()

        # 4. 啟動多個 Consumer
        consumers = []
        for _ in range(args.max_video_processes):
            p = Process(target=process_video_worker,
                        args=(q, results, debug_logs))
            p.start()
            consumers.append(p)

        # 5. 等待所有行程結束
        producer.join()
        for p in consumers:
            p.join()

        # 6. 輸出結果
        fieldnames = ["videoid", "Imgurl"] + VBENCH_DIMS
        with open(OUTPUT_FILE, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames,
                                    delimiter="\t")
            writer.writerows(results)
        with open(DEBUG_FILE, "w") as f_dbg:
            f_dbg.write("\n".join(debug_logs))

    print(f"[DONE   ]  wrote {len(results)} rows -> {OUTPUT_FILE}")
    print(f"[DONE   ]  debug log -> {DEBUG_FILE}")

###############################################################################
if __name__ == "__main__":
    main()
