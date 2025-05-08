import os
import subprocess
import csv
import argparse
import json
import time
from tqdm import tqdm
from multiprocessing import Pool, Semaphore, Manager, Queue

import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True, help="Input TSV file")
parser.add_argument("--output_path", required=True, help="Output directory (e.g., /vc_data/.../Output1)")
parser.add_argument("--max_video_processes", type=int, default=2, help="Maximum number of video processes")
parser.add_argument("--max_queue_size", type=int, default=20, help="Maximum number of videos in the queue")
args = parser.parse_args()

vbench_dimensions = ["motion_smoothness", "dynamic_degree"]
os.makedirs(args.output_path, exist_ok=True)
os.makedirs("./tmp", exist_ok=True)  # Local directory for converted mp4 files
output_file = os.path.join(args.output_path, "output.txt")
debug_file = os.path.join(args.output_path, "debug.txt")

# Semaphore to limit the number of concurrent video processes
video_semaphore = Semaphore(args.max_video_processes)

# 在 convert_to_mp4_worker 中增加日誌
def convert_to_mp4_worker(video_list, queue, max_queue_size):
    """Worker to convert videos to mp4."""
    for video_path in video_list:
        print(f"[DEBUG] Starting conversion for video: {video_path}")
        if video_path.endswith(".mov"):
            mp4_path = os.path.join("./tmp", os.path.basename(video_path) + ".mp4")
            if not os.path.exists(mp4_path):
                try:
                    start_time = time.time()
                    subprocess.run([
                        imageio_ffmpeg.get_ffmpeg_exe(), "-i", video_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-c:a", "aac", "-b:a", "128k", mp4_path
                    ], check=True)
                    conversion_time = time.time() - start_time
                    queue.put((video_path, mp4_path, conversion_time))
                    print(f"[DEBUG] Converted {video_path} to {mp4_path} in {conversion_time:.2f}s")
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed to convert {video_path}: {e}")
                    queue.put((video_path, None, None))
            else:
                queue.put((video_path, mp4_path, 0))  # Already converted
                print(f"[DEBUG] {mp4_path} already exists, skipping conversion.")
        else:
            print(f"[DEBUG] Skipping non-mov file: {video_path}")

def process_dimension(video_path, mp4_path, dim, dim_output):
    """Process a single dimension for a video."""
    start_time = time.time()
    try:
        subprocess.run([
            "vbench", "evaluate",
            "--ngpus", "1",
            "--videos_path", mp4_path,
            "--dimension", dim,
            "--mode", "custom_input",
            "--output_path", dim_output
        ], check=True)
        processing_time = time.time() - start_time
        return processing_time
    except subprocess.CalledProcessError:
        print(f"[ERROR] Failed to process dimension {dim} for video {video_path}")
        return None

# 在 process_video_worker 中增加日誌
def process_video_worker(queue, results, debug_logs):
    """Worker to process videos."""
    while not queue.empty() or not queue._closed:
        try:
            video_path, mp4_path, conversion_time = queue.get(timeout=5)
            print(f"[DEBUG] Processing video {video_path}, converted to {mp4_path}")
        except Exception as e:
            print(f"[DEBUG] Queue is empty or closed: {e}")
            continue  # 等待新的轉檔結果

        if not mp4_path:
            print(f"[DEBUG] Skipping video {video_path} due to failed conversion.")
            continue  # 跳過轉檔失敗的視頻

        print(f"[DEBUG] Starting dimension processing for video: {video_path}")

        score_row = {
            "videoid": os.path.basename(video_path),
            "Imgurl": video_path
        }

        # Process dimensions
        dim_output_base = os.path.join("evaluate_result")
        os.makedirs(dim_output_base, exist_ok=True)

        dim_times = {}
        with Pool(processes=2) as pool:
            dim_results = [
                pool.apply_async(process_dimension, (video_path, mp4_path, dim, os.path.join(dim_output_base, dim)))
                for dim in vbench_dimensions
            ]
            for dim, result in zip(vbench_dimensions, dim_results):
                try:
                    dim_times[dim] = result.get()
                    print(f"[DEBUG] Processed dimension {dim} for video {video_path} in {dim_times[dim]:.2f}s")
                except Exception as e:
                    print(f"[ERROR] Failed to process dimension {dim} for video {video_path}: {e}")
                    dim_times[dim] = -1

        # Collect results
        for dim in vbench_dimensions:
            score_row[dim] = dim_times.get(dim, -1)

        # Append results and debug logs
        results.append(score_row)
        debug_logs.append(f"Video Path: {video_path}, Conversion Time: {conversion_time:.2f}s, "
                          f"Motion Smoothness Time: {dim_times.get('motion_smoothness', -1):.2f}s, "
                          f"Dynamic Degree Time: {dim_times.get('dynamic_degree', -1):.2f}s")

        # 清理轉檔後的 mp4 文件
        if mp4_path.startswith("./tmp") and os.path.exists(mp4_path):
            os.remove(mp4_path)
            print(f"[DEBUG] Removed temporary file {mp4_path}")

def main():
    results = Manager().list()
    debug_logs = Manager().list()
    queue = Queue()

    # 讀取輸入文件
    with open(args.input_tsv, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        video_list = [row[0] for row in reader if len(row) >= 4]
        print(f"[DEBUG] Loaded {len(video_list)} videos from input file.")

    print(f"[DEBUG] Video list: {video_list}")
    # 啟動轉檔進程
    convert_process = Pool(processes=1)
    convert_process.apply_async(convert_to_mp4_worker, (video_list, queue, args.max_queue_size))
    print("[DEBUG] Started convert_to_mp4_worker process.")

    # 啟動處理進程
    process_pool = Pool(processes=args.max_video_processes)
    process_pool.apply_async(process_video_worker, (queue, results, debug_logs))
    print("[DEBUG] Started process_video_worker process.")

    convert_process.close()
    convert_process.join()
    process_pool.close()
    process_pool.join()

    # Write results to output file
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["videoid", "Imgurl"] + vbench_dimensions, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
        print(f"[DEBUG] Wrote results to {output_file}")

    # Write debug logs to debug file
    with open(debug_file, "w") as f:
        f.write("\n".join(debug_logs))
        print(f"[DEBUG] Wrote debug logs to {debug_file}")

    print(f"[FINISHED] Wrote {len(results)} rows to {output_file}")
    print(f"[DEBUG] Logs written to {debug_file}")

if __name__ == "__main__":
    main()