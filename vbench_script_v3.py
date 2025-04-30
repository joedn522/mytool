import os
import subprocess
import csv
import argparse
import json
import time
from tqdm import tqdm
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True, help="Input TSV file with video_path, video_id, video_url")
args = parser.parse_args()
input_tsv = args.input_tsv

local_dir = "./tmp"
output_dir = "./evaluation_results"
progress_file = "progress.txt"
score_file = "scores.csv"
vbench_dimensions = ["motion_smoothness", "dynamic_degree"]

os.makedirs(output_dir, exist_ok=True)

# Load completed video IDs
completed_tasks = set()
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        completed_tasks = set(line.strip() for line in f)

# Prepare score file
if not os.path.exists(score_file):
    with open(score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_url"] + vbench_dimensions)

# Load previous scores
scores = {}
if os.path.exists(score_file):
    with open(score_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["video_id"]] = row

# Main processing loop
with open(input_tsv, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in tqdm(reader, desc="Processing videos"):
        time.sleep(10)  # let docker live longer for debugging
        if len(row) < 3:
            print(f"[SKIP] Malformed row: {row}")
            continue

        video_path, video_id, video_url = row[:3]

        if video_id in completed_tasks:
            continue

        print(f"\n=== Processing {video_id} === vvvvv")
        local_file = video_path  # use first column as the actual full video path

        if not os.path.exists(local_file):
            print(f"[WARN] No local file for {video_id}, skipping.")
            continue

        # Convert .mov to .mp4 using imageio_ffmpeg
        if local_file.endswith(".mov"):
            mp4_file = f"{local_file}.mp4"
            if not os.path.exists(mp4_file):
                print(f"[INFO] Converting {local_file} ➜ {mp4_file}")
                try:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    subprocess.run([
                        ffmpeg_exe, "-i", local_file,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-c:a", "aac", "-b:a", "128k",
                        mp4_file
                    ], check=True)
                except subprocess.CalledProcessError:
                    print(f"[ERROR] ffmpeg failed for {video_id}")
                    continue
            local_file = mp4_file

        score_row = {
            "video_id": video_id,
            "video_url": video_url
        }

        for dim in vbench_dimensions:
            dim_dir = os.path.join(output_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)
            print(f"[INFO] Running vbench for {video_id} - {dim}")

            try:
                subprocess.run([
                    "vbench", "evaluate",
                    "--videos_path", local_file,
                    "--dimension", dim,
                    "--mode", "custom_input",
                    "--output_path", dim_dir
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] vbench failed for {video_id} - {dim}")
                score_row[dim] = "FAILED"
                continue

            # Extract score
            try:
                result_file = os.path.join(dim_dir, "results_*.json")
                fallback_file = os.path.join(dim_dir, "eval_results.json")

                result_json = None
                if os.path.exists(fallback_file):
                    with open(fallback_file, "r") as f:
                        result_json = json.load(f)

                if result_json and dim in result_json and isinstance(result_json[dim], list):
                    if len(result_json[dim]) > 1 and isinstance(result_json[dim][1], dict):
                        score_row[dim] = result_json[dim][1].get("video_results", "N/A")
                    else:
                        print(f"[WARN] Score missing or malformed for {video_id} - {dim}")
                        score_row[dim] = "N/A"
                else:
                    print(f"[WARN] Result file not found or incomplete for {video_id} - {dim}")
                    score_row[dim] = "N/A"

            except Exception as e:
                print(f"[ERROR] Failed to parse result for {video_id} - {dim}: {e}")
                score_row[dim] = "ERROR"

        # Save result
        scores[video_id] = score_row
        with open(score_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "video_url"] + vbench_dimensions)
            writer.writeheader()
            writer.writerows(scores.values())

        # Mark progress
        with open(progress_file, "a") as f:
            f.write(f"{video_id}\n")

        print(f"[DONE] {video_id}: {score_row}")
