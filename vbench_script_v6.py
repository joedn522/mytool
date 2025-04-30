import os
import subprocess
import csv
import argparse
import json
import time
from tqdm import tqdm
import imageio_ffmpeg
import glob

def find_latest_result_file(dim_dir):
    pattern = os.path.join(dim_dir, "*_eval_results.json")
    result_files = glob.glob(pattern)
    if not result_files:
        return None
    result_files.sort(key=os.path.getmtime, reverse=True)
    return result_files[0]

parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True, help="Input TSV file with video_path, video_id, video_url")
args = parser.parse_args()
input_tsv = args.input_tsv

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

scores = {}
if os.path.exists(score_file):
    with open(score_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["video_id"]] = row

# Main loop
with open(input_tsv, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in tqdm(reader, desc="Processing videos"):
        time.sleep(2)
        if len(row) < 3:
            print(f"[SKIP] Malformed row: {row}")
            continue

        video_path, video_id, video_url = row[:3]
        if video_id in completed_tasks:
            continue
        if not os.path.exists(video_path):
            print(f"[WARN] File not found: {video_path}")
            continue

        print(f"\n=== Processing {video_id} ===")

        # Convert .mov to .mp4
        if video_path.endswith(".mov"):
            mp4_path = f"{video_path}.mp4"
            if not os.path.exists(mp4_path):
                try:
                    print(f"[INFO] Converting {video_path} to {mp4_path}")
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    subprocess.run([
                        ffmpeg_exe, "-i", video_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-c:a", "aac", "-b:a", "128k",
                        mp4_path
                    ], check=True)
                except subprocess.CalledProcessError:
                    print(f"[ERROR] FFmpeg conversion failed for {video_id}")
                    continue
            video_path = mp4_path

        score_row = {"video_id": video_id, "video_url": video_url}

        for dim in vbench_dimensions:
            dim_dir = os.path.join(output_dir, dim)
            os.makedirs(dim_dir, exist_ok=True)
            print(f"[INFO] Evaluating {dim} for {video_id}")

            try:
                subprocess.run([
                    "vbench", "evaluate",
                    "--ngpus", "1",
                    "--videos_path", video_path,
                    "--dimension", dim,
                    "--mode", "custom_input",
                    "--output_path", dim_dir
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"[ERROR] VBench failed for {video_id} - {dim}")
                score_row[dim] = "FAILED"
                continue

            # Load latest eval result
            try:
                result_file = find_latest_result_file(dim_dir)
                if not result_file:
                    raise FileNotFoundError("No *_eval_results.json file found.")
                with open(result_file, "r") as f:
                    result_json = json.load(f)
                if isinstance(result_json.get(dim), list) and len(result_json[dim]) > 1:
                    inner_list = result_json[dim][1]
                    if isinstance(inner_list, list) and len(inner_list) > 0 and isinstance(inner_list[0], dict):
                        score = inner_list[0].get("video_results", "N/A")
                        score_row[dim] = score
                        print(f"[INFO] Score for {video_id} - {dim}: {score}")
                    else:
                        print(f"[WARN] Unexpected inner structure for {video_id} - {dim}")
                        score_row[dim] = "N/A"
                else:
                    score_row[dim] = "N/A"
                    print(f"[WARN] Missing score data for {video_id} - {dim}")
            except Exception as e:
                print(f"[ERROR] Failed to load result for {video_id} - {dim}: {e}")
                score_row[dim] = "ERROR"

        # Write score
        scores[video_id] = score_row
        with open(score_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "video_url"] + vbench_dimensions)
            writer.writeheader()
            writer.writerows(scores.values())

        # Mark progress
        with open(progress_file, "a") as f:
            f.write(f"{video_id}\n")

        print(f"[DONE] {video_id}: {score_row}")
