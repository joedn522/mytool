import os
import subprocess
import csv
import argparse
from tqdm import tqdm

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

if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        completed_tasks = set(line.strip() for line in f.readlines())
else:
    completed_tasks = set()

if not os.path.exists(score_file):
    with open(score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_url", "local_file"] + vbench_dimensions)

scores = {}
if os.path.exists(score_file):
    with open(score_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["video_id"]] = row

with open(input_tsv, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in tqdm(reader, desc="Processing videos"):
        if len(row) < 3:
            print(f"Skipping malformed row: {row}")
            continue

        video_path, video_id, video_url = row[:3]

        if video_id in completed_tasks:
            continue

        print(f"\n=== Processing {video_id} ===")
        print(f"Input file path: {video_path}")

        local_file = None
        for ext in [".mov", ".mp4"]:
            candidate_file = os.path.join(local_dir, f"{video_id}{ext}")
            if os.path.exists(candidate_file):
                local_file = candidate_file
                break

        if not local_file:
            print(f"[WARN] Local file not found for {video_id}, skipping...")
            continue

        if local_file.endswith(".mov"):
            mp4_file = f"{local_file}.mp4"
            if not os.path.exists(mp4_file):
                print(f"[INFO] Converting {local_file} ➜ {mp4_file} ...")
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", local_file,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    mp4_file,
                ]
                try:
                    subprocess.run(ffmpeg_command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] ffmpeg failed for {video_id}: {e}")
                    continue
            local_file = mp4_file

        score_row = {
            "video_id": video_id,
            "video_url": video_url,
            "local_file": local_file
        }

        for dim in vbench_dimensions:
            output_path = os.path.join(output_dir, dim)
            os.makedirs(output_path, exist_ok=True)
            result_file = os.path.join(output_path, f"{video_id}.json")

            if os.path.exists(result_file):
                print(f"[INFO] {dim} already exists for {video_id}, skipping...")
                with open(result_file, "r") as f:
                    score_row[dim] = f.read().strip()
                continue

            print(f"[INFO] Running vbench for {dim} on {video_id} ...")
            vbench_command = [
                "vbench",
                "evaluate",
                "--videos_path", local_file,
                "--dimension", dim,
                "--mode", "custom_input",
                "--output_path", output_path,
            ]
            try:
                subprocess.run(vbench_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] vbench failed for {video_id} - {dim}: {e}")
                score_row[dim] = "FAILED"
                continue

            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    score_row[dim] = f.read().strip()
            else:
                print(f"[WARN] No output found for {video_id} - {dim}")
                score_row[dim] = "N/A"

        scores[video_id] = score_row

        with open(score_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "video_url", "local_file"] + vbench_dimensions)
            writer.writeheader()
            writer.writerows(scores.values())

        with open(progress_file, "a") as progress:
            progress.write(f"{video_id}\n")
            progress.flush()

        print(f"[DONE] Completed {video_id}: {score_row}")

