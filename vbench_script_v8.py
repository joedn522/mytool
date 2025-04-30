import os
import subprocess
import csv
import argparse
import json
import time
from tqdm import tqdm
import imageio_ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--input_tsv", required=True, help="Input TSV file")
parser.add_argument("--output_path", required=True, help="Output directory (e.g., /vc_data/.../Output1)")
args = parser.parse_args()

vbench_dimensions = ["motion_smoothness", "dynamic_degree"]
os.makedirs(args.output_path, exist_ok=True)
output_file = os.path.join(args.output_path, "output.txt")

results = []
with open(args.input_tsv, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for idx, row in enumerate(tqdm(reader, desc="Processing videos")):
        if idx >= 3:  # 只處理前 3 筆
            break

        if len(row) < 3:
            continue
        video_path, video_id, video_url = row[:3]

        if not os.path.exists(video_path):
            print(f"[SKIP] Missing file: {video_path}")
            continue

        # Convert .mov to .mp4 if needed
        if video_path.endswith(".mov"):
            mp4_path = f"{video_path}.mp4"
            if not os.path.exists(mp4_path):
                try:
                    subprocess.run([
                        imageio_ffmpeg.get_ffmpeg_exe(), "-i", video_path,
                        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                        "-c:a", "aac", "-b:a", "128k", mp4_path
                    ], check=True)
                except subprocess.CalledProcessError:
                    print(f"[ERROR] Failed to convert {video_path}")
                    continue
            video_path = mp4_path

        score_row = {
            "videoid": video_id,
            "Imgurl": video_url
        }

        for dim in vbench_dimensions:
            dim_output = os.path.join(args.output_path, dim)
            os.makedirs(dim_output, exist_ok=True)

            try:
                subprocess.run([
                    "vbench", "evaluate",
                    "--ngpus", "1",
                    "--videos_path", video_path,
                    "--dimension", dim,
                    "--mode", "custom_input",
                    "--output_path", dim_output
                ], check=True)

                result_files = [f for f in os.listdir(dim_output) if f.endswith("_eval_results.json")]
                result_files.sort(key=lambda x: os.path.getmtime(os.path.join(dim_output, x)), reverse=True)
                latest = result_files[0]
                with open(os.path.join(dim_output, latest), "r") as jf:
                    j = json.load(jf)
                    score = j.get(dim, "N/A")
                    if isinstance(score, list):
                        score_row[dim] = score[1][0]["video_results"]
                    else:
                        score_row[dim] = score
            except Exception as e:
                score_row[dim] = "ERROR"

        results.append(score_row)

# Write output.txt
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["videoid", "Imgurl"] + vbench_dimensions)
    writer.writeheader()
    writer.writerows(results)

print(f"[FINISHED] Wrote {len(results)} rows to {output_file}")
