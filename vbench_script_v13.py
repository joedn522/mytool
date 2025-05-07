import os
import subprocess
import csv
import argparse
import json
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
        if len(row) < 4:
            continue
        video_path, video_id, vq_score, video_url = row[:4]

        # Convert vq_score to float
        try:
            vq_score = float(vq_score)
        except ValueError:
            print(f"[SKIP] Invalid vq_score for video {video_id}: {vq_score}")
            continue

        # Skip processing if vq_score > 0.3
        if vq_score > 0.3:
            print(f"[SKIP] High vq_score ({vq_score}) for video {video_id}, skipping ffmpeg and vbench.")
            score_row = {
                "videoid": video_id,
                "Imgurl": video_url,
                "motion_smoothness": -1,
                "dynamic_degree": -1
            }
            results.append(score_row)
            continue

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
            dim_output = os.path.join("evaluate_result", dim)
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
                    score = j.get(dim, -1)
                    if isinstance(score, list):
                        if dim == "dynamic_degree":
                            video_result = score[1][0].get("video_results", -1)
                            score_row[dim] = float(video_result) if video_result != -1 else -1
                        else:
                            score_row[dim] = score[1][0].get("video_results", -1)
                    else:
                        score_row[dim] = float(score) if score != -1 else -1
            except Exception as e:
                print(f"[ERROR] Failed to process dimension {dim} for video {video_id}: {e}")
                score_row[dim] = -1

        results.append(score_row)

# Write output.txt
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["videoid", "Imgurl"] + vbench_dimensions, delimiter="\t")
    writer.writerows(results)

print(f"[FINISHED] Wrote {len(results)} rows to {output_file}")