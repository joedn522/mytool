import csv
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

def run(args: Namespace):
    data_path = Path(args.data)
    out_path = Path(args.out)

    processed = set()
    if out_path.exists():
        with out_path.open(newline="") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts and len(parts) > 0:
                    processed.add(parts[0])  # vid

    # input: localpath, videoid, label (no header)
    with data_path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            localpath, videoid = row[0], row[1]
            # label = row[2]  # 可選，若要用 label 可加這行
            if videoid in processed:
                continue
            video_fp = Path(localpath)
            print(f"\n=== Processing {videoid} ===")

            # call Tarsier inference
            cmd = [
                sys.executable, "-m", "tasks.inference_quick_start",
                "--model_name_or_path", args.tarsier_checkpoints,
                "--instruction", "Describe the camera motion in detail.",
                "--input_path", str(video_fp),
                "--temperature", "0", "--top_p", "0"
            ]
            if args.device != "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
            try:
                pred = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print(e.output, file=sys.stderr)
                pred = "ERROR: inference failed"

            # crude extraction of the last ###Prediction block
            m = re.search(r"###Prediction:\s*(.*)\Z", pred, re.S)
            tarsier_text = m.group(1).strip() if m else pred.strip()

            # append to results (vid, file_path, tarsier_text) tab-separated, no header
            safe_text = tarsier_text.replace('\t', ' ').replace('\n', ' ')
            with out_path.open("a", newline="") as f_out:
                f_out.write(f"{videoid}\t{localpath}\t{safe_text}\n")
            processed.add(videoid)
            print(f"✓ Done {videoid}")

            time.sleep(args.sleep)

def cli() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--data", required=True, help="input TSV path (no header, localpath, videoid, ...)")
    p.add_argument("--out", required=True, help="results TSV path")
    p.add_argument("--tarsier-checkpoints", required=True)
    p.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between items")
    return p.parse_args()

if __name__ == "__main__":
    run(cli())