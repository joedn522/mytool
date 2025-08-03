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
    outlog_path = Path(args.outlog) if hasattr(args, "outlog") and args.outlog else None

    processed = set()
    if out_path.exists():
        with out_path.open(newline="") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts and len(parts) > 0:
                    processed.add(parts[0])  # vid

    # input: localpath, videoid, label (no header)
    all_rows = []
    with data_path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            all_rows.append(row)
    total_rows = len(all_rows)
    already_processed = sum(1 for row in all_rows if row[1] in processed)
    to_process = total_rows - already_processed

    def log_write(msg):
        if outlog_path:
            with outlog_path.open("a", encoding="utf-8") as flog:
                flog.write(msg + "\n")

    log_write(f"Total rows in data_path: {total_rows}")
    log_write(f"Already processed (in out_path): {already_processed}")
    log_write(f"To process this run: {to_process}")

    print(f"Total rows in data_path: {total_rows}")
    print(f"Already processed (in out_path): {already_processed}")
    print(f"To process this run: {to_process}")

    for idx, row in enumerate(all_rows, 1):
        localpath, videoid = row[0], row[1]
        if videoid in processed:
            continue
        video_fp = Path(localpath)
        msg = f"\n=== Processing {videoid} ({idx}/{total_rows}) ==="
        print(msg)
        log_write(msg)
        start_time = time.time()

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
            log_write(f"[{videoid}] STDOUT:\n{pred}")
        except subprocess.CalledProcessError as e:
            print(e.output, file=sys.stderr)
            log_write(f"[{videoid}] ERROR:\n{e.output}")
            pred = "ERROR: inference failed"

        # crude extraction of the last ###Prediction block
        m = re.search(r"###Prediction:\s*(.*)\Z", pred, re.S)
        tarsier_text = m.group(1).strip() if m else pred.strip()

        # append to results (vid, file_path, tarsier_text) tab-separated, no header
        safe_text = tarsier_text.replace('\t', ' ').replace('\n', ' ')
        with out_path.open("a", newline="") as f_out:
            f_out.write(f"{videoid}\t{localpath}\t{safe_text}\n")
        processed.add(videoid)

        elapsed = time.time() - start_time
        msg = f"✓ Done {videoid} | Time: {elapsed:.2f} sec\nSafe text: {safe_text}"
        print(msg)
        log_write(msg)

        time.sleep(args.sleep)

def cli() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--data", required=True, help="input TSV path (no header, localpath, videoid, ...)")
    p.add_argument("--out", required=True, help="results TSV path")
    p.add_argument("--tarsier-checkpoints", required=True)
    p.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between items")
    p.add_argument("--outlog", required=False, help="log file path (append log line by line)")
    return p.parse_args()

if __name__ == "__main__":
    run(cli())