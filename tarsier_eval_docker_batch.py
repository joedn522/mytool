# fast_tarsier_eval.py  (Python ≥3.9)
from __future__ import annotations   

import csv, os, re, sys, time, yaml, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from itertools import cycle       

import torch
from tasks.utils import load_model_and_processor
from tasks.inference_quick_start import process_one


# ---------- util ----------
def safe_write(path: Path | None, txt: str):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with threading.Lock():
        with path.open("a", encoding="utf-8") as f:
            f.write(txt + "\n")

# ---------- init once ----------
def init_model(ckpt: str, cfg_path: str, device: str):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    model, processor = load_model_and_processor(ckpt, data_config=cfg)
    model.to(device).eval().half()
    return model, processor


# ---------- worker ----------
def run_one(video_fp: Path, vid: str,
            model, processor, device: str,
            prompt: str, out_path: Path, outlog: Path | None):

    start = time.time()
    gen_kwargs = dict(max_new_tokens=256, do_sample=False,
                      temperature=0, top_p=0, use_cache=True)

    try:
        with torch.cuda.device(device):
            pred_txt = process_one(model, processor, prompt, str(video_fp), gen_kwargs)
    except Exception as e:
        msg = f"[{vid}] ERROR: {e}"
        safe_write(outlog, msg) if outlog else None
        pred_txt = msg

    clean = pred_txt.replace('\t', ' ').replace('\n', ' ')
    safe_write(out_path, f"{vid}\t{video_fp}\t{clean}")

    elapsed = time.time() - start
    log = f"✓ Done {vid} | Time: {elapsed:.2f} s | Safe text: {clean}"
    print(log)
    safe_write(outlog, log) if outlog else None
    return vid


# ---------- main ----------
def main(args):
    # ① 解析裝置並設環境變數  ────────────────────────
    devices: List[str] = [d.strip() for d in args.device.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([d.split(":")[-1] for d in devices])

    out_path  = Path(args.out)
    outlog    = Path(args.outlog) if args.outlog else None
    data_rows = [row for row in csv.reader(Path(args.data).open(), delimiter="\t") if len(row) >= 2]

    done = {line.split("\t", 1)[0] for line in out_path.open() } if out_path.exists() else set()
    todo = [row for row in data_rows if row[1] not in done]

    print(f"Total {len(data_rows)} | Already {len(done)} | To-run {len(todo)}")

    # ── init model once ──
    devices: List[str] = args.device.split(",")  # 支援 cuda:0,cuda:1
    model, processor   = init_model(args.ckpt, args.config, devices[0])

    # ── pool ──
    prompt = "Describe the camera motion in detail."
    device_cycle = cycle(devices)            # 若只給一張卡就永遠取同一張

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut2vid = {
            pool.submit(run_one, Path(r[0]), r[1],
                        model, processor, next(device_cycle),
                        prompt, out_path, outlog): r[1]
            for r in todo
        }
        for fut in as_completed(fut2vid):
            fut.result()   # exceptions re-raised here


# ---------- CLI ----------
def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--outlog", default=None)
    ap.add_argument("--ckpt", required=True, help="tarsier checkpoint dir")
    ap.add_argument("--config", default="configs/tarser2_default_config.yaml")
    ap.add_argument("--device", default="cuda:0", help="cpu 或多卡 'cuda:0,cuda:1'")
    ap.add_argument("--workers", type=int, default=3, help="並行推理執行緒數")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_cli())
