# fast_tarsier_eval.py  (Python ≥3.9)
from __future__ import annotations   

import csv, os, re, sys, time, yaml, argparse, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from itertools import cycle   
import multiprocessing as mp    

import torch
from tasks.utils import load_model_and_processor
from tasks.inference_quick_start import process_one


# ---------- util ----------
def safe_write(path: Path | None, txt: str, lock=None):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(txt + "\n")
                f.flush()
                os.fsync(f.fileno())
    else:
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
            prompt: str, out_path: Path, outlog: Path | None,
            max_new_tokens: int = 256, file_lock=None):

    start = time.time()
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      temperature=0, top_p=0, use_cache=True)

    try:
        with torch.cuda.device(device):
            pred_txt = process_one(model, processor, prompt, str(video_fp), gen_kwargs)
    except Exception as e:
        msg = f"[{vid}] ERROR: {e}"
        safe_write(outlog, msg) if outlog else None
        pred_txt = msg

    clean = pred_txt.replace('\t', ' ').replace('\n', ' ')
    safe_write(out_path, f"{vid}\t{video_fp}\t{clean}", file_lock)

    elapsed = time.time() - start
    log = f"✓ Done {vid} | Time: {elapsed:.2f} s | Safe text: {clean}"
    print(log)
    safe_write(outlog, log, file_lock) if outlog else None
    return vid


def init_multiple_models(ckpt: str, cfg_path: str, device: str, num_models: int):
    models = []
    processors = []
    for i in range(num_models):
        cfg = yaml.safe_load(open(cfg_path, "r"))
        model, processor = load_model_and_processor(ckpt, data_config=cfg)
        model.to(device).eval().half()
        models.append(model)
        processors.append(processor)
        print(f"[Model {i}] Initialized on {device}, memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    return models, processors

# ---------- worker ----------
def worker_run(task_queue, device, ckpt, config, prompt, out_path, outlog, max_new_tokens, file_lock):
    model, processor = init_model(ckpt, config, device)
    while not task_queue.empty():
        try:
            video_fp, vid = task_queue.get_nowait()
        except:
            break
        run_one(video_fp, vid, model, processor, device, prompt, out_path, outlog, max_new_tokens, file_lock)

# ---------- main ----------
def main(args):
    import time
    start_time = time.time()

    # 印出 input parameter
    print("\n=== Input Parameters ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # 讀取 data_config 並印出
    cfg = yaml.safe_load(open(args.config, "r"))
    print("\n=== data_config ===")
    print(yaml.dump(cfg, sort_keys=False))

    devices: List[str] = [d.strip() for d in args.device.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([d.split(":")[-1] for d in devices])
    out_path = Path(args.out)
    outlog = Path(args.outlog) if args.outlog else None

    data_rows = [row for row in csv.reader(Path(args.data).open(), delimiter="\t") if len(row) >= 2]
    done = {line.split("\t", 1)[0] for line in out_path.open()} if out_path.exists() else set()
    todo = [row for row in data_rows if row[1] not in done]
    print(f"Total {len(data_rows)} | Already {len(done)} | To-run {len(todo)}")

    # ── 初始化任务队列 ──
    task_queue = mp.Queue()
    for row in todo:
        task_queue.put((Path(row[0]), row[1]))

    # ── 启动多个进程，每个进程加载自己的模型 ──
    prompt = "Describe the camera motion in detail."
    num_workers = min(args.workers, len(todo))
    processes = []
    file_lock = mp.Lock()

    for i in range(num_workers):
        p = mp.Process(
            target=worker_run,
            args=(task_queue, devices[0], args.ckpt, args.config,
                  prompt, out_path, outlog, args.max_new_tokens, file_lock)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 統計總時間與平均
    total_time = time.time() - start_time
    entry_count = len(todo)
    avg_time = total_time / entry_count if entry_count > 0 else 0

    # 再次印出 input parameter 和 data_config
    print("\n=== Input Parameters (End) ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    print("\n=== data_config (End) ===")
    print(yaml.dump(cfg, sort_keys=False))

    print(f"\n=== Summary ===")
    print(f"Processed entries: {entry_count}")
    print(f"Total elapsed time: {total_time:.2f} seconds")
    print(f"Average time per video: {avg_time:.2f} seconds")

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
    ap.add_argument("--max_new_tokens", type=int, default=256, help="生成最大 token 數")
    return ap.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # 已設定過就略過
    main(parse_cli())
