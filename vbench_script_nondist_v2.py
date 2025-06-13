#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量評估 motion_smoothness / dynamic_degree
------------------------------------------------
⚙  python vbench_batch_driver.py \
        --input_tsv videos.tsv \
        --output_path ./batch_out \
        --batch_size 50 \
        --skip_conversion
"""
import os, csv, time, json, argparse, subprocess, tempfile, contextlib, socket
from multiprocessing import Process, Manager, Queue
import imageio_ffmpeg

# ─────────── CLI ───────────
P = argparse.ArgumentParser()
P.add_argument("--input_tsv", required=True)
P.add_argument("--output_path", required=True)
P.add_argument("--batch_size", type=int, default=50)
P.add_argument("--max_video_processes", type=int, default=1)
P.add_argument("--max_queue_size", type=int, default=20)
P.add_argument("--skip_conversion", action="store_true")
args = P.parse_args()

VBENCH_DIMS = ["motion_smoothness", "dynamic_degree"]
TMP_DIR     = "./tmp"
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
OUT_FILE = os.path.join(args.output_path, "output.txt")
DBG_FILE = os.path.join(args.output_path, "debug.txt")

# ─────────── 轉檔 Producer ───────────
def convert_worker(task_list, q: Queue, n_consumer):
    try:
        for vpath, vid, vurl in task_list:
            mp4 = vpath
            if (not args.skip_conversion) and vpath.lower().endswith(".mov"):
                mp4 = os.path.join(TMP_DIR, f"{vid}.mp4")
                if not os.path.exists(mp4):
                    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-i", vpath,
                           "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                           "-c:a", "aac", "-b:a", "128k", mp4]
                    try:
                        subprocess.run(cmd, check=True,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                    except subprocess.CalledProcessError:
                        mp4 = None
            q.put((vpath, vid, vurl, mp4))
    finally:
        for _ in range(n_consumer):
            q.put(("__DONE__", None, None, None))

# ---------- 呼叫 VBench（逐-dimension 版本．修正版） ----------
def run_vbench_batch(rows, odir, batch_idx):
    """
    rows : [(mp4_path, video_id, url), ...]
    回傳 : {video_id: {"motion_smoothness": s, "dynamic_degree": d}}
    """
    batch_tsv = os.path.join(odir, f"batch_{batch_idx:03d}.tsv")
    with open(batch_tsv, "w") as f:
        for mp4, vid, url in rows:
            # ⇩⇩⇩ 轉成絕對路徑 ⇩⇩⇩
            print("\t".join([os.path.abspath(mp4), vid, "", url]), file=f)

    name2vid = {os.path.abspath(mp4): vid for mp4, vid, _ in rows}
    out = {vid: {d: -1 for d in VBENCH_DIMS} for _, vid, _ in rows}

    for dim in VBENCH_DIMS:
        dim_out = os.path.join(odir, dim)
        os.makedirs(dim_out, exist_ok=True)

        cmd = [
            "python", "./evaluate_safe.py",
            "--videos_path", batch_tsv,
            "--dimension", dim,
            "--mode", "custom_input",
            "--output_path", dim_out,
        ]
        env = os.environ.copy()
        env["HUB_NO_GIT"] = "1"
        env["RANK"] = "0"          # ← 加這行，確保 get_rank()==0

        err_path = os.path.join(dim_out,
                                f"{dim}_batch_{batch_idx:03d}.err")
        try:
            with open(err_path, "w") as ef:                # ← 第二個 open（新增）
                subprocess.run(cmd, env=env, check=True,
                               stdout=subprocess.DEVNULL,
                               stderr=ef)
        except subprocess.CalledProcessError:
            # 這個維度整批失敗，維持 -1
            continue

        # 讀最新 json
        jfile = max((p for p in os.listdir(dim_out)
                     if p.endswith("_eval_results.json")),
                    key=lambda p: os.path.getmtime(os.path.join(dim_out, p)))
        with open(os.path.join(dim_out, jfile)) as jf:
            data = json.load(jf)[dim]

        # -------- ① 純數值：整批同分 --------
        if isinstance(data, (int, float)):
            for _, vid, _ in rows:
                out[vid][dim] = float(data)
            continue

        # -------- ② list 結構 --------
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            for itm in data[1]:
                vpath = os.path.abspath(itm["video_path"])
                vid   = name2vid.get(vpath)
                if vid:
                    out[vid][dim] = itm.get("video_results",
                                            itm.get("video_score", -1))
            continue

        # -------- ③ 其他型別：保持 -1 --------
        #（可在此處 print debug 資訊以便日後擴充）
    return out


# ─────────── Consumer (批次 flush) ───────────
def consumer(q: Queue, results, dbg, total_tasks):
    def log(m): print(m, file=open(dbg, "a"), flush=True)

    bucket, batch_idx, done = [], 0, 0
    start = time.time()

    def flush():
        nonlocal bucket, batch_idx, done
        if not bucket: return
        batch_idx += 1
        odir = os.path.join(args.output_path, "evaluate_result",
                            f"batch_{batch_idx:03d}")
        os.makedirs(odir, exist_ok=True)
        try:
            preds = run_vbench_batch(bucket, odir, batch_idx)
        except subprocess.CalledProcessError as e:
            preds = {vid: {d: -1} for _, vid, _ in bucket}
            log(f"[BATCH_FAIL] {batch_idx:03d} rc={e.returncode}")

        for mp4, vid, url in bucket:
            row = {"videoid": vid, "Imgurl": url}
            row.update(preds.get(vid, {d: -1 for d in VBENCH_DIMS}))
            results.append(row)
            with open(OUT_FILE, "a", newline="") as f:
                csv.DictWriter(f, ["videoid", "Imgurl"]+VBENCH_DIMS,
                               delimiter="\t").writerow(row)
            done += 1
            print(f"[PROGRESS] {done}/{total_tasks} {vid}", flush=True)
            if mp4.startswith(TMP_DIR) and os.path.exists(mp4):
                os.remove(mp4)
        bucket = []

    while True:
        vpath, vid, url, mp4 = q.get()
        if vpath == "__DONE__":
            flush(); break
        if not mp4:
            results.append({"videoid": vid, "Imgurl": url,
                            **{d: -1 for d in VBENCH_DIMS}})
            continue
        bucket.append((mp4, vid, url))
        if len(bucket) >= args.batch_size:
            flush()

    print(f"[DONE] {done} vids in {time.time()-start:.1f}s", flush=True)

# ─────────── main ───────────
def main():
    # ── 補 resume ──
    processed = set()
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            for r in csv.reader(f, delimiter="\t"):
                processed.add(r[0])
    tasks, skipped = [], []
    with open(args.input_tsv) as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) < 4: continue
            vpath, vid, vq, url = r[:4]
            try: vq = float(vq)
            except: continue
            if vq > .3 or not os.path.exists(vpath):
                skipped.append({"videoid": vid, "Imgurl": url,
                                **{d:-1 for d in VBENCH_DIMS}})
                continue
            if vid in processed: continue
            tasks.append((vpath, vid, url))
    print(f"[MAIN] todo={len(tasks)} skipped={len(skipped)}")

    with Manager() as m:
        q   = m.Queue(args.max_queue_size)
        res = m.list(skipped)
        open(DBG_FILE, "w").close()

        prod = Process(target=convert_worker,
                       args=(tasks, q, args.max_video_processes))
        prod.start()
        cons = [Process(target=consumer,
                        args=(q, res, DBG_FILE, len(tasks)))
                for _ in range(args.max_video_processes)]
        for c in cons: c.start()
        prod.join(); [c.join() for c in cons]

    print(f"[DONE] -> {OUT_FILE}")

if __name__ == "__main__":
    main()
