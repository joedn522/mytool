import os
import importlib

# ------------------------------------------------------------------
# 1) **關掉單進程時的 torch.distributed.init_process_group**
#    => 避免重複監聽 29500
# ------------------------------------------------------------------
dist_mod = importlib.import_module("vbench.distributed")

def patched_dist_init():
    # 只要 WORLD_SIZE 沒設或 ==1，就直接返回
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return
    # 否則仍呼叫原本的 init_process_group
    return dist_mod._orig_dist_init()

# 先把原函式備份，然後替換
if not hasattr(dist_mod, "_orig_dist_init"):
    dist_mod._orig_dist_init = dist_mod.dist_init
    dist_mod.dist_init = patched_dist_init

# ------------------------------------------------------------------
# 2) 呼叫原生 evaluate CLI
# ------------------------------------------------------------------
from vbench.launch import evaluate
evaluate.main()