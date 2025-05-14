import pickle
import torch
import torch.serialization
import os
from collections import OrderedDict

# ------------------------------------------------------------------
# 1) 讓 torch.load() 自動 fallback  →  你原本就有
# ------------------------------------------------------------------
torch.serialization.add_safe_globals([OrderedDict])

_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    try:
        return _original_torch_load(*args, **kwargs)
    except pickle.UnpicklingError:
        print("[Patch] Caught UnpicklingError, retrying torch.load with weights_only=False")
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

# ------------------------------------------------------------------
# 2) **關掉單進程時的 torch.distributed.init_process_group**
#    => 避免重複監聽 29500
# ------------------------------------------------------------------
import importlib, types
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
    dist_mod.dist_init      = patched_dist_init

# ------------------------------------------------------------------
# 3) 呼叫原生 evaluate CLI
# ------------------------------------------------------------------
from vbench.launch import evaluate
evaluate.main()
