#!/usr/bin/env bash
# ============================================
# run_tarsier2.sh  ─  一鍵部署＋評估 Tarsier-2
# 用法：./run_tarsier2.sh <OUTPUT_DIR> [INPUT_DATA]
# --------------------------------------------
# 參數：
#   1. OUTPUT_DIR ＝ Aether 置換好的 ${Output1}
#   2. INPUT_DATA ＝ (選) Aether 置換好的 ${Input1}
# ============================================

set -euo pipefail

# ---------- 參數處理 ----------
OUT_DIR=${1:?「請傳入輸出資料夾 (Output1)」}
IN_DATA=${2:-}          # 有些 pipeline 會先替換好 ${Input1}，沒有就留空
LOG_FILE="$OUT_DIR/debug.txt"

mkdir -p "$OUT_DIR"

# ---------- 全局日誌重導 ----------
# 同步寫到檔案 ＋ 顯示到螢幕，方便線上 debug
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] ▶️  Tarsier-2 pipeline 開始執行"
echo "OUT_DIR = $OUT_DIR"
[[ -n "$IN_DATA" ]] && echo "IN_DATA = $IN_DATA"

# ---------- 環境變數 ----------
export PATH="$HOME/.local/bin:$PATH"

# ---------- 安裝與下載 ----------
if [[ ! -d "tarsier" ]]; then
  git clone --depth=1 --branch tarsier2 https://github.com/bytedance/tarsier.git
fi
cd tarsier

# setup.sh 有些步驟偶爾失敗不影響後續，因此用 || true
bash setup.sh || true

pip install -q flash-attn einops pyarrow 'accelerate>=0.26.0'

if [[ ! -d "checkpoints/tarsier2_full7b" ]]; then
  huggingface-cli download omni-research/Tarsier2-Recap-7b \
    --local-dir checkpoints/tarsier2_full7b
fi

wget -q -N https://raw.githubusercontent.com/joedn522/mytool/main/tarsier_eval_docker.py

# ---------- 執行評估 ----------
python tarsier_eval_docker.py \
  --data   "${IN_DATA:-$INPUT_DATA}" \
  --out    "$OUT_DIR/output.tsv" \
  --outlog "$OUT_DIR/debug_inner.txt" \
  --tarsier-checkpoints checkpoints/tarsier2_full7b \
  --sleep  0.5

echo "[$(date '+%F %T')] ✅  Pipeline 完成"
