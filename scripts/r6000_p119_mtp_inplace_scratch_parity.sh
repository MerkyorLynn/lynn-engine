#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/reports/p119}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-$OUT_DIR/p119_mtp_inplace_scratch_parity_r6000_${STAMP}.json}"
LOG="${LOG:-$OUT_DIR/p119_mtp_inplace_scratch_parity_r6000_${STAMP}.log}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LYNN_NATIVE_FP4_LM_HEAD="${LYNN_NATIVE_FP4_LM_HEAD:-1}"
export LYNN_NATIVE_ACTIVE_MOE="${LYNN_NATIVE_ACTIVE_MOE:-1}"
export LYNN_NATIVE_ACTIVE_MOE_BACKEND="${LYNN_NATIVE_ACTIVE_MOE_BACKEND:-packed_nvfp4}"
export LYNN_ROUTER_TOPK_SORTED="${LYNN_ROUTER_TOPK_SORTED:-1}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

{
  echo "[r6000-p119] start $(date)"
  echo "[r6000-p119] model=$BASE_MODEL"
  echo "[r6000-p119] max_events=${MAX_EVENTS:-8} max_seq_len=${MAX_SEQ_LEN:-4096} dtype=${DTYPE:-bfloat16}"
  "$PYTHON_BIN" benchmarks/p119_mtp_inplace_scratch_parity.py \
    --base-model "$BASE_MODEL" \
    --out "$OUT_JSON" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" \
    --max-seq-len "${MAX_SEQ_LEN:-4096}" \
    --max-events "${MAX_EVENTS:-8}" \
    ${PROMPTS_FILE:+--prompts-file "$PROMPTS_FILE"} \
    ${USE_CHAT_TEMPLATE:+--use-chat-template}
  echo "[r6000-p119] report=$OUT_JSON"
  echo "[r6000-p119] done $(date)"
} 2>&1 | tee "$LOG"
