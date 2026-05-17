#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/reports/v4pro_w4a16_w4a8}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-$OUT_DIR/r6000_v4pro_w4a16_w4a8_generation_matrix_${STAMP}.json}"
LOG="${LOG:-$OUT_DIR/r6000_v4pro_w4a16_w4a8_generation_matrix_${STAMP}.log}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LYNN_MOE_IMPL="${LYNN_MOE_IMPL:-packed_nvfp4}"
export LYNN_NATIVE_FP4_LM_HEAD="${LYNN_NATIVE_FP4_LM_HEAD:-1}"
export LYNN_NATIVE_ACTIVE_MOE="${LYNN_NATIVE_ACTIVE_MOE:-1}"
export LYNN_NATIVE_ACTIVE_MOE_BACKEND="${LYNN_NATIVE_ACTIVE_MOE_BACKEND:-packed_nvfp4}"
export LYNN_ROUTER_TOPK_SORTED="${LYNN_ROUTER_TOPK_SORTED:-1}"
export LYNN_W4A8_FAKE_QUANT_FORMAT="${LYNN_W4A8_FAKE_QUANT_FORMAT:-e4m3}"
export LYNN_W4A8_FAKE_QUANT_GRANULARITY="${LYNN_W4A8_FAKE_QUANT_GRANULARITY:-per16}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

{
  echo "[r6000-v4-w4a16-w4a8] start $(date)"
  echo "[r6000-v4-w4a16-w4a8] model=$MODEL"
  if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "[r6000-v4-w4a16-w4a8][fail] missing model index: $MODEL" >&2
    exit 2
  fi
  "$PYTHON_BIN" benchmarks/v4_w4a16_w4a8_generation_matrix.py \
    --model "$MODEL" \
    --out "$OUT_JSON" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bf16}" \
    --max-new "${MAX_NEW:-48}" \
    --top-k "${TOP_K:-5}" \
    ${PROMPTS_FILE:+--prompts-file "$PROMPTS_FILE"} \
    ${USE_CHAT_TEMPLATE:+--use-chat-template}
  echo "[r6000-v4-w4a16-w4a8] report=$OUT_JSON"
  echo "[r6000-v4-w4a16-w4a8] done $(date)"
} 2>&1 | tee "$LOG"
