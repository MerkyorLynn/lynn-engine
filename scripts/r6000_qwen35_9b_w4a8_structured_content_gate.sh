#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
PROMPTS_JSON="${PROMPTS_JSON:-${ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LIMIT="${LIMIT:-70}"
MAX_NEW="${MAX_NEW:-96}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p196_qwen35_9b_w4a8_structured_content_gate_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p196] root=$ROOT"
echo "[p196] model=$MODEL"
echo "[p196] prompts=$PROMPTS_JSON limit=$LIMIT"
echo "[p196] out=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \
  --model "$MODEL" \
  --prompts-json "$PROMPTS_JSON" \
  --limit "$LIMIT" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --out "$OUT_JSON"

echo "[p196] done report=$OUT_JSON"
