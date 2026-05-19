#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
PROMPTS_JSON="${PROMPTS_JSON:-${ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW="${MAX_NEW:-64}"
LIMIT="${LIMIT:-70}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p184_qwen35_9b_nvfp4_convstrict_exact_gate_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p184-wrapper] root=$ROOT"
echo "[p184-wrapper] model=$MODEL"
echo "[p184-wrapper] prompts=$PROMPTS_JSON"
echo "[p184-wrapper] limit=$LIMIT max_new=$MAX_NEW"
echo "[p184-wrapper] out=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p184_qwen35_9b_nvfp4_convstrict_exact_gate.py \
  --model "$MODEL" \
  --prompts-json "$PROMPTS_JSON" \
  --limit "$LIMIT" \
  --max-new "$MAX_NEW" \
  --out "$OUT_JSON"

echo "[p184-wrapper] done report=$OUT_JSON"
