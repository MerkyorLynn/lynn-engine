#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW="${MAX_NEW:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p183_qwen35_9b_nvfp4_exact_fast_isolation_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p183-wrapper] root=$ROOT"
echo "[p183-wrapper] model=$MODEL"
echo "[p183-wrapper] max_new=$MAX_NEW"
echo "[p183-wrapper] out=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p183_qwen35_9b_nvfp4_exact_fast_isolation.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --out "$OUT_JSON"

echo "[p183-wrapper] done report=$OUT_JSON"
