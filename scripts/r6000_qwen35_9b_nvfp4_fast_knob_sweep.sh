#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p149_qwen35_9b_nvfp4_fast_knob_sweep_${STAMP}.json}"
MAX_NEW="${MAX_NEW:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p149-wrapper] root=$ROOT"
echo "[p149-wrapper] python=$PYTHON_BIN"
echo "[p149-wrapper] model=$MODEL"
echo "[p149-wrapper] max_new=$MAX_NEW"
echo "[p149-wrapper] out=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p149_qwen35_9b_nvfp4_fast_knob_sweep.py \
  --model "$MODEL" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-new "$MAX_NEW" \
  --out "$OUT_JSON"

echo "[p149-wrapper] done report=$OUT_JSON"
