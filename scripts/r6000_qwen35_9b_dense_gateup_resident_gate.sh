#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p169_dense_gateup_resident_gate_${STAMP}.json}"
MAX_NEW="${MAX_NEW:-128 512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p169] root=$ROOT"
echo "[p169] python=$PYTHON_BIN"
echo "[p169] model=$MODEL"
echo "[p169] max_new=$MAX_NEW"
echo "[p169] out=$OUT_JSON"

# shellcheck disable=SC2206
max_new_args=($MAX_NEW)

"$PYTHON_BIN" benchmarks/p169_qwen35_9b_dense_gateup_resident_gate.py \
  --model "$MODEL" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --max-new "${max_new_args[@]}" \
  --out "$OUT_JSON"

echo "[p169] done report=$OUT_JSON"
