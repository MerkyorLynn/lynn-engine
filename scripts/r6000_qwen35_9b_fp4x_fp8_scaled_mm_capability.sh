#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p189_qwen35_9b_fp4x_fp8_scaled_mm_capability_${STAMP}.json}"
K="${K:-2048}"
N="${N:-4096}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p189] root=$ROOT"
echo "[p189] python=$PYTHON_BIN"
echo "[p189] out=$OUT_JSON k=$K n=$N"

"$PYTHON_BIN" benchmarks/p189_qwen35_9b_fp4x_fp8_scaled_mm_capability.py \
  --out "$OUT_JSON" \
  --k "$K" \
  --n "$N"

echo "[p189] done report=$OUT_JSON"
