#!/usr/bin/env bash
# P198 · Native FP4×FP8 build preflight
# Run on R6000 with CUDA build.
# Usage: bash scripts/r6000_qwen35_9b_native_fp4_build_preflight.sh
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${REPORT_DIR}/p198_native_fp4_preflight_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p198] root=$ROOT"
echo "[p198] out=$OUT"

"$PYTHON_BIN" benchmarks/p198_qwen35_9b_native_fp4_build_preflight.py \
  --out "$OUT"

echo "[p198] done report=$OUT"
