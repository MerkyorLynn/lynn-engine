#!/usr/bin/env bash
# P199 · NVFP4 artifact size audit
# Run on R6000.  Model dir must exist; if not, writes PENDING JSON.
# Usage: bash scripts/r6000_qwen35_9b_nvfp4_size_audit.sh
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine-main}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${REPORT_DIR}/p199_nvfp4_size_audit_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[p199] root=$ROOT"
echo "[p199] model=$MODEL"
echo "[p199] out=$OUT"

"$PYTHON_BIN" benchmarks/p199_qwen35_9b_nvfp4_size_audit.py \
  --model-dir "$MODEL" \
  --out "$OUT"

echo "[p199] done report=$OUT"
