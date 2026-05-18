#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$REPORT_DIR/p144_decode_launch_census_${STAMP}.json}"
MAX_NEW="${MAX_NEW:-12}"
WARMUP_NEW="${WARMUP_NEW:-2}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

exec "$PY" benchmarks/p144_decode_launch_census.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --warmup-new "$WARMUP_NEW" \
  --out "$OUT" \
  "$@"
