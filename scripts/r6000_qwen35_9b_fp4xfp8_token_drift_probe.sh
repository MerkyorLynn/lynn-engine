#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

source benchmarks/.config/r6000.env
VENV="$LYNN_VENV"
MODEL="$LYNN_MODEL"
OUT="${LYNN_REPORT_ROOT}/p197_token_drift_probe.json"

exec "$VENV" python benchmarks/p197_qwen35_9b_fp4xfp8_token_drift_probe.py \
  --model "$MODEL" \
  --max-new 8 \
  --max-seq-len 4096 \
  --limit 5 \
  --out "$OUT"
