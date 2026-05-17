#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
P97_LAYERS=${P97_LAYERS:-"4 12 20 28 36"}
P97_REPEATS=${P97_REPEATS:-30}
P97_WARMUP=${P97_WARMUP:-8}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
REPORT_DIR=$REPORT_ROOT/p16_155
P97_SUMMARY=$REPORT_DIR/p97_r6000_v2_multilayer_summary_${TS}.json
LOG=$REPORT_DIR/p97_r6000_v2_serial_${TS}.log

mkdir -p "$REPORT_DIR"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

{
  echo "[r6000-p97-v2] start $(date)"
  echo "[r6000-p97-v2] model=$MODEL"
  echo "[r6000-p97-v2] layers=$P97_LAYERS"
  echo "[r6000-p97-v2] repeats=$P97_REPEATS warmup=$P97_WARMUP"

  for layer in $P97_LAYERS; do
    out=$REPORT_DIR/p97_r6000_v2_layer${layer}_${TS}.json
    echo "[r6000-p97-v2] layer=$layer -> $out"
    "$PY" benchmarks/p97_sm120a_active_moe_interval_decomposition.py \
      --model "$MODEL" \
      --out "$out" \
      --layer "$layer" \
      --repeats "$P97_REPEATS" \
      --warmup "$P97_WARMUP"
  done

  echo "[r6000-p97-v2] summarize -> $P97_SUMMARY"
  "$PY" scripts/r6000_summarize_p97_reports.py \
    --glob "$REPORT_DIR/p97_r6000_v2_layer*_${TS}.json" \
    --model "$MODEL" \
    --out "$P97_SUMMARY"

  echo "[r6000-p97-v2] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
