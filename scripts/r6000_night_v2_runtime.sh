#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/lynn-engine
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
PROMPTS=${PROMPTS:-reports/a100/w4a8_structured_recovery_prompts_v1.json}
P97_LAYERS=${P97_LAYERS:-"4 12 20 28 36"}
MAX_NEW=${MAX_NEW:-96}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
P105_OUT=$REPORT_ROOT/p105/p105_w4a8_generate_gate_r6000_v2_${TS}.json
P97_SUMMARY=$REPORT_ROOT/p16_155/p97_r6000_v2_multilayer_summary_${TS}.json
LOG=$REPORT_ROOT/night_r6000_v2_runtime_${TS}.log

mkdir -p "$REPORT_ROOT/p105" "$REPORT_ROOT/p16_155"

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

{
  echo "[r6000-night] start $(date)"
  echo "[r6000-night] model=$MODEL"
  echo "[r6000-night] prompts=$PROMPTS"

  echo "[r6000-night] run P105 generate gate"
  "$PY" benchmarks/p105_w4a8_generate_gate.py \
    --model "$MODEL" \
    --out "$P105_OUT" \
    --max-new "$MAX_NEW" \
    --prompts-file "$PROMPTS"

  echo "[r6000-night] run P97 interval decomposition"
  for layer in $P97_LAYERS; do
    out=$REPORT_ROOT/p16_155/p97_r6000_v2_layer${layer}_${TS}.json
    echo "[r6000-night] layer=$layer -> $out"
    "$PY" benchmarks/p97_sm120a_active_moe_interval_decomposition.py \
      --model "$MODEL" \
      --out "$out" \
      --layer "$layer" \
      --repeats 30 \
      --warmup 8
  done

  echo "[r6000-night] summarize P97"
  "$PY" scripts/r6000_summarize_p97_reports.py \
    --glob "$REPORT_ROOT/p16_155/p97_r6000_v2_layer*_${TS}.json" \
    --model "$MODEL" \
    --out "$P97_SUMMARY"

  echo "[r6000-night] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
