#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/mnt/data/lynn-a100/lynn-engine}
PY=${PY:-/mnt/data/lynn-a100/envs/train/bin/python}
SRC=${SRC:-/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-final}
BASE_ALPHA=${BASE_ALPHA:-/mnt/data2/lynn-a100/artifacts/w4a8_alpha_overlay_structured_v10_top6}
NAME=${NAME:-structured_v16_top6_damped075}
DAMPING=${DAMPING:-0.75}
LAYERS=${LAYERS:-"2 3 24 26 29 30"}
PROMPTS=${PROMPTS:-reports/a100/w4a8_structured_recovery_prompts_v1.json}
MAX_NEW=${MAX_NEW:-48}
TOP_K=${TOP_K:-5}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/mnt/data2/lynn-a100/artifacts}
MODEL_ROOT=${MODEL_ROOT:-/mnt/data2/lynn-a100/models}
REPORT_DIR=${REPORT_DIR:-reports/a100}
LOG_DIR=${LOG_DIR:-logs}
ALPHA=$ARTIFACT_ROOT/w4a8_alpha_overlay_${NAME}
FOLDED=$MODEL_ROOT/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-${NAME}
LOG=$LOG_DIR/a100_w4a8_${NAME}_pipeline.log

mkdir -p "$LOG_DIR" "$REPORT_DIR"
cd "$REPO"
export PYTHONPATH="$REPO"

{
  echo "[damped-pipeline] start $(date)"
  echo "[damped-pipeline] name=$NAME damping=$DAMPING layers=$LAYERS"
  echo "[damped-pipeline] base_alpha=$BASE_ALPHA"

  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" scripts/a100_dampen_w4a8_alpha_overlay.py \
    --source-alpha-dir "$BASE_ALPHA" \
    --out-alpha-dir "$ALPHA" \
    --damping "$DAMPING" \
    --layers $LAYERS \
    --out "$REPORT_DIR/a100_w4a8_dampen_${NAME}.json" \
    --overwrite

  echo "[damped-pipeline] folding artifact -> $FOLDED"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" scripts/a100_fold_w4a8_alpha_overlay.py \
    --src-model "$SRC" \
    --alpha-dir "$ALPHA" \
    --out-model "$FOLDED" \
    --overwrite \
    > "$REPORT_DIR/a100_w4a8_fold_${NAME}_manifest_stdout.json"

  echo "[damped-pipeline] folded local gate"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" scripts/a100_w4a8_folded_vs_original_gate.py \
    --original-model "$SRC" \
    --folded-model "$FOLDED" \
    --out "$REPORT_DIR/a100_w4a8_folded_vs_original_${NAME}.json" \
    --layers $LAYERS \
    --prompts-file "$PROMPTS" \
    --fmt e4m3

  echo "[damped-pipeline] generation gate"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    LYNN_MOE_IMPL=bmm \
    LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3 \
    LYNN_W4A8_FAKE_QUANT_GRANULARITY=per16 \
    "$PY" scripts/a100_w4a8_generation_gate.py \
      --model "$SRC" \
      --folded-model "$FOLDED" \
      --max-new "$MAX_NEW" \
      --top-k "$TOP_K" \
      --prompts-file "$PROMPTS" \
      --out "$REPORT_DIR/a100_w4a8_generation_gate_${NAME}_12prompt_${MAX_NEW}tok.json"

  echo "[damped-pipeline] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
