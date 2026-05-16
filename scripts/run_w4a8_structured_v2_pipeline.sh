#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/lynn-a100/lynn-engine
source /mnt/data/lynn-a100/envs/train/bin/activate

LOG=logs/a100_w4a8_structured_v2_pipeline.log
SRC=/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-final
ALPHA=/mnt/data/lynn-a100/artifacts/w4a8_alpha_overlay_structured_v2
FOLDED=/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-structured-v2
PROMPTS=reports/a100/w4a8_structured_recovery_prompts_v1.json
LAYERS="20 26 24 12 23 32 5 3 19 29 15 17 7 2 10 6 30 8 25 13 37 18 11"

{
  echo "[pipeline-v2] start $(date)"
  rm -rf "$ALPHA"
  mkdir -p "$ALPHA" logs reports/a100

  echo "[pipeline-v2] recovery conservative alpha clamp"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    python scripts/a100_w4a8_intermediate_scale_recovery_probe.py \
      --model "$SRC" \
      --out reports/a100/a100_w4a8_recovery_structured_v2_fail23.json \
      --layers $LAYERS \
      --prompts-file "$PROMPTS" \
      --fmt e4m3 \
      --steps 180 \
      --lr 0.012 \
      --reg 0.0005 \
      --alpha-mode expert \
      --alpha-min 0.85 \
      --alpha-max 1.15 \
      --save-alpha-dir "$ALPHA"

  echo "[pipeline-v2] folding artifact -> $FOLDED"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    python scripts/a100_fold_w4a8_alpha_overlay.py \
      --src-model "$SRC" \
      --alpha-dir "$ALPHA" \
      --out-model "$FOLDED" \
      --overwrite \
      > reports/a100/a100_w4a8_fold_structured_v2_manifest_stdout.json

  echo "[pipeline-v2] folded gate structured prompts"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    python scripts/a100_w4a8_folded_vs_original_gate.py \
      --original-model "$SRC" \
      --folded-model "$FOLDED" \
      --out reports/a100/a100_w4a8_folded_vs_original_structured_v2.json \
      --layers $LAYERS \
      --prompts-file "$PROMPTS" \
      --fmt e4m3

  echo "[pipeline-v2] generation gate structured prompts"
  env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    LYNN_MOE_IMPL=bmm \
    LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3 \
    LYNN_W4A8_FAKE_QUANT_GRANULARITY=per16 \
    python scripts/a100_w4a8_generation_gate.py \
      --model "$SRC" \
      --folded-model "$FOLDED" \
      --max-new 48 \
      --top-k 5 \
      --prompts-file "$PROMPTS" \
      --out reports/a100/a100_w4a8_generation_gate_structured_v2_12prompt_48tok.json

  echo "[pipeline-v2] done $(date)"
} >> "$LOG" 2>&1
