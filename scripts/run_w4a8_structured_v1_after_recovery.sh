#!/usr/bin/env bash
set -u

cd /mnt/data/lynn-a100/lynn-engine
source /mnt/data/lynn-a100/envs/train/bin/activate

PID_FILE=logs/a100_w4a8_recovery_structured_v1_fail23.pid
REC_OUT=reports/a100/a100_w4a8_recovery_structured_v1_fail23.json
LOG=logs/a100_w4a8_structured_v1_pipeline.log
SRC=/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-final
ALPHA=/mnt/data/lynn-a100/artifacts/w4a8_alpha_overlay_structured_v1
FOLDED=/mnt/data/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-structured-v1
PROMPTS=reports/a100/w4a8_structured_recovery_prompts_v1.json
LAYERS="20 26 24 12 23 32 5 3 19 29 15 17 7 2 10 6 30 8 25 13 37 18 11"

{
  echo "[pipeline] start $(date)"
  while [ ! -f "$PID_FILE" ]; do sleep 5; done
  PID=$(cat "$PID_FILE")
  echo "[pipeline] waiting recovery pid=$PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 30; done
  echo "[pipeline] recovery pid exited $(date)"
  if [ ! -f "$REC_OUT" ]; then
    echo "[pipeline] missing recovery report $REC_OUT"
    exit 2
  fi
  python - <<PY
import json
p = "$REC_OUT"
d = json.load(open(p))
print("[pipeline] recovery decision:", d.get("decision"))
print("[pipeline] recovery aggregate:", d.get("aggregate"))
PY

  echo "[pipeline] folding artifact -> $FOLDED"
  PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    python scripts/a100_fold_w4a8_alpha_overlay.py \
      --src-model "$SRC" \
      --alpha-dir "$ALPHA" \
      --out-model "$FOLDED" \
      --overwrite \
      > reports/a100/a100_w4a8_fold_structured_v1_manifest_stdout.json

  echo "[pipeline] folded gate structured prompts"
  PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    python scripts/a100_w4a8_folded_vs_original_gate.py \
      --original-model "$SRC" \
      --folded-model "$FOLDED" \
      --out reports/a100/a100_w4a8_folded_vs_original_structured_v1.json \
      --layers $LAYERS \
      --prompts-file "$PROMPTS" \
      --fmt e4m3

  echo "[pipeline] generation gate structured prompts"
  env PYTHONPATH=/mnt/data/lynn-a100/lynn-engine \
    LYNN_MOE_IMPL=bmm \
    LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3 \
    LYNN_W4A8_FAKE_QUANT_GRANULARITY=per16 \
    python scripts/a100_w4a8_generation_gate.py \
      --model "$SRC" \
      --folded-model "$FOLDED" \
      --max-new 48 \
      --top-k 5 \
      --prompts-file "$PROMPTS" \
      --out reports/a100/a100_w4a8_generation_gate_structured_v1_12prompt_48tok.json

  echo "[pipeline] done $(date)"
} >> "$LOG" 2>&1
