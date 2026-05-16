#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/data/lynn-a100/lynn-engine
PY=${PY:-/mnt/data/lynn-a100/envs/train/bin/python}
BASE_MODEL=${BASE_MODEL:-/mnt/data2/lynn-a100/models/lynn-27b-variable-recovery-step5000-bf16-w4a8-alpha-overlay-structured_v10_top6}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-/mnt/data2/lynn-a100/models/qwen36_35b_a3b_mtp_shards}
SIDECAR_DIR=${SIDECAR_DIR:-/mnt/data2/lynn-a100/models/mtp_sidecars/qwen36-35b-a3b-mtp}
REPORT_DIR=${REPORT_DIR:-/mnt/data2/lynn-a100/reports/mtp}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
LOG=$REPORT_DIR/a100_night_mtp_prep_${TS}.log

mkdir -p "$REPORT_DIR" "$SIDECAR_DIR" "$DOWNLOAD_DIR"

cd "$REPO"
export PYTHONPATH="$REPO"

{
  echo "[a100-mtp] start $(date)"
  echo "[a100-mtp] base_model=$BASE_MODEL"

  echo "[a100-mtp] audit official Qwen3.6-35B-A3B MTP index"
  "$PY" scripts/a100_qwen36_a3b_mtp_index_audit.py \
    --out "$REPORT_DIR/a100_qwen36_a3b_mtp_index_audit_${TS}.json"

  echo "[a100-mtp] extract official mtp shards into sidecar"
  "$PY" scripts/a100_extract_qwen36_a3b_mtp_sidecar.py \
    --download-dir "$DOWNLOAD_DIR" \
    --sidecar-dir "$SIDECAR_DIR" \
    --out "$REPORT_DIR/a100_qwen36_a3b_mtp_sidecar_extract_${TS}.json"

  echo "[a100-mtp] shape-audit sidecar against Lynn base"
  "$PY" scripts/a100_mtp_sidecar_shape_audit.py \
    --sidecar-dir "$SIDECAR_DIR" \
    --base-model "$BASE_MODEL" \
    --out "$REPORT_DIR/a100_qwen36_a3b_mtp_sidecar_shape_audit_${TS}.json"

  echo "[a100-mtp] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
