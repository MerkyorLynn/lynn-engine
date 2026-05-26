#!/usr/bin/env bash
set -euo pipefail

# Maintenance-window Qwen35 MTP smoke.
#
# The production APEX-MTP llama.cpp fallback has Restart=always, so a plain
# systemctl stop is not enough for long 35B Python runner loads. This wrapper
# temporarily runtime-masks the service, runs one smoke, and always unmask/starts
# the service on exit.

SERVICE="${SERVICE:-lynn-apex-mtp-llamacpp.service}"
MODEL="${MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
SIDECAR="${SIDECAR:-/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors}"
LYNN_DIR="${LYNN_DIR:-/home/merkyor/lynn-engine}"
IMAGE="${IMAGE:-lynn-eval-base:cu13}"
MAX_NEW="${MAX_NEW:-8}"
SPEC_K_LIST="${SPEC_K_LIST:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
OUT="${OUT:-$LYNN_DIR/reports/mtp/qwen35_mtp_maintenance_$(date +%Y%m%d_%H%M%S).json}"
LOG="${LOG:-$LYNN_DIR/logs/qwen35_mtp_maintenance_$(date +%Y%m%d_%H%M%S).log}"
MODEL_IN_CONTAINER="/models/${MODEL#/home/merkyor/models/}"
SIDECAR_IN_CONTAINER="/models/${SIDECAR#/home/merkyor/models/}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

log() {
  printf '[qwen35-mtp-maint] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

restore_service() {
  set +e
  sudo -n systemctl unmask "$SERVICE" >/dev/null 2>&1
  sudo -n systemctl start "$SERVICE" >/dev/null 2>&1
  sleep 2
  status="$(systemctl is-active "$SERVICE" 2>/dev/null || true)"
  log "restore service status=${status:-unknown}"
}

trap restore_service EXIT

log "runtime-masking and stopping $SERVICE"
sudo -n systemctl mask --runtime "$SERVICE"
sudo -n systemctl stop "$SERVICE" || true
sleep 3
log "pre-run service status=$(systemctl is-active "$SERVICE" 2>/dev/null || true)"
log "model=$MODEL"
log "sidecar=$SIDECAR"
log "out=$OUT"

timeout "$TIMEOUT_SECONDS" docker run --rm --gpus all --ipc=host \
  "$@" \
  -v "$LYNN_DIR:/workspace" \
  -v /home/merkyor/models:/models \
  -w /workspace \
  "$IMAGE" \
  python3 scripts/spark_mtp_speculative_smoke.py \
    --model "$MODEL_IN_CONTAINER" \
    --sidecar "$SIDECAR_IN_CONTAINER" \
    --out "${OUT#$LYNN_DIR/}" \
    --max-new "$MAX_NEW" \
    --spec-k-list "$SPEC_K_LIST" \
  2>&1 | tee -a "$LOG"

log "smoke complete"
