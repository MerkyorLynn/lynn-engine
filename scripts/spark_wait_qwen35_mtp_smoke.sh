#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526}"
SIDECAR="${SIDECAR:-/home/merkyor/models/mtp_sidecars/qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors}"
LYNN_DIR="${LYNN_DIR:-/home/merkyor/lynn-engine}"
IMAGE="${IMAGE:-lynn-eval-base:cu13}"
MAX_NEW="${MAX_NEW:-32}"
SPEC_K_LIST="${SPEC_K_LIST:-2,4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_USED_MIB_BEFORE_RUN="${MAX_USED_MIB_BEFORE_RUN:-60000}"
OUT="${OUT:-$LYNN_DIR/reports/mtp/qwen35_mtp_kn_smoke_$(date +%Y%m%d_%H%M%S).json}"
LOG="${LOG:-$LYNN_DIR/logs/qwen35_mtp_kn_smoke_wait_$(date +%Y%m%d_%H%M%S).log}"
MODEL_IN_CONTAINER="/models/${MODEL#/home/merkyor/models/}"
SIDECAR_IN_CONTAINER="/models/${SIDECAR#/home/merkyor/models/}"

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

gpu_used_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += int($1)} END {print sum + 0}'
}

log() {
  printf '[qwen35-mtp-wait] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

log "model=$MODEL"
log "sidecar=$SIDECAR"
log "out=$OUT"
log "waiting until GPU process memory <= ${MAX_USED_MIB_BEFORE_RUN} MiB"

while true; do
  used="$(gpu_used_mib || true)"
  if [[ -n "$used" && "$used" =~ ^[0-9]+$ && "$used" -le "$MAX_USED_MIB_BEFORE_RUN" ]]; then
    log "GPU process memory ${used} MiB <= threshold; starting smoke"
    break
  fi
  log "GPU process memory ${used:-unknown} MiB; sleeping ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

docker run --rm --gpus all --ipc=host \
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

log "done"
