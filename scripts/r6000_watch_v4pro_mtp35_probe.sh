#!/usr/bin/env bash
set -euo pipefail

# Wait for the V4-Pro 35B W4A16 artifact and official 35B MTP sidecar, then
# run the 35B MTP probe without colliding with the W4A16/W4A8 matrix watcher.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
BF16_MODEL="${BF16_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged}"
W4A16_MODEL="${W4A16_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
SIDECAR_FILE="${SIDECAR_FILE:-/root/autodl-tmp/models/mtp_sidecars/qwen36-35b-a3b-mtp-official/mtp.safetensors}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/reports/v4pro_mtp35}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$OUT_DIR/watch_v4pro_mtp35_probe_${STAMP}.log}"
MIN_SIDECAR_BYTES="${MIN_SIDECAR_BYTES:-1600000000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_FOR_MATRIX_WATCHER="${WAIT_FOR_MATRIX_WATCHER:-1}"

mkdir -p "$OUT_DIR"

{
  echo "[watch-v4pro-mtp35] start $(date)"
  echo "[watch-v4pro-mtp35] bf16=$BF16_MODEL"
  echo "[watch-v4pro-mtp35] w4a16=$W4A16_MODEL"
  echo "[watch-v4pro-mtp35] sidecar=$SIDECAR_FILE min_bytes=$MIN_SIDECAR_BYTES"

  while [[ ! -f "$BF16_MODEL/model.safetensors.index.json" ]]; do
    echo "[watch-v4pro-mtp35] waiting BF16 index $(date)"
    sleep "$POLL_SECONDS"
  done

  while [[ ! -f "$W4A16_MODEL/model.safetensors.index.json" ]]; do
    echo "[watch-v4pro-mtp35] waiting W4A16 index $(date)"
    sleep "$POLL_SECONDS"
  done

  while true; do
    bytes=0
    if [[ -f "$SIDECAR_FILE" ]]; then
      bytes=$(stat -c '%s' "$SIDECAR_FILE")
    fi
    if (( bytes >= MIN_SIDECAR_BYTES )); then
      break
    fi
    echo "[watch-v4pro-mtp35] waiting sidecar bytes=$bytes $(date)"
    sleep "$POLL_SECONDS"
  done

  if [[ "$WAIT_FOR_MATRIX_WATCHER" == "1" ]]; then
    while pgrep -f 'watch_and_run_matrix|v4_w4a16_w4a8_generation_matrix|r6000_v4pro_w4a16_w4a8_matrix' >/dev/null; do
      echo "[watch-v4pro-mtp35] waiting matrix watcher/job to finish $(date)"
      sleep "$POLL_SECONDS"
    done
  fi

  echo "[watch-v4pro-mtp35] launch probe $(date)"
  env \
    ROOT="$ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    BF16_MODEL="$BF16_MODEL" \
    W4A16_MODEL="$W4A16_MODEL" \
    SIDECAR_FILE="$SIDECAR_FILE" \
    OUT_DIR="$OUT_DIR" \
    STAMP="$STAMP" \
    bash "$ROOT/scripts/r6000_v4pro_mtp35_probe.sh"
  echo "[watch-v4pro-mtp35] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
