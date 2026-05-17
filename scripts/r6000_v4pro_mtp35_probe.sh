#!/usr/bin/env bash
set -euo pipefail

# Probe the official Qwen3.6-35B-A3B MTP sidecar against the unpruned
# Lynn V4-Pro 35B-A3B route.
#
# This is the 35B counterpart to the 27B MTP ladder. It answers, in order:
#   1. does the official 35B MTP sidecar match the V4-Pro 35B config?
#   2. can the sidecar produce finite draft logits on the BF16 base?
#   3. what is the one-token iterative accept rate before any Lynn calibration?
#   4. after W4A16 packing, what shadow-serving credit does the sidecar get?

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
BF16_MODEL="${BF16_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged}"
W4A16_MODEL="${W4A16_MODEL:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/models/mtp_sidecars/qwen36-35b-a3b-mtp-official}"
SIDECAR_FILE="${SIDECAR_FILE:-$SIDECAR_DIR/mtp.safetensors}"
OUT_DIR="${OUT_DIR:-/root/autodl-tmp/reports/v4pro_mtp35}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$OUT_DIR/r6000_v4pro_mtp35_probe_${STAMP}.log}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LYNN_MOE_IMPL="${LYNN_MOE_IMPL:-packed_nvfp4}"
export LYNN_NATIVE_FP4_LM_HEAD="${LYNN_NATIVE_FP4_LM_HEAD:-1}"
export LYNN_NATIVE_ACTIVE_MOE="${LYNN_NATIVE_ACTIVE_MOE:-1}"
export LYNN_NATIVE_ACTIVE_MOE_BACKEND="${LYNN_NATIVE_ACTIVE_MOE_BACKEND:-packed_nvfp4}"
export LYNN_ROUTER_TOPK_SORTED="${LYNN_ROUTER_TOPK_SORTED:-1}"
export LYNN_MTP_LAYER_MOE="${LYNN_MTP_LAYER_MOE:-decode_slot_sorted}"

mkdir -p "$OUT_DIR"
cd "$ROOT"

{
  echo "[r6000-v4pro-mtp35] start $(date)"
  echo "[r6000-v4pro-mtp35] bf16=$BF16_MODEL"
  echo "[r6000-v4pro-mtp35] w4a16=$W4A16_MODEL"
  echo "[r6000-v4pro-mtp35] sidecar=$SIDECAR_FILE"
  df -h "$(dirname "$BF16_MODEL")" "$OUT_DIR" 2>/dev/null || true

  if [[ ! -f "$SIDECAR_FILE" ]]; then
    echo "[r6000-v4pro-mtp35][fail] missing sidecar file: $SIDECAR_FILE" >&2
    exit 2
  fi
  if [[ ! -f "$BF16_MODEL/config.json" ]]; then
    echo "[r6000-v4pro-mtp35][fail] missing BF16 config: $BF16_MODEL/config.json" >&2
    exit 2
  fi

  echo "[r6000-v4pro-mtp35] shape audit official 35B sidecar vs V4-Pro config"
  "$PYTHON_BIN" scripts/a100_mtp_sidecar_shape_audit.py \
    --sidecar-dir "$SIDECAR_DIR" \
    --base-model "$BF16_MODEL" \
    --out "$OUT_DIR/r6000_v4pro_mtp35_shape_audit_${STAMP}.json"

  if [[ -f "$BF16_MODEL/model.safetensors.index.json" ]]; then
    echo "[r6000-v4pro-mtp35] BF16 forward smoke"
    "$PYTHON_BIN" scripts/a100_mtp_forward_smoke.py \
      --base-model "$BF16_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$OUT_DIR/r6000_v4pro_mtp35_bf16_forward_smoke_${STAMP}.json" \
      --prompt "Return one JSON object with keys city and unit for Tokyo in metric units. No markdown." \
      --use-chat-template \
      --top-k "${TOP_K:-8}" \
      --dtype "${DTYPE:-bfloat16}"

    echo "[r6000-v4pro-mtp35] BF16 iterative one-token accept probe"
    "$PYTHON_BIN" scripts/a100_mtp_iterative_accept_probe.py \
      --base-model "$BF16_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$OUT_DIR/r6000_v4pro_mtp35_bf16_iter_accept_${STAMP}.json" \
      --use-chat-template \
      --max-new "${MAX_NEW:-8}" \
      --top-k "${TOP_K:-8}" \
      --dtype "${DTYPE:-bfloat16}"
  else
    echo "[r6000-v4pro-mtp35] BF16 model weights incomplete; skip forward/accept"
  fi

  if [[ -f "$W4A16_MODEL/model.safetensors.index.json" ]]; then
    echo "[r6000-v4pro-mtp35] W4A16 shadow serving-credit probe"
    "$PYTHON_BIN" benchmarks/p107_mtp_shadow_serving_credit_probe.py \
      --model "$W4A16_MODEL" \
      --sidecar-file "$SIDECAR_FILE" \
      --out "$OUT_DIR/r6000_v4pro_mtp35_w4a16_p107_shadow_${STAMP}.json" \
      --use-chat-template \
      --max-new "${P107_MAX_NEW:-16}" \
      --top-k "${P107_TOP_K:-8}" \
      --dtype "${DTYPE:-bfloat16}"
  else
    echo "[r6000-v4pro-mtp35] W4A16 model not ready; skip P107"
  fi

  echo "[r6000-v4pro-mtp35] done $(date)"
} 2>&1 | tee "$LOG"

echo "$LOG"
