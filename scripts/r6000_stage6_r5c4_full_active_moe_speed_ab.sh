#!/usr/bin/env bash
# R6000 wrapper for Stage 6 R5-C4 full active-MoE speed A/B validation.
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/src/lynn-engine-r5c2c-codex}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT}/reports/stage6/r5c4_full_active_moe_prefill_speed_ab_${STAMP}}"
CANDIDATE_JSON="${CANDIDATE_JSON:-}"
CANDIDATE_METRICS_JSON="${CANDIDATE_METRICS_JSON:-}"
INPUT_R5C3C="${INPUT_R5C3C:-${ROOT}/reports/stage6/r5c3c_down_weighted_parity_smoke_20260604_130243/result.json}"

if [[ -z "$CANDIDATE_JSON" && -z "$CANDIDATE_METRICS_JSON" ]]; then
  echo "CANDIDATE_JSON or CANDIDATE_METRICS_JSON is required; it must point to same-scope speed A/B evidence." >&2
  exit 64
fi

mkdir -p "$OUT_DIR"
nvidia-smi > "${OUT_DIR}/nvidia_smi_before.txt" || true
git -C "$ROOT" rev-parse HEAD > "${OUT_DIR}/lynn_engine_head.txt" || true
git -C "$ROOT" status --short > "${OUT_DIR}/lynn_engine_status.txt" || true

if [[ -z "$CANDIDATE_JSON" ]]; then
  CANDIDATE_JSON="${OUT_DIR}/candidate.json"
  "$PYTHON_BIN" "$ROOT/scripts/stage6_r5c4_candidate_from_metrics.py" \
    --raw-metrics "$CANDIDATE_METRICS_JSON" \
    --out "$CANDIDATE_JSON" \
    --strict
fi

"$PYTHON_BIN" "$ROOT/scripts/r6000_stage6_r5c4_full_active_moe_speed_ab.py" \
  --input-r5c3c "$INPUT_R5C3C" \
  --candidate-json "$CANDIDATE_JSON" \
  --out "${OUT_DIR}/result.json"

"$PYTHON_BIN" "$ROOT/scripts/summarize_stage6_r5c4_full_active_moe_speed_ab.py" \
  "${OUT_DIR}/result.json" \
  --markdown-out "${OUT_DIR}/summary.md"

echo "[r5c4-speed-ab] artifact=$OUT_DIR"
