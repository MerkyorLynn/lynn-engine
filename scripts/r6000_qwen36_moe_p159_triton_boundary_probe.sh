#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
P147_REFERENCE_DIR="${P147_REFERENCE_DIR:-/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318}"
P157_REPORT="${P157_REPORT:-}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p159_qwen36_triton_active_boundary_probe_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p159-wrapper] root=$ROOT"
echo "[p159-wrapper] packed_fixtures=$PACKED_FIXTURES"
echo "[p159-wrapper] p147_reference_dir=$P147_REFERENCE_DIR"
echo "[p159-wrapper] p157_report=${P157_REPORT:-<cli-default-expectations>}"
echo "[p159-wrapper] out_json=$OUT_JSON"

args=(
  benchmarks/p159_qwen36_triton_active_boundary_probe.py
  --packed-fixtures "$PACKED_FIXTURES"
  --p147-reference-dir "$P147_REFERENCE_DIR"
  --out "$OUT_JSON"
  --warmup "${WARMUP:-10}"
  --iters "${ITERS:-50}"
)

if [[ -n "$P157_REPORT" ]]; then
  args+=(--p157-report "$P157_REPORT")
fi

if [[ "${PROFILE_EVENTS:-1}" != "0" ]]; then
  args+=(--profile-events)
fi

if [[ "${MAX_FIXTURES:-0}" != "0" ]]; then
  args+=(--max-fixtures "$MAX_FIXTURES")
fi

"$PYTHON_BIN" "${args[@]}"

echo "[p159-wrapper] done"
echo "[p159-wrapper] report=$OUT_JSON"
