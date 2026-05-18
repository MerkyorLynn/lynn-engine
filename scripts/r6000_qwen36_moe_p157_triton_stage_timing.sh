#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p157_triton_moe_stage_timing_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p157-wrapper] root=$ROOT"
echo "[p157-wrapper] out_json=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p157_triton_moe_stage_timing_correction.py \
  --packed-fixtures "${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}" \
  --p147-reference-dir "${P147_REFERENCE_DIR:-/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318}" \
  --out "$OUT_JSON" \
  --warmup "${WARMUP:-10}" \
  --iters "${ITERS:-50}"

echo "[p157-wrapper] done"
echo "[p157-wrapper] report=$OUT_JSON"
