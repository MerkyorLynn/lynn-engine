#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${REPORT_DIR}/p152_native_packed_outputs_${STAMP}}"
P152_OUT="${P152_OUT:-${REPORT_DIR}/p152_native_packed_outputs_${STAMP}.json}"
P147_OUT="${P147_OUT:-${REPORT_DIR}/p152_native_packed_vs_p147_${STAMP}.json}"
P147_REF_DIR="${P147_REF_DIR:-${REPORT_DIR}/p147_triton_stage_reference_${STAMP}}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p152-wrapper] packed_fixtures=$PACKED_FIXTURES"
echo "[p152-wrapper] candidate_dir=$CANDIDATE_DIR"
echo "[p152-wrapper] p152_out=$P152_OUT"
echo "[p152-wrapper] p147_out=$P147_OUT"

"$PYTHON_BIN" benchmarks/p152_native_packed_moe_stage_outputs.py \
  --packed-fixtures "$PACKED_FIXTURES" \
  --candidate-output-dir "$CANDIDATE_DIR" \
  --out "$P152_OUT"

set +e
"$PYTHON_BIN" benchmarks/p147_triton_contract_moe_stage_gate.py \
  --packed-fixtures "$PACKED_FIXTURES" \
  --candidate-output-dir "$CANDIDATE_DIR" \
  --write-reference-dir "$P147_REF_DIR" \
  --out "$P147_OUT"
p147_rc=$?
set -e

echo "[p152-wrapper] p147_rc=$p147_rc"
echo "[p152-wrapper] p152_out=$P152_OUT"
echo "[p152-wrapper] p147_out=$P147_OUT"
exit 0
