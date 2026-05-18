#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
REFERENCE_DIR="${REFERENCE_DIR:-${REPORT_DIR}/p147_triton_stage_reference_${STAMP}}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p147_triton_stage_gate_${STAMP}.json}"
CANDIDATE_OUTPUT_DIR="${CANDIDATE_OUTPUT_DIR:-}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

echo "[p147-wrapper] root=$ROOT"
echo "[p147-wrapper] python=$PYTHON_BIN"
echo "[p147-wrapper] packed_fixtures=$PACKED_FIXTURES"
echo "[p147-wrapper] reference_dir=$REFERENCE_DIR"
echo "[p147-wrapper] out_json=$OUT_JSON"
if [[ -n "$CANDIDATE_OUTPUT_DIR" ]]; then
  echo "[p147-wrapper] candidate_output_dir=$CANDIDATE_OUTPUT_DIR"
fi

args=(
  benchmarks/p147_triton_contract_moe_stage_gate.py
  --packed-fixtures "$PACKED_FIXTURES"
  --write-reference-dir "$REFERENCE_DIR"
  --out "$OUT_JSON"
  --warmup "${WARMUP:-5}"
  --iters "${ITERS:-20}"
)

if [[ -n "$CANDIDATE_OUTPUT_DIR" ]]; then
  args+=(--candidate-output-dir "$CANDIDATE_OUTPUT_DIR")
fi

"$PYTHON_BIN" "${args[@]}"

echo "[p147-wrapper] done"
echo "[p147-wrapper] report=$OUT_JSON"
