#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PACKED_FIXTURES="${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}"
P147_REFERENCE_DIR="${P147_REFERENCE_DIR:-/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p153_native_packed_stage_split_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

export LYNN_NATIVE_CUDA_BUILD_DIR="${LYNN_NATIVE_CUDA_BUILD_DIR:-/tmp/lynn_engine_native_build/p153_${STAMP}}"

echo "[p153-wrapper] root=$ROOT"
echo "[p153-wrapper] python=$PYTHON_BIN"
echo "[p153-wrapper] packed_fixtures=$PACKED_FIXTURES"
echo "[p153-wrapper] p147_reference_dir=$P147_REFERENCE_DIR"
echo "[p153-wrapper] native_build_dir=$LYNN_NATIVE_CUDA_BUILD_DIR"
echo "[p153-wrapper] out_json=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p153_native_packed_moe_stage_split.py \
  --packed-fixtures "$PACKED_FIXTURES" \
  --p147-reference-dir "$P147_REFERENCE_DIR" \
  --out "$OUT_JSON" \
  --warmup "${WARMUP:-10}" \
  --iters "${ITERS:-50}"

echo "[p153-wrapper] done"
echo "[p153-wrapper] report=$OUT_JSON"
