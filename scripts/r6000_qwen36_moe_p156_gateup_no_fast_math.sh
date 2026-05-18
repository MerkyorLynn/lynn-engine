#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
OUT_JSON="${OUT_JSON:-${REPORT_DIR}/p156_native_packed_gateup_no_fast_math_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

export LYNN_NATIVE_CUDA_NO_FAST_MATH=1
export LYNN_NATIVE_CUDA_BUILD_DIR="${LYNN_NATIVE_CUDA_BUILD_DIR:-/tmp/lynn_engine_native_build/p156_no_fast_math_${STAMP}}"

echo "[p156-wrapper] no-fast-math native build"
echo "[p156-wrapper] root=$ROOT"
echo "[p156-wrapper] native_build_dir=$LYNN_NATIVE_CUDA_BUILD_DIR"
echo "[p156-wrapper] out_json=$OUT_JSON"

"$PYTHON_BIN" benchmarks/p155_native_packed_moe_gateup_raw_accum.py \
  --packed-fixtures "${PACKED_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518}" \
  --p147-reference-dir "${P147_REFERENCE_DIR:-/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318}" \
  --out "$OUT_JSON" \
  --warmup "${WARMUP:-5}" \
  --iters "${ITERS:-20}"

echo "[p156-wrapper] done"
echo "[p156-wrapper] report=$OUT_JSON"
