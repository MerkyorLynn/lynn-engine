#!/usr/bin/env bash
# P194: full dense FFN projection gate for Qwen3.5-9B FP4xFP8 MMA.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
FIXTURES="${FIXTURES:-/root/autodl-tmp/reports/qwen35_9b/p159_dense_ffn_fixtures_20260519_0458}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
LAYERS="${LAYERS:-0,16}"
PROJECTIONS="${PROJECTIONS:-gate_proj,up_proj,down_proj}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT:-/root/autodl-tmp/reports/qwen35_9b/p194_dense_fp4xfp8_projection_gate_${TIMESTAMP}.json}"

cd "${REPO_DIR}"
export LYNN_ENABLE_SM120A_FP4_MMA="${LYNN_ENABLE_SM120A_FP4_MMA:-1}"

"${PYTHON}" benchmarks/p194_qwen35_9b_dense_fp4x_fp8_projection_gate.py \
  --fixtures "${FIXTURES}" \
  --sidecar-dir "${SIDECAR_DIR}" \
  --layers "${LAYERS}" \
  --projections "${PROJECTIONS}" \
  --out "${OUT}"

echo "Report: ${OUT}"
