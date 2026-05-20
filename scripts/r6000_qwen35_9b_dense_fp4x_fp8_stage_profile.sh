#!/usr/bin/env bash
# P200: stage profile for Qwen3.5-9B FP4xFP8 dense FFN on R6000.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
FIXTURES="${FIXTURES:-/root/autodl-tmp/reports/qwen35_9b/p159_dense_ffn_fixtures_20260519_0458}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
LAYERS="${LAYERS:-0,8,16,24,31}"
ITERS="${ITERS:-80}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT:-/root/autodl-tmp/reports/qwen35_9b/p200_dense_fp4xfp8_stage_profile_${TIMESTAMP}.json}"

cd "${REPO_DIR}"
export LYNN_ENABLE_SM120A_FP4_MMA="${LYNN_ENABLE_SM120A_FP4_MMA:-1}"
export LYNN_NATIVE_CUDA_ARCH_AUTO="${LYNN_NATIVE_CUDA_ARCH_AUTO:-1}"
export LYNN_NATIVE_CUDA_BUILD_DIR="${LYNN_NATIVE_CUDA_BUILD_DIR:-/tmp/lynn_engine_native_build/p200_${TIMESTAMP}}"

"${PYTHON}" benchmarks/p200_qwen35_9b_dense_fp4x_fp8_stage_profile.py \
  --fixtures "${FIXTURES}" \
  --sidecar-dir "${SIDECAR_DIR}" \
  --layers "${LAYERS}" \
  --iters "${ITERS}" \
  --out "${OUT}"

echo "Report: ${OUT}"
