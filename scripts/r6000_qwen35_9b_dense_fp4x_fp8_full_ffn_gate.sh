#!/usr/bin/env bash
# P195: full dense FFN composition gate for Qwen3.5-9B FP4xFP8 MMA.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
FIXTURES="${FIXTURES:-/root/autodl-tmp/reports/qwen35_9b/p159_dense_ffn_fixtures_20260519_0458}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar}"
PYTHON="${PYTHON:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
LAYERS="${LAYERS:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="${OUT:-/root/autodl-tmp/reports/qwen35_9b/p195_dense_fp4xfp8_full_ffn_gate_${TIMESTAMP}.json}"
CANDIDATE_OUTPUT_DIR="${CANDIDATE_OUTPUT_DIR:-/root/autodl-tmp/reports/qwen35_9b/p195_dense_fp4xfp8_candidate_outputs_${TIMESTAMP}}"

cd "${REPO_DIR}"
export LYNN_ENABLE_SM120A_FP4_MMA="${LYNN_ENABLE_SM120A_FP4_MMA:-1}"

"${PYTHON}" benchmarks/p195_qwen35_9b_dense_fp4x_fp8_full_ffn_gate.py \
  --fixtures "${FIXTURES}" \
  --sidecar-dir "${SIDECAR_DIR}" \
  --layers "${LAYERS}" \
  --candidate-output-dir "${CANDIDATE_OUTPUT_DIR}" \
  --out "${OUT}"

echo "Report: ${OUT}"
echo "Candidate outputs: ${CANDIDATE_OUTPUT_DIR}"
