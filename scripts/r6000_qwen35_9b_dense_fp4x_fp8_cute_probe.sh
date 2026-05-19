#!/usr/bin/env bash
# P191: Dense FP4xFP8 CuTe PoC probe on R6000
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
FIXTURES="${FIXTURES:-/root/autodl-tmp/reports/qwen35_9b/p159_dense_ffn_fixtures_20260519_0458}"
SIDECAR_DIR="${SIDECAR_DIR:-/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.12}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="/root/autodl-tmp/reports/qwen35_9b/p191_dense_fp4xfp8_poc_${TIMESTAMP}.json"

echo "P191: Dense FP4xFP8 CuTe PoC"
echo "  Model: ${MODEL_DIR}"
echo "  Fixtures: ${FIXTURES}"
echo "  Sidecar: ${SIDECAR_DIR}"
echo ""

cd "${REPO_DIR}"

# Rebuild native extension (ensure new .cu is included)
echo "Rebuilding native extension..."
rm -rf /tmp/lynn_engine_native_build/runtime
export LYNN_ENABLE_SM120A_FP4_MMA=1
${PYTHON} -c "import sys; sys.path.insert(0,'.'); from engine.native_cuda import load_lynn_native_extension; ext=load_lynn_native_extension(verbose=True); print('scalar:', hasattr(ext,'dense_fp4xfp8_scalar_reference')); print('mma:', hasattr(ext,'dense_fp4xfp8_mma_probe'))"

echo ""
echo "Running probe..."
${PYTHON} benchmarks/p191_qwen35_9b_dense_fp4x_fp8_cute_probe.py \
    --fixtures "${FIXTURES}" \
    --model-dir "${MODEL_DIR}" \
    --sidecar-dir "${SIDECAR_DIR}" \
    --layers 0,16 \
    --out "${OUT}"

echo ""
echo "Report: ${OUT}"
