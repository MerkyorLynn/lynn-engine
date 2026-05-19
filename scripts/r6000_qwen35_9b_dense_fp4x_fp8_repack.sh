#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# R6000 Qwen3.5-9B Dense FP4xFP8 Offline Repack
# ---------------------------------------------------------------------------
#
# Runs P192 repack + P192-B contract on R6000.
# Does NOT modify r6000-eval; uses the r6000-eval Python directly.
#
# Usage:
#   bash scripts/r6000_qwen35_9b_dense_fp4x_fp8_repack.sh
#
#   # Override paths
#   MODEL_DIR=/path/to/model SIDEcar_DIR=/path/to/sidecar \
#     bash scripts/r6000_qwen35_9b_dense_fp4x_fp8_repack.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
SIDECAR_DIR="${SIDECAR_DIR:-${REPORT_DIR}/p192_dense_fp4x_fp8_sidecar}"
CONTRACT_OUT="${CONTRACT_OUT:-${REPORT_DIR}/p192b_dense_fp4x_fp8_repack_contract.json}"
LAYERS="${LAYERS:-}"

echo "========================================================================"
echo " R6000 Qwen3.5-9B Dense FP4xFP8 Repack + Contract"
echo "========================================================================"
echo " Repo:       ${REPO_DIR}"
echo " Model:      ${MODEL_DIR}"
echo " Sidecar:    ${SIDECAR_DIR}"
echo " Contract:   ${CONTRACT_OUT}"
echo ""

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: repo directory not found: ${REPO_DIR}" >&2
    exit 1
fi
if [ ! -f "${PY}" ]; then
    echo "ERROR: python not found: ${PY}" >&2
    exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi

cd "${REPO_DIR}"

# Syntax check
${PY} -m py_compile benchmarks/p192_qwen35_9b_dense_fp4x_fp8_repack.py
${PY} -m py_compile benchmarks/p192b_qwen35_9b_dense_fp4x_fp8_repack_contract.py
bash -n scripts/r6000_qwen35_9b_dense_fp4x_fp8_repack.sh

mkdir -p "${SIDECAR_DIR}"
mkdir -p "$(dirname "${CONTRACT_OUT}")"

# ---------------------------------------------------------------------------
# Repack
# ---------------------------------------------------------------------------
echo "Running P192 repack..."
REPACK_ARGS=(
    benchmarks/p192_qwen35_9b_dense_fp4x_fp8_repack.py
    --model-dir "${MODEL_DIR}"
    --out-dir "${SIDECAR_DIR}"
)
if [ -n "${LAYERS}" ]; then
    REPACK_ARGS+=(--layers "${LAYERS}")
fi

set +e
"${PY}" "${REPACK_ARGS[@]}"
REPACK_EXIT=$?
set -e

if [ "${REPACK_EXIT}" -ne 0 ]; then
    echo "ERROR: P192 repack failed with exit code ${REPACK_EXIT}" >&2
    exit "${REPACK_EXIT}"
fi

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
echo ""
echo "Running P192-B contract..."
CONTRACT_ARGS=(
    benchmarks/p192b_qwen35_9b_dense_fp4x_fp8_repack_contract.py
    --model-dir "${MODEL_DIR}"
    --sidecar-dir "${SIDECAR_DIR}"
    --out "${CONTRACT_OUT}"
)
if [ -n "${LAYERS}" ]; then
    CONTRACT_ARGS+=(--layers "${LAYERS}")
fi

set +e
"${PY}" "${CONTRACT_ARGS[@]}"
CONTRACT_EXIT=$?
set -e

OVERALL=$(${PY} -c "import json,sys; print(json.load(open('${CONTRACT_OUT}'))['overall'])" 2>/dev/null || echo "UNKNOWN")

echo ""
echo "========================================================================"
echo " Sidecar:    ${SIDECAR_DIR}"
echo " Contract:   ${CONTRACT_OUT}"
echo " Overall:    ${OVERALL}"
if [ "${CONTRACT_EXIT}" -eq 0 ] && [ "${OVERALL}" = "GREEN" ]; then
    echo " VERDICT:    GREEN — repack contract passed"
else
    echo " VERDICT:    RED — repack contract failed"
fi
echo "========================================================================"

exit "${CONTRACT_EXIT}"
