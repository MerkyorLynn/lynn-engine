#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 MoE Slot Repack + Contract Verification
# ─────────────────────────────────────────────────────────────────────────────
#
# Machine: RTX PRO 6000 Blackwell (R6000)
# Purpose: Repack p133 MoE fixtures into slot-ordered expert weights (p135),
#          then verify slot-only computation matches reference (p136).
#
# Prerequisites:
#   - p133 fixtures at $P133_FIXTURES
#   - Model at $MODEL_DIR
#   - Python env with: torch, safetensors, transformers
#
# Usage:
#   bash scripts/r6000_qwen36_moe_slot_repack.sh
#
# Outputs:
#   reports/qwen36_35b/p135_repacked_fixtures/  — slot repacked safetensors + manifest
#   reports/qwen36_35b/p136_slot_repack_contract_report.json — contract results
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
P133_FIXTURES="${P133_FIXTURES:-/root/autodl-tmp/reports/qwen36_35b/p133_fixtures_official_w4a16}"
P135_OUT="${P135_OUT:-/root/autodl-tmp/reports/qwen36_35b/p135_repacked_fixtures_official_w4a16}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_NATIVE_SLOT_CANDIDATE="${RUN_NATIVE_SLOT_CANDIDATE:-1}"
NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR:-/tmp/lynn_engine_native_build/slot_output_owned_bf16}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 MoE Slot Repack + Contract Verification"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " Repo:        ${REPO_DIR}"
echo " Python:      ${PY}"
echo " Model:       ${MODEL_DIR}"
echo " p133 input:  ${P133_FIXTURES}"
echo " p135 output: ${P135_OUT}"
echo " Device:      ${DEVICE}"
echo " Dtype:       ${DTYPE}"
echo " Candidate:   ${RUN_NATIVE_SLOT_CANDIDATE}"
echo ""

# ── Verify prerequisites ──
if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    echo "Set MODEL_DIR env var to the correct path."
    exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "ERROR: config.json not found in ${MODEL_DIR}"
    exit 1
fi

if [ ! -f "${P133_FIXTURES}/manifest.json" ]; then
    echo "ERROR: p133 manifest not found: ${P133_FIXTURES}/manifest.json"
    echo "Run p133 export first or set P133_FIXTURES env var."
    exit 1
fi

cd "${REPO_DIR}"

# Verify Python imports
echo "── Checking Python dependencies..."
"${PY}" -c "
import torch
import safetensors
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer
print(f'  torch:        {torch.__version__}')
print(f'  safetensors:  {safetensors.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
echo ""

# ── Verify scripts compile ──
echo "── Verifying Python syntax..."
"${PY}" -m py_compile benchmarks/p135_repack_moe_fixture_slots.py
echo "  p135: OK"
"${PY}" -m py_compile benchmarks/p136_moe_slot_repack_contract.py
echo "  p136: OK"
if [ "${RUN_NATIVE_SLOT_CANDIDATE}" = "1" ]; then
    "${PY}" -m py_compile benchmarks/candidates/native_slot_output_owned_bf16.py
    echo "  native_slot_output_owned_bf16: OK"
fi
echo ""

# ── Verify bash syntax ──
echo "── Verifying bash syntax..."
bash -n scripts/r6000_qwen36_moe_slot_repack.sh
echo "  script: OK"
echo ""

# ── Step 1: Slot repack (p135) ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " STEP 1: Slot repack (p135)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p "${P135_OUT}"
mkdir -p "${REPORT_DIR}"

P135_START=$(date +%s)

"${PY}" benchmarks/p135_repack_moe_fixture_slots.py \
    --fixtures "${P133_FIXTURES}" \
    --model-dir "${MODEL_DIR}" \
    --out "${P135_OUT}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"

P135_EXIT=$?
P135_END=$(date +%s)
P135_ELAPSED=$((P135_END - P135_START))

echo ""
echo "  p135 completed in ${P135_ELAPSED}s (exit code: ${P135_EXIT})"
echo ""

if [ "${P135_EXIT}" -ne 0 ]; then
    echo "ERROR: p135 slot repack failed."
    exit 1
fi

# Verify repacked fixtures exist
REPACK_COUNT=$(find "${P135_OUT}" -name "*.safetensors" | wc -l | tr -d ' ')
echo "  Repacked fixtures: ${REPACK_COUNT}"

if [ "${REPACK_COUNT}" -lt 1 ]; then
    echo "  ERROR: No repacked fixtures found."
    exit 1
fi

if [ ! -f "${P135_OUT}/manifest.json" ]; then
    echo "  ERROR: p135 manifest.json not created!"
    exit 1
fi
echo "  manifest.json: OK"
echo ""

# ── Step 2: Contract verification (p136) ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " STEP 2: Slot repack contract verification (p136)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Expected: ALL fixtures pass with max_abs <= 1e-3, cosine >= 0.999999"
echo "  Goal:     max_abs = 0 (exact match) preferred"
echo ""

P136_START=$(date +%s)

set +e
"${PY}" benchmarks/p136_moe_slot_repack_contract.py \
    --fixtures "${P135_OUT}" \
    --model-dir "${MODEL_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-abs-threshold 1e-3 \
    --cosine-threshold 0.999999 \
    --out "${REPORT_DIR}/p136_slot_repack_contract_report.json"
P136_EXIT=$?
set -e
P136_END=$(date +%s)
P136_ELAPSED=$((P136_END - P136_START))

echo ""
echo "  p136 completed in ${P136_ELAPSED}s (exit code: ${P136_EXIT})"
echo ""

# ── Step 3: Native slot output-owned candidate ──
NATIVE_EXIT=0
NATIVE_ELAPSED=0
NATIVE_REPORT="${REPORT_DIR}/native_slot_output_owned_bf16_report.json"

if [ "${RUN_NATIVE_SLOT_CANDIDATE}" = "1" ] && [ "${P136_EXIT}" -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════════════════"
    echo " STEP 3: Native slot output-owned BF16 candidate"
    echo "═══════════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Build dir: ${NATIVE_BUILD_DIR}"
    echo ""

    NATIVE_START=$(date +%s)
    set +e
    LYNN_NATIVE_CUDA_BUILD_DIR="${NATIVE_BUILD_DIR}" \
    "${PY}" benchmarks/candidates/native_slot_output_owned_bf16.py \
        --fixtures "${P135_OUT}" \
        --out "${NATIVE_REPORT}"
    NATIVE_EXIT=$?
    set -e
    NATIVE_END=$(date +%s)
    NATIVE_ELAPSED=$((NATIVE_END - NATIVE_START))

    echo ""
    echo "  native slot candidate completed in ${NATIVE_ELAPSED}s (exit code: ${NATIVE_EXIT})"
    echo ""
elif [ "${RUN_NATIVE_SLOT_CANDIDATE}" = "1" ]; then
    NATIVE_EXIT=99
    echo "  Skipping native slot candidate because p136 contract failed."
    echo ""
fi

# ── Step 4: Summary ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Slot repack (p135):         ${P135_ELAPSED}s"
echo "  Contract check (p136):      ${P136_ELAPSED}s"
echo "  Native slot candidate:      ${NATIVE_ELAPSED}s"
echo "  Fixtures repacked:          ${REPACK_COUNT} files"
echo ""

if [ "${P136_EXIT}" -eq 0 ]; then
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  CONTRACT: GREEN                                      │"
    echo "  │                                                        │"
    echo "  │  Slot-repacked fixtures reproduce reference exactly.   │"
    echo "  │  Candidate kernels can now consume slot-ordered        │"
    echo "  │  weights without dynamic expert gather.                │"
    echo "  └──────────────────────────────────────────────────────┘"
    echo ""
else
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  CONTRACT: RED                                        │"
    echo "  │                                                        │"
    echo "  │  Slot repack DOES NOT match reference.                 │"
    echo "  │  Investigate before using for kernel development.      │"
    echo "  └──────────────────────────────────────────────────────┘"
    echo ""
fi

echo "  Output files:"
echo "    ${P135_OUT}/manifest.json"
echo "    ${REPORT_DIR}/p136_slot_repack_contract_report.json"
if [ "${RUN_NATIVE_SLOT_CANDIDATE}" = "1" ]; then
    echo "    ${NATIVE_REPORT}"
fi
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

if [ "${P136_EXIT}" -ne 0 ]; then
    exit "${P136_EXIT}"
fi
exit "${NATIVE_EXIT}"
