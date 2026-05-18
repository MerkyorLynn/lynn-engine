#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 MoE Fixture Export + Contract Verification
# ─────────────────────────────────────────────────────────────────────────────
#
# Machine: RTX PRO 6000 Blackwell (R6000)
# Purpose: Export MoE fixtures from the full Qwen3.6-35B W4A16 model, then
#          verify the Triton reference contract (must be exact: max_abs=0).
#
# Prerequisites:
#   - Model at $MODEL_DIR
#   - Lynn engine repo at $REPO_DIR
#   - Python env with: torch, safetensors, transformers, triton
#
# Usage:
#   bash scripts/r6000_export_qwen36_moe_fixtures.sh
#
# Outputs:
#   reports/qwen36_35b/p133_fixtures/   — fixture safetensors + manifest
#   reports/qwen36_35b/p134_contract_report.json — contract test results
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──
REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN}"
FIXTURE_DIR="${REPO_DIR}/reports/qwen36_35b/p133_fixtures"
REPORT_DIR="${REPO_DIR}/reports/qwen36_35b"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

# Layers to export (covers early, middle, late layers; at least 9 layers)
LAYERS="0,4,8,16,20,28,32,36,39"

# Prompts (at least 2 per layer)
PROMPT_1="Hello"
PROMPT_2="The capital of France is"

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 MoE Fixture Export + Contract Verification"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " Repo:      ${REPO_DIR}"
echo " Model:     ${MODEL_DIR}"
echo " Fixtures:  ${FIXTURE_DIR}"
echo " Layers:    ${LAYERS}"
echo " Device:    ${DEVICE}"
echo " Dtype:     ${DTYPE}"
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

cd "${REPO_DIR}"

# Verify Python imports
echo "── Checking Python dependencies..."
python -c "
import torch
import safetensors
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer
print(f'  torch:        {torch.__version__}')
print(f'  safetensors:  {safetensors.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# ── Verify scripts compile ──
echo "── Verifying Python syntax..."
python -m py_compile benchmarks/p133_export_active_moe_fixtures.py
echo "  p133: OK"
python -m py_compile benchmarks/p134_active_moe_fixture_contract.py
echo "  p134: OK"
echo ""

# ── Step 1: Export fixtures ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " STEP 1: Exporting MoE fixtures (p133)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p "${FIXTURE_DIR}"
mkdir -p "${REPORT_DIR}"

P133_START=$(date +%s)

python benchmarks/p133_export_active_moe_fixtures.py \
    --model-dir "${MODEL_DIR}" \
    --layers "${LAYERS}" \
    --prompts "${PROMPT_1}" "${PROMPT_2}" \
    --out "${FIXTURE_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}"

P133_END=$(date +%s)
P133_ELAPSED=$((P133_END - P133_START))

echo ""
echo "  p133 completed in ${P133_ELAPSED}s"
echo ""

# Verify fixtures exist
FIXTURE_COUNT=$(find "${FIXTURE_DIR}" -name "*.safetensors" | wc -l | tr -d ' ')
echo "  Exported fixtures: ${FIXTURE_COUNT}"

if [ "${FIXTURE_COUNT}" -lt 18 ]; then
    echo "  WARNING: Expected at least 18 fixtures (9 layers × 2 prompts), got ${FIXTURE_COUNT}"
fi

if [ ! -f "${FIXTURE_DIR}/manifest.json" ]; then
    echo "  ERROR: manifest.json not created!"
    exit 1
fi
echo "  manifest.json: OK"
echo ""

# ── Step 2: Contract verification (Triton self-check) ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " STEP 2: Contract verification (p134 — Triton self-check)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Expected: ALL fixtures must pass with max_abs=0 (exact match)"
echo ""

P134_START=$(date +%s)

python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures "${FIXTURE_DIR}" \
    --model-dir "${MODEL_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-abs-threshold 0.0 \
    --cosine-threshold 1.0 \
    --out "${REPORT_DIR}/p134_triton_selfcheck_report.json"

P134_EXIT=$?
P134_END=$(date +%s)
P134_ELAPSED=$((P134_END - P134_START))

echo ""
echo "  p134 completed in ${P134_ELAPSED}s (exit code: ${P134_EXIT})"
echo ""

# ── Step 3: Summary ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Fixture export (p133):      ${P133_ELAPSED}s"
echo "  Contract check (p134):      ${P134_ELAPSED}s"
echo "  Fixtures:                   ${FIXTURE_COUNT} files"
echo ""

if [ "${P134_EXIT}" -eq 0 ]; then
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  CONTRACT: GREEN                                      │"
    echo "  │                                                        │"
    echo "  │  Triton reference reproduces all fixtures exactly.     │"
    echo "  │  These fixtures are now valid as Stream A native       │"
    echo "  │  grouped kernel admission gate.                        │"
    echo "  │                                                        │"
    echo "  │  Native kernel developers can now:                     │"
    echo "  │    1. Load fixture safetensors (~16 KB each)           │"
    echo "  │    2. Run their kernel on hidden_in + expert weights   │"
    echo "  │    3. Compare output against moe_output in fixture     │"
    echo "  │    4. Iterate without loading the full 35B model       │"
    echo "  └──────────────────────────────────────────────────────┘"
    echo ""
else
    echo "  ┌──────────────────────────────────────────────────────┐"
    echo "  │  CONTRACT: RED                                        │"
    echo "  │                                                        │"
    echo "  │  Triton reference DOES NOT reproduce stored fixtures.  │"
    echo "  │  Investigate before using fixtures for native kernel   │"
    echo "  │  contract testing.                                     │"
    echo "  └──────────────────────────────────────────────────────┘"
    echo ""
fi

echo "  Output files:"
echo "    ${FIXTURE_DIR}/manifest.json"
echo "    ${REPORT_DIR}/p134_triton_selfcheck_report.json"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"

exit ${P134_EXIT}
