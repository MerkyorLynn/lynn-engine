#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 Qwen3.6 Active-MoE Fixture Candidate Gate
# ─────────────────────────────────────────────────────────────────────────────
#
# Purpose:
#   Standardize the first admission gate for native active-MoE candidates.
#   Candidates should pass this fixture gate before spending R6000 time on
#   full P37/P25/structured service gates.
#
# Usage examples:
#
#   # Stored fixture self-check, full MoE output.
#   bash scripts/r6000_qwen36_moe_fixture_gate.sh
#
#   # Stored fixture self-check, routed experts only.
#   ROUTED_ONLY=1 bash scripts/r6000_qwen36_moe_fixture_gate.sh
#
#   # Compare precomputed native outputs written as safetensors.
#   CANDIDATE_OUTPUT_DIR=/tmp/native_outputs \
#     bash scripts/r6000_qwen36_moe_fixture_gate.sh
#
#   # Compare a Python candidate backend from benchmarks/candidates/<name>.py.
#   CANDIDATE_BACKEND=native_grouped \
#     MAX_ABS_THRESHOLD=0.001 \
#     COSINE_THRESHOLD=0.999999 \
#     bash scripts/r6000_qwen36_moe_fixture_gate.sh
#
# Candidate-output contract:
#   The candidate directory may mirror fixture filenames and contain one of:
#   `moe_output`, `routed_output`, `candidate_output`, or `output`.
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
FIXTURE_DIR="${FIXTURE_DIR:-${REPORT_DIR}/p133_fixtures_official_w4a16}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
ROUTED_ONLY="${ROUTED_ONLY:-0}"
CANDIDATE_BACKEND="${CANDIDATE_BACKEND:-}"
CANDIDATE_OUTPUT_DIR="${CANDIDATE_OUTPUT_DIR:-}"
MAX_ABS_THRESHOLD="${MAX_ABS_THRESHOLD:-0.0}"
COSINE_THRESHOLD="${COSINE_THRESHOLD:-0.999999}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"

if [ -n "${CANDIDATE_BACKEND}" ] && [ -n "${CANDIDATE_OUTPUT_DIR}" ]; then
    echo "ERROR: set only one of CANDIDATE_BACKEND or CANDIDATE_OUTPUT_DIR" >&2
    exit 2
fi

if [ -n "${CANDIDATE_BACKEND}" ]; then
    CANDIDATE_LABEL="${CANDIDATE_BACKEND}"
elif [ -n "${CANDIDATE_OUTPUT_DIR}" ]; then
    CANDIDATE_LABEL="candidate_output"
else
    CANDIDATE_LABEL="selfcheck"
fi

if [ "${ROUTED_ONLY}" = "1" ]; then
    MODE_LABEL="routed_only"
else
    MODE_LABEL="full_moe"
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-${REPORT_DIR}/p134_${MODE_LABEL}_${CANDIDATE_LABEL}_${STAMP}.json}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 Qwen3.6 Active-MoE Fixture Candidate Gate"
echo "═══════════════════════════════════════════════════════════════════════"
echo " Repo:       ${REPO_DIR}"
echo " Fixtures:   ${FIXTURE_DIR}"
echo " Model:      ${MODEL_DIR}"
echo " Candidate:  ${CANDIDATE_LABEL}"
echo " Mode:       ${MODE_LABEL}"
echo " Thresholds: max_abs<=${MAX_ABS_THRESHOLD}, cosine>=${COSINE_THRESHOLD}"
echo " Output:     ${OUT}"
echo ""

if [ ! -d "${REPO_DIR}" ]; then
    echo "ERROR: repo directory not found: ${REPO_DIR}" >&2
    exit 1
fi
if [ ! -f "${FIXTURE_DIR}/manifest.json" ]; then
    echo "ERROR: fixture manifest not found: ${FIXTURE_DIR}/manifest.json" >&2
    echo "Run scripts/r6000_export_qwen36_moe_fixtures.sh first." >&2
    exit 1
fi
if [ -z "${CANDIDATE_OUTPUT_DIR}" ] && [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: model directory not found: ${MODEL_DIR}" >&2
    exit 1
fi
if [ -n "${CANDIDATE_OUTPUT_DIR}" ] && [ ! -d "${CANDIDATE_OUTPUT_DIR}" ]; then
    echo "ERROR: candidate output directory not found: ${CANDIDATE_OUTPUT_DIR}" >&2
    exit 1
fi

cd "${REPO_DIR}"

"${PY}" -m py_compile benchmarks/p134_active_moe_fixture_contract.py
mkdir -p "$(dirname "${OUT}")"

ARGS=(
    benchmarks/p134_active_moe_fixture_contract.py
    --fixtures "${FIXTURE_DIR}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --max-abs-threshold "${MAX_ABS_THRESHOLD}"
    --cosine-threshold "${COSINE_THRESHOLD}"
    --warmup "${WARMUP}"
    --iters "${ITERS}"
    --out "${OUT}"
)

if [ "${ROUTED_ONLY}" = "1" ]; then
    ARGS+=(--routed-only)
fi
if [ -n "${CANDIDATE_OUTPUT_DIR}" ]; then
    ARGS+=(--candidate-output-dir "${CANDIDATE_OUTPUT_DIR}")
else
    ARGS+=(--model-dir "${MODEL_DIR}")
fi
if [ -n "${CANDIDATE_BACKEND}" ]; then
    ARGS+=(--candidate-backend "${CANDIDATE_BACKEND}")
fi

set +e
"${PY}" "${ARGS[@]}"
EXIT_CODE=$?
set -e

if [ "${EXIT_CODE}" -eq 0 ]; then
    echo ""
    echo "VERDICT: GREEN — fixture gate passed"
else
    echo ""
    echo "VERDICT: RED — fixture gate failed"
fi
echo "Report: ${OUT}"

exit "${EXIT_CODE}"
