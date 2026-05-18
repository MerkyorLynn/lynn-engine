#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 Unified Native MoE Candidate Summary
# ─────────────────────────────────────────────────────────────────────────────
#
# Aggregates all native MoE candidate reports into a single summary table.
# Optionally merges reports from extra directories (e.g. other worktrees).
#
# Usage:
#   bash scripts/r6000_qwen36_native_moe_candidate_summary.sh
#
# Env overrides:
#   EXTRA_REPORT_DIR  — colon-separated list of additional report dirs
#
# Outputs:
#   reports/qwen36_35b/native_moe_candidate_summary.json
#   reports/qwen36_35b/native_moe_candidate_summary.md
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"
EXTRA_REPORT_DIR="${EXTRA_REPORT_DIR:-}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 Unified Native MoE Candidate Summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " Repo:        ${REPO_DIR}"
echo " Python:      ${PY}"
echo " Report dir:  ${REPORT_DIR}"
echo " Extra dirs:  ${EXTRA_REPORT_DIR:-<none>}"
echo ""

cd "${REPO_DIR}"

echo "── Verifying Python syntax..."
"${PY}" -m py_compile scripts/summarize_qwen36_native_moe_candidates.py
echo "  summarize: OK"

echo "── Verifying bash syntax..."
bash -n scripts/r6000_qwen36_native_moe_candidate_summary.sh
echo "  script: OK"
echo ""

# Build --extra-report-dir args from colon-separated EXTRA_REPORT_DIR
EXTRA_ARGS=""
if [[ -n "${EXTRA_REPORT_DIR}" ]]; then
    IFS=':' read -ra _EXTRA_DIRS <<< "${EXTRA_REPORT_DIR}"
    for _d in "${_EXTRA_DIRS[@]}"; do
        if [[ -d "${_d}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --extra-report-dir ${_d}"
        else
            echo "  WARN: extra dir not found, skipping: ${_d}"
        fi
    done
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo " Running candidate summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# shellcheck disable=SC2086
"${PY}" scripts/summarize_qwen36_native_moe_candidates.py \
    --report-dir "${REPORT_DIR}" \
    ${EXTRA_ARGS} \
    --out "${REPORT_DIR}/native_moe_candidate_summary.json" \
    --md-out "${REPORT_DIR}/native_moe_candidate_summary.md"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Output:"
echo "   ${REPORT_DIR}/native_moe_candidate_summary.json"
echo "   ${REPORT_DIR}/native_moe_candidate_summary.md"
echo "═══════════════════════════════════════════════════════════════════════"
