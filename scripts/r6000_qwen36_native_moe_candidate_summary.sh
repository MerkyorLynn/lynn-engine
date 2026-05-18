#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 Unified Native MoE Candidate Summary
# ─────────────────────────────────────────────────────────────────────────────
#
# Aggregates all native MoE candidate reports into a single summary table.
#
# Usage:
#   bash scripts/r6000_qwen36_native_moe_candidate_summary.sh
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

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 Unified Native MoE Candidate Summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " Repo:        ${REPO_DIR}"
echo " Python:      ${PY}"
echo " Report dir:  ${REPORT_DIR}"
echo ""

cd "${REPO_DIR}"

echo "── Verifying Python syntax..."
"${PY}" -m py_compile scripts/summarize_qwen36_native_moe_candidates.py
echo "  summarize: OK"

echo "── Verifying bash syntax..."
bash -n scripts/r6000_qwen36_native_moe_candidate_summary.sh
echo "  script: OK"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo " Running candidate summary"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

"${PY}" scripts/summarize_qwen36_native_moe_candidates.py \
    --report-dir "${REPORT_DIR}" \
    --out "${REPORT_DIR}/native_moe_candidate_summary.json" \
    --md-out "${REPORT_DIR}/native_moe_candidate_summary.md"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Output:"
echo "   ${REPORT_DIR}/native_moe_candidate_summary.json"
echo "   ${REPORT_DIR}/native_moe_candidate_summary.md"
echo "═══════════════════════════════════════════════════════════════════════"
