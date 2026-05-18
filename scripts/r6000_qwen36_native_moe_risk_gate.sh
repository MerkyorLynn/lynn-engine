#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# R6000 Native MoE Candidate Risk Gate (P140)
# ─────────────────────────────────────────────────────────────────────────────
#
# Machine: RTX PRO 6000 Blackwell (R6000)
# Purpose: Read-only gate that consumes existing p136/candidate/p137/strict
#          reports and outputs a DEFAULT / AMBER / CLOSED verdict.
#
# Prerequisites:
#   - p136 slot-order contract report exists
#   - At least one native candidate report exists
#
# Usage:
#   bash scripts/r6000_qwen36_native_moe_risk_gate.sh
#
# Outputs:
#   reports/qwen36_35b/p140_native_moe_risk_gate.json
#   reports/qwen36_35b/p140_native_moe_risk_gate.md
#
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " R6000 Native MoE Candidate Risk Gate (P140)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " Repo:        ${REPO_DIR}"
echo " Python:      ${PY}"
echo " Report dir:  ${REPORT_DIR}"
echo ""

cd "${REPO_DIR}"

# ── Verify syntax ──
echo "── Verifying Python syntax..."
"${PY}" -m py_compile benchmarks/p140_native_moe_candidate_risk_gate.py
echo "  p140: OK"
echo ""

echo "── Verifying bash syntax..."
bash -n scripts/r6000_qwen36_native_moe_risk_gate.sh
echo "  script: OK"
echo ""

# ── Run gate ──
echo "═══════════════════════════════════════════════════════════════════════"
echo " Running P140 risk gate"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

set +e
"${PY}" benchmarks/p140_native_moe_candidate_risk_gate.py \
    --report-dir "${REPORT_DIR}" \
    --out "${REPORT_DIR}/p140_native_moe_risk_gate.json" \
    --md-out "${REPORT_DIR}/p140_native_moe_risk_gate.md"
GATE_EXIT=$?
set -e

echo ""
echo "  Gate exit code: ${GATE_EXIT}"
echo ""

# ── Print verdict from JSON ──
if [ -f "${REPORT_DIR}/p140_native_moe_risk_gate.json" ]; then
    echo "── Verdict summary ──"
    "${PY}" - "${REPORT_DIR}/p140_native_moe_risk_gate.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  Tier:                    {d['verdict']}")
print(f"  Recommend P37 exploratory: {d['recommend_p37_exploratory']}")
for r in d.get("reasons", []):
    print(f"  Reason:  {r}")
for a in d.get("annotations", []):
    print(f"  Note:    {a}")
PY
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Output files:"
echo "   ${REPORT_DIR}/p140_native_moe_risk_gate.json"
echo "   ${REPORT_DIR}/p140_native_moe_risk_gate.md"
echo "═══════════════════════════════════════════════════════════════════════"

exit "${GATE_EXIT}"
