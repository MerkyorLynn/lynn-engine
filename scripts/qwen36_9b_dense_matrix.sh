#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.6 9B Dense Release Matrix Scaffold
# ─────────────────────────────────────────────────────────────────────────────
#
# Purpose: Generate / refresh the 9B dense release matrix Markdown report
#          from the JSON schema.  No model inference here — just report gen.
#
# Usage:
#   bash scripts/qwen36_9b_dense_matrix.sh
#
# Outputs:
#   docs/QWEN36_9B_DENSE_RELEASE_MATRIX_20260518.md
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3}"
JSON="${JSON:-${REPO_DIR}/reports/qwen36_9b/qwen36_9b_dense_matrix_schema_v1.json}"
OUT_MD="${OUT_MD:-${REPO_DIR}/docs/QWEN36_9B_DENSE_RELEASE_MATRIX_20260518.md}"

cd "${REPO_DIR}"

echo "═══════════════════════════════════════════════════════════════════════"
echo " Qwen3.6 9B Dense Release Matrix Scaffold"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo " JSON:  ${JSON}"
echo " OUT:   ${OUT_MD}"
echo ""

# Verify Python syntax
echo "── Verifying Python syntax..."
"${PY}" -m py_compile scripts/summarize_qwen36_9b_matrix.py
echo "  summarize_qwen36_9b_matrix.py: OK"

# Verify bash syntax
echo "── Verifying bash syntax..."
bash -n scripts/qwen36_9b_dense_matrix.sh
echo "  qwen36_9b_dense_matrix.sh: OK"
echo ""

# Generate report
echo "── Generating Markdown report..."
"${PY}" scripts/summarize_qwen36_9b_matrix.py \
    --json "${JSON}" \
    --out "${OUT_MD}"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " DONE"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Markdown: ${OUT_MD}"
echo "  JSON:     ${JSON}"
echo ""
