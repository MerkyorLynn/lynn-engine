#!/usr/bin/env bash
# P191: Run true-FP8 layer-mask summary on R6000
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
REPORT_DIR="/root/autodl-tmp/reports/qwen35_9b"
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.12}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

OUT_JSON="${REPORT_DIR}/qwen35_9b_true_fp8_layer_mask_summary.json"
OUT_MD="${REPORT_DIR}/QWEN35_9B_TRUE_FP8_LAYER_MASK_SUMMARY.md"

echo "P191: Summarizing true-FP8 layer-mask sweep results"
echo "  Reports: ${REPORT_DIR}/p190_*.json"
echo ""

cd "${REPO_DIR}"

P190_FILES=$(find "${REPORT_DIR}" -name "p190_qwen35_9b_true_fp8_resident_gate_*.json" 2>/dev/null | sort)

if [ -z "${P190_FILES}" ]; then
    echo "ERROR: no p190 reports found in ${REPORT_DIR}"
    exit 1
fi

COUNT=$(echo "${P190_FILES}" | wc -l | tr -d ' ')
echo "  Found ${COUNT} p190 report(s)"
echo ""

${PYTHON} scripts/summarize_qwen35_9b_true_fp8_layer_masks.py \
    --reports ${REPORT_DIR}/p190_qwen35_9b_true_fp8_resident_gate_*.json \
    --out-json "${OUT_JSON}" \
    --out-md "${OUT_MD}"

echo ""
echo "Done."
echo "  JSON: ${OUT_JSON}"
echo "  Markdown: ${OUT_MD}"
