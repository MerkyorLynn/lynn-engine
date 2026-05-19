#!/usr/bin/env bash
# P193 · Qwen3.5-9B native boundary admission gate runner
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="${ROOT}/reports/qwen35_9b"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${REPORT_DIR}/p193_native_boundary_admission_${TIMESTAMP}.json"

mkdir -p "$REPORT_DIR"

echo "=== P193 Qwen3.5-9B Native Boundary Admission Gate ==="
echo "Report dir: $REPORT_DIR"
echo ""

# Count available upstream reports
FOUND=0
for prefix in p160_ p185_ p189_ p191_ p192b_ p192_; do
    COUNT=$(find "$REPORT_DIR" -maxdepth 1 -name "${prefix}*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$COUNT" -gt 0 ]; then
        LATEST=$(find "$REPORT_DIR" -maxdepth 1 -name "${prefix}*.json" -exec ls -t {} + 2>/dev/null | head -1)
        echo "  Found ${prefix%.json} report: $(basename "$LATEST")"
        FOUND=$((FOUND + 1))
    else
        echo "  No ${prefix%.json} report found (will be skipped)"
    fi
done
echo ""

if [ "$FOUND" -eq 0 ]; then
    echo "WARNING: No upstream reports found. Running gate with empty inputs."
    echo "  Run P160, P185, P189 first for meaningful admission decision."
    echo ""
fi

python3 "${ROOT}/benchmarks/p193_qwen35_9b_native_boundary_admission.py" \
    --report-dir "$REPORT_DIR" \
    --out "$OUT" \
    "$@"

echo ""
echo "=== Done ==="
