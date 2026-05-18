#!/usr/bin/env bash
# R6000 wrapper for p143 resident P37 admission gate
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen36_35b}"

ARGS=(
    --report-dir "$REPORT_DIR"
)

[ -n "${STAGE_REPORT:-}" ] && ARGS+=(--stage-report "$STAGE_REPORT")
[ -n "${P37_REPORT:-}" ]   && ARGS+=(--p37-report "$P37_REPORT")

exec "$PY" benchmarks/p143_resident_p37_admission_gate.py "${ARGS[@]}"
