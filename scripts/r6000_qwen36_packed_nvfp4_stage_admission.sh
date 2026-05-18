#!/usr/bin/env bash
# R6000 wrapper for p142 packed NVFP4 stage admission gate
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-/root/miniconda3/envs/lynn-engine/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/qwen36_35b}"

ARGS=(
    --report-dir "$REPORT_DIR"
)

# Optional explicit paths
[ -n "${P141_REPORT:-}" ]          && ARGS+=(--p141-report "$P141_REPORT")
[ -n "${P140_PACKED_REPORT:-}" ]   && ARGS+=(--p140-packed-report "$P140_PACKED_REPORT")
[ -n "${CANDIDATE_SUMMARY:-}" ]    && ARGS+=(--candidate-summary "$CANDIDATE_SUMMARY")

exec "$PY" benchmarks/p142_packed_nvfp4_stage_admission_gate.py "${ARGS[@]}"
