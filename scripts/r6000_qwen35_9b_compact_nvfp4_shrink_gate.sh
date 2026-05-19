#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports}
P199_JSON=${P199_JSON:-$REPORT_ROOT/qwen35_9b/p199_nvfp4_size_audit_20260519_live_size2.json}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
REPORT_DIR=$REPORT_ROOT/qwen35_9b
OUT_JSON=$REPORT_DIR/compact_nvfp4_shrink_gate_${TS}.json
LOG=$REPORT_DIR/compact_nvfp4_shrink_gate_${TS}.log

mkdir -p "$REPORT_DIR"

cd "$REPO"

{
  echo "[compact-shrink-gate] start $(date)"
  echo "[compact-shrink-gate] p199=$P199_JSON"

  "$PY" scripts/qwen35_9b_compact_nvfp4_shrink_gate.py \
    --p199-json "$P199_JSON" \
    --out-json  "$OUT_JSON"

  echo "[compact-shrink-gate] report=$OUT_JSON"
  echo "[compact-shrink-gate] done $(date)"
} >> "$LOG" 2>&1

echo "$LOG"
