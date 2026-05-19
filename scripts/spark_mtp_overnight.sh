#!/usr/bin/env bash
# Spark long-form MTP K=1 speculative smoke runner — designed for ~1 hour
# wall budget, robust to SSH disconnect (setsid + nohup + PPID=1 detach).
#
# Usage (run from any shell on Spark; you can disconnect after launch):
#
#   chmod +x scripts/spark_mtp_overnight.sh
#   MODEL=/home/merkyor/models/<lynn-native-w4a16-nvfp4-dir> \
#   SIDECAR=/home/merkyor/models/mtp_sidecars/<...>/mtp.safetensors \
#   nohup setsid bash scripts/spark_mtp_overnight.sh > /dev/null 2>&1 < /dev/null &
#   disown
#
# Then check progress:
#   tail -f /home/merkyor/reports/mtp_overnight/latest/run.log
#
# Output goes to /home/merkyor/reports/mtp_overnight/<TS>/
# A symlink "latest" always points to the most recent run.

set -euo pipefail

# ---------- config ----------
REPO=${REPO:-/home/merkyor/lynn-engine}
PY=${PY:-python3}
MODEL=${MODEL:?MODEL env var must point to Lynn-native W4A16 NVFP4 model dir}
SIDECAR=${SIDECAR:?SIDECAR env var must point to mtp.safetensors}
PROMPTS_JSON=${PROMPTS_JSON:-$REPO/scripts/mtp_smoke_prompts.json}
MAX_NEW=${MAX_NEW:-256}
REPORT_ROOT=${REPORT_ROOT:-/home/merkyor/reports/mtp_overnight}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$REPORT_ROOT/$TS
LOG=$RUN_DIR/run.log

mkdir -p "$RUN_DIR"
# Relative symlink so host bind mounts resolve it correctly (container's
# absolute /reports/<TS> would be unreachable from host /home/merkyor/reports/).
ln -sfn "$TS" "$REPORT_ROOT/latest"

# Redirect all subsequent output to the log file
exec > "$LOG" 2>&1

echo "[overnight] start $(date)"
echo "[overnight] REPO=$REPO"
echo "[overnight] MODEL=$MODEL"
echo "[overnight] SIDECAR=$SIDECAR"
echo "[overnight] PROMPTS_JSON=$PROMPTS_JSON"
echo "[overnight] MAX_NEW=$MAX_NEW"
echo "[overnight] RUN_DIR=$RUN_DIR"
echo "[overnight] PPID=$PPID PID=$$"

# ---------- env ----------
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Lynn-native W4A16 NVFP4 production env (matches Spark Config D)
export LYNN_MOE_IMPL=${LYNN_MOE_IMPL:-packed_nvfp4}
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=${LYNN_LINEAR_ATTN_RECURRENT_INPLACE:-1}
export LYNN_NATIVE_FP4_LM_HEAD=${LYNN_NATIVE_FP4_LM_HEAD:-1}
export LYNN_PACKED_DECODE_BACKEND=${LYNN_PACKED_DECODE_BACKEND:-native_fast_2d}
# Speculative env knobs set per-config inside the python runner.

echo "[overnight] python: $($PY --version 2>&1)"
echo "[overnight] torch:  $($PY -c 'import torch; print(torch.__version__, "cuda?", torch.cuda.is_available())' 2>&1)"
echo "---"

# ---------- run ----------
OUT_JSON=$RUN_DIR/mtp_smoke.json
$PY -u scripts/spark_mtp_speculative_smoke.py \
    --model "$MODEL" \
    --sidecar "$SIDECAR" \
    --prompts-json "$PROMPTS_JSON" \
    --max-new "$MAX_NEW" \
    --out "$OUT_JSON" \
    < /dev/null

EXIT=$?
echo "---"
echo "[overnight] smoke exit=$EXIT"

# ---------- quick summary ----------
if [ -f "$OUT_JSON" ]; then
    echo "[overnight] === SUMMARY ==="
    $PY -u - <<EOF
import json, sys
report = json.load(open("$OUT_JSON"))
summary = report.get("summary", {})
for label, s in summary.items():
    print(f"[{label}]")
    print(f"  exact_match_rate         = {s.get('exact_match_rate')}")
    print(f"  mean_decode_tps_baseline = {s.get('mean_decode_tps_baseline_loop')}")
    print(f"  mean_spec_effective_tps  = {s.get('mean_spec_effective_tps')}")
    print(f"  mean_spec_accept_rate    = {s.get('mean_spec_accept_rate')}")
    print(f"  mean_shadow_accept_rate  = {s.get('mean_shadow_accept_rate')}")
gates = report.get("gates", {})
print()
print(f"correctness_spec_k1_matches_baseline = {gates.get('correctness_spec_k1_matches_baseline')}")
print(f"tps_ratio_spec_over_baseline         = {gates.get('tps_ratio_spec_over_baseline')}")
EOF
fi

echo "[overnight] end $(date)"
exit $EXIT
