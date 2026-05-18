#!/usr/bin/env bash
# Spark-side watcher that runs the official Qwen3.6-35B-A3B Lynn-native W4A16
# NVFP4 MMLU/GPQA eval as soon as the R6000 artifact has been copied over.

set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/home/merkyor/quality-eval-20260517/results}"
RUNNER="${RUNNER:-/home/merkyor/quality-eval-20260517/scripts/run_candidate_eval.sh}"
NVFP4_MODEL="${NVFP4_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000}"
CAND="${CAND:-nvfp4-qwen36-w4a16-r6000}"
SERVED_NAME="${SERVED_NAME:-Qwen3.6-35B-A3B-W4A16-NVFP4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOCK_DIR="${LOCK_DIR:-/tmp/lynn-qwen36-nvfp4-eval.lock}"
EXTRA_ENV="${EXTRA_ENV:-LYNN_PACKED_DECODE=0 LYNN_PACKED_DECODE_FULL_ATTN=0 LYNN_PACKED_DECODE_LINEAR_ATTN=0 LYNN_PACKED_SHARED_EXPERT=0 LYNN_LINEAR_BLOCK_GRAPH=0 LYNN_LINEAR_BLOCK_GRAPH_REUSE=0 LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0 LYNN_MOE_FAST_FIXED=0}"

log() {
    printf '[nvfp4-eval-watch] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

has_summary() {
    [ -s "$RESULTS_DIR/mmlu_${CAND}_n500.summary.json" ] && [ -s "$RESULTS_DIR/gpqa_${CAND}.summary.json" ]
}

nvfp4_ready() {
    MODEL_DIR="$NVFP4_MODEL" python3 - <<'PY'
import json
import os
import pathlib

d = pathlib.Path(os.environ['MODEL_DIR'])
index_path = d / 'model.safetensors.index.json'
manifest_path = d / 'lynn_quant_manifest.json'
if not index_path.exists() or not manifest_path.exists():
    raise SystemExit(1)
index = json.loads(index_path.read_text())
manifest = json.loads(manifest_path.read_text())
files = set(index.get('weight_map', {}).values())
if not files:
    raise SystemExit(2)
if any(not (d / name).exists() or (d / name).stat().st_size <= 0 for name in files):
    raise SystemExit(3)
if int(manifest.get('quantized_count', 0)) <= 0:
    raise SystemExit(4)
PY
}

wait_for_eval_slot() {
    while true; do
        if sudo docker ps --format '{{.Names}}' | grep -Eq '^(lynn-eval-|lynn-eval-llamacpp)'; then
            log "eval container active; waiting"
            sleep "$POLL_SECONDS"
            continue
        fi
        if pgrep -af 'run_candidate_eval.sh|run_llamacpp_eval.sh|imatrix_q4_evals.sh' >/dev/null 2>&1; then
            log "eval driver active; waiting"
            sleep "$POLL_SECONDS"
            continue
        fi
        return 0
    done
}

while ! has_summary; do
    if nvfp4_ready; then
        log "NVFP4 artifact ready: $NVFP4_MODEL"
        break
    fi
    log "waiting for NVFP4 artifact: $NVFP4_MODEL"
    sleep "$POLL_SECONDS"
done

if has_summary; then
    log "$CAND summaries already exist; done"
    exit 0
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "another NVFP4 eval watcher owns $LOCK_DIR; exiting"
    exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

wait_for_eval_slot
if has_summary; then
    log "$CAND summaries appeared while waiting; done"
    exit 0
fi

LOG_PATH="$RESULTS_DIR/${CAND}_watch_$(date '+%Y%m%d_%H%M%S').log"
log "launching $CAND eval -> $LOG_PATH"
bash "$RUNNER" "$CAND" "$NVFP4_MODEL" "$SERVED_NAME" "$EXTRA_ENV" "bfloat16" >"$LOG_PATH" 2>&1
log "$CAND eval finished"
