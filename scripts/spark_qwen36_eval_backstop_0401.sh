#!/usr/bin/env bash
# Spark-side backstop for official Qwen3.6-35B-A3B BF16 and Lynn-native W4A16
# NVFP4 MMLU/GPQA evals. It waits until the target clock time, then starts any
# missing evals sequentially using the existing quality harness.

set -euo pipefail

TARGET_HHMM="${TARGET_HHMM:-04:01}"
RESULTS_DIR="${RESULTS_DIR:-/home/merkyor/quality-eval-20260517/results}"
RUNNER="${RUNNER:-/home/merkyor/quality-eval-20260517/scripts/run_candidate_eval.sh}"
BF16_MODEL="${BF16_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-BF16-official-n5}"
NVFP4_MODEL="${NVFP4_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() {
    printf '[eval-backstop] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

seconds_now() {
    date +%s
}

seconds_target_today() {
    local today
    today="$(date '+%Y-%m-%d')"
    date -d "$today $TARGET_HHMM:00" +%s
}

wait_until_target() {
    local target now
    target="$(seconds_target_today)"
    now="$(seconds_now)"
    if [ "$now" -ge "$target" ]; then
        log "target $TARGET_HHMM already reached"
        return 0
    fi
    log "sleeping until $TARGET_HHMM"
    while [ "$(seconds_now)" -lt "$target" ]; do
        sleep "$POLL_SECONDS"
    done
}

has_summary() {
    local cand="$1"
    [ -s "$RESULTS_DIR/mmlu_${cand}_n500.summary.json" ] && [ -s "$RESULTS_DIR/gpqa_${cand}.summary.json" ]
}

bf16_ready() {
    [ -s "$BF16_MODEL/model.safetensors.index.json" ] || return 1
    local n
    n="$(find "$BF16_MODEL" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
    [ "$n" = "26" ]
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
        if sudo docker ps --format '{{.Names}}' | grep -q '^lynn-eval-'; then
            log "another lynn-eval container is active; waiting"
            sleep 120
            continue
        fi
        if pgrep -af '/home/merkyor/quality-eval-20260517/scripts/run_candidate_eval.sh' >/dev/null 2>&1; then
            log "run_candidate_eval.sh is active; waiting"
            sleep 120
            continue
        fi
        return 0
    done
}

run_one() {
    local cand="$1"
    local model_path="$2"
    local served_name="$3"
    local extra_env="$4"
    local dtype="$5"
    local log_path="$RESULTS_DIR/${cand}_backstop_$(date '+%Y%m%d_%H%M%S').log"

    if has_summary "$cand"; then
        log "$cand summaries already exist; skipping"
        return 0
    fi
    wait_for_eval_slot
    if has_summary "$cand"; then
        log "$cand summaries appeared while waiting; skipping"
        return 0
    fi
    log "launching $cand eval -> $log_path"
    bash "$RUNNER" "$cand" "$model_path" "$served_name" "$extra_env" "$dtype" >"$log_path" 2>&1
    log "$cand eval finished"
}

wait_until_target

if bf16_ready; then
    run_one "bf16-qwen36-official" "$BF16_MODEL" "Qwen3.6-35B-A3B-BF16-official" \
        "LYNN_MOE_IMPL=optimized LYNN_PACKED_DECODE=0 LYNN_PACKED_SHARED_EXPERT=0" "bfloat16"
else
    log "BF16 official package is not ready at $BF16_MODEL"
fi

if nvfp4_ready; then
    run_one "nvfp4-qwen36-w4a16-r6000" "$NVFP4_MODEL" "Qwen3.6-35B-A3B-W4A16-NVFP4" "" "bfloat16"
else
    log "NVFP4 package is not ready at $NVFP4_MODEL"
fi

log "backstop complete"
