#!/usr/bin/env bash
# Spark-side low-priority Q4_K_M-imatrix backstop for official Qwen3.6-35B-A3B.
# It waits until the higher-priority BF16/NVFP4 quality runs have had the night
# slot, then attempts GGUF F16 conversion, imatrix calibration, Q4_K_M quant,
# and MMLU/GPQA evaluation through llama.cpp.

set -euo pipefail

TARGET_HHMM="${TARGET_HHMM:-05:30}"
BF16_WAIT_DEADLINE_HHMM="${BF16_WAIT_DEADLINE_HHMM:-06:30}"
NVFP4_WAIT_DEADLINE_HHMM="${NVFP4_WAIT_DEADLINE_HHMM:-07:00}"
BF16_MODEL="${BF16_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-BF16-official-n5}"
NVFP4_MODEL="${NVFP4_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000}"
OUT_DIR="${OUT_DIR:-/home/merkyor/models/Qwen3.6-35B-A3B-GGUF-imatrix}"
CALIB="${CALIB:-/home/merkyor/calibration/calib_combined.txt}"
RESULTS_DIR="${RESULTS_DIR:-/home/merkyor/quality-eval-20260517/results}"
LLAMACPP_EVAL="${LLAMACPP_EVAL:-/home/merkyor/quality-eval-20260517/scripts/run_llamacpp_eval.sh}"
P25="${P25:-/home/merkyor/lynn-engine/benchmarks/p25_server_decode_tps_probe.py}"
TPS_PORT="${TPS_PORT:-18096}"
TPS_MAX_TOKENS="${TPS_MAX_TOKENS:-128 256}"
TPS_RUNS="${TPS_RUNS:-1}"
POLL_SECONDS="${POLL_SECONDS:-120}"

F16="$OUT_DIR/Qwen3.6-35B-A3B-F16.gguf"
IMATRIX="$OUT_DIR/Qwen3.6-35B-A3B.imatrix"
Q4="$OUT_DIR/Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf"
LOG_DIR="/home/merkyor/reports/qwen36_q4km_imatrix_$(date '+%Y%m%d_%H%M')"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$RESULTS_DIR"
LOG="$LOG_DIR/pipeline.log"

log() {
    printf '[q4km-imatrix] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

sec_for_hhmm() {
    date -d "$(date '+%Y-%m-%d') $1:00" +%s
}

wait_until_hhmm() {
    local target now
    target="$(sec_for_hhmm "$1")"
    now="$(date +%s)"
    if [ "$now" -ge "$target" ]; then
        log "target $1 already reached"
        return 0
    fi
    log "sleeping until $1"
    while [ "$(date +%s)" -lt "$target" ]; do
        sleep "$POLL_SECONDS"
    done
}

bf16_ready() {
    [ -s "$BF16_MODEL/model.safetensors.index.json" ] || return 1
    local n
    n="$(find "$BF16_MODEL" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
    [ "$n" = "26" ]
}

nvfp4_ready() {
    [ -s "$NVFP4_MODEL/model.safetensors.index.json" ] && [ -s "$NVFP4_MODEL/lynn_quant_manifest.json" ]
}

summary_pair_exists() {
    local cand="$1"
    [ -s "$RESULTS_DIR/mmlu_${cand}_n500.summary.json" ] && [ -s "$RESULTS_DIR/gpqa_${cand}.summary.json" ]
}

wait_for_eval_slot() {
    while true; do
        if sudo docker ps --format '{{.Names}}' | grep -Eq '^(lynn-eval-|lynn-eval-llamacpp)'; then
            log "eval container active; waiting"
            sleep "$POLL_SECONDS"
            continue
        fi
        if pgrep -af 'run_candidate_eval.sh|run_llamacpp_eval.sh|imatrix_q4_evals.sh' >/dev/null 2>&1; then
            log "another eval driver active; waiting"
            sleep "$POLL_SECONDS"
            continue
        fi
        return 0
    done
}

wait_for_priority_quality() {
    local bf16_deadline nvfp4_deadline now
    bf16_deadline="$(sec_for_hhmm "$BF16_WAIT_DEADLINE_HHMM")"
    nvfp4_deadline="$(sec_for_hhmm "$NVFP4_WAIT_DEADLINE_HHMM")"
    while true; do
        now="$(date +%s)"
        if summary_pair_exists "bf16-qwen36-official"; then
            log "BF16 official summaries exist"
            break
        fi
        if [ "$now" -ge "$bf16_deadline" ]; then
            log "BF16 deadline reached without summaries; continuing as low-priority backstop"
            break
        fi
        log "waiting for BF16 official summaries before Q4_K_M work"
        sleep "$POLL_SECONDS"
    done
    while nvfp4_ready; do
        now="$(date +%s)"
        if summary_pair_exists "nvfp4-qwen36-w4a16-r6000"; then
            log "NVFP4 official summaries exist"
            break
        fi
        if [ "$now" -ge "$nvfp4_deadline" ]; then
            log "NVFP4 deadline reached without summaries; continuing"
            break
        fi
        log "NVFP4 artifact exists, waiting for its quality summaries first"
        sleep "$POLL_SECONDS"
    done
}

DOCKER_LLAMA="sudo docker run --rm --gpus all -v /home/merkyor/models:/models -v /home/merkyor/calibration:/calib -v $LOG_DIR:/log --entrypoint /app"
DOCKER_CONVERT="sudo docker run --rm -v /home/merkyor/models:/models -v $LOG_DIR:/log --entrypoint python3 ghcr.io/ggml-org/llama.cpp:full-cuda"

convert_f16() {
    if [ -s "$F16" ]; then
        log "F16 GGUF exists: $(du -sh "$F16" | awk '{print $1}')"
        return 0
    fi
    log "converting official BF16 HF package to F16 GGUF"
    $DOCKER_CONVERT /app/convert_hf_to_gguf.py \
        "/models/$(basename "$BF16_MODEL")" \
        --outfile "/models/$(basename "$OUT_DIR")/$(basename "$F16")" \
        --outtype f16 2>&1 | tee -a "$LOG"
    test -s "$F16"
    log "F16 GGUF ready: $(du -sh "$F16" | awk '{print $1}')"
}

run_imatrix() {
    if [ -s "$IMATRIX" ]; then
        log "imatrix exists: $(du -sh "$IMATRIX" | awk '{print $1}')"
        return 0
    fi
    if [ ! -s "$CALIB" ]; then
        log "missing calibration file: $CALIB"
        return 2
    fi
    log "running imatrix calibration; if llama.cpp still has the Qwen3.6 tokenizer issue this will fail visibly"
    $DOCKER_LLAMA/llama-imatrix ghcr.io/ggml-org/llama.cpp:full-cuda \
        -m "/models/$(basename "$OUT_DIR")/$(basename "$F16")" \
        -f "/calib/$(basename "$CALIB")" \
        -o "/models/$(basename "$OUT_DIR")/$(basename "$IMATRIX")" \
        --chunks 200 -ngl 99 -c 512 \
        2>&1 | tee -a "$LOG"
    test -s "$IMATRIX"
    log "imatrix ready: $(du -sh "$IMATRIX" | awk '{print $1}')"
}

quant_q4() {
    if [ -s "$Q4" ]; then
        log "Q4_K_M-imatrix GGUF exists: $(du -sh "$Q4" | awk '{print $1}')"
        return 0
    fi
    log "quantizing Q4_K_M with imatrix"
    $DOCKER_LLAMA/llama-quantize ghcr.io/ggml-org/llama.cpp:full-cuda \
        --imatrix "/models/$(basename "$OUT_DIR")/$(basename "$IMATRIX")" \
        "/models/$(basename "$OUT_DIR")/$(basename "$F16")" \
        "/models/$(basename "$OUT_DIR")/$(basename "$Q4")" \
        Q4_K_M 2>&1 | tee -a "$LOG"
    test -s "$Q4"
    log "Q4_K_M-imatrix GGUF ready: $(du -sh "$Q4" | awk '{print $1}')"
}

eval_q4() {
    if summary_pair_exists "qwen36-q4km-imatrix"; then
        log "Q4_K_M-imatrix summaries already exist"
        return 0
    fi
    log "running llama.cpp MMLU/GPQA for Q4_K_M-imatrix"
    bash "$LLAMACPP_EVAL" "qwen36-q4km-imatrix" "$Q4" 2>&1 | tee -a "$LOG"
}

bench_q4_tps() {
    local ts cont served server_log out ready smoke
    ts="$(date '+%Y%m%d_%H%M%S')"
    cont="qwen36-q4km-imatrix-tps-$ts"
    served="Qwen3.6-35B-A3B-Q4KM-imatrix"
    server_log="$RESULTS_DIR/qwen36_q4km_imatrix_llamacpp_server_${ts}.log"
    out="$RESULTS_DIR/qwen36_q4km_imatrix_llamacpp_p25_${ts}.json"
    if [ ! -s "$P25" ]; then
        log "missing P25 probe script: $P25"
        return 0
    fi
    log "running llama.cpp Q4_K_M-imatrix P25 TPS -> $out"
    sudo docker rm -f "$cont" 2>/dev/null || true
    sudo docker run -d --name "$cont" --network host --gpus all --shm-size 16g \
        -v "$(dirname "$Q4"):/gguf" \
        ghcr.io/ggml-org/llama.cpp:server-cuda \
        --model "/gguf/$(basename "$Q4")" \
        --host 0.0.0.0 --port "$TPS_PORT" \
        --n-gpu-layers 99 \
        --ctx-size 4096 \
        --threads 12 \
        --jinja \
        -a "$served" >"$server_log" 2>&1
    ready=0
    for _ in $(seq 1 90); do
        sleep 10
        smoke="$(curl -s -m 30 -H 'Content-Type: application/json' \
            -d '{"model":"'"$served"'","prompt":"A","max_tokens":4,"temperature":0}' \
            "http://127.0.0.1:$TPS_PORT/v1/completions" 2>&1 || true)"
        if echo "$smoke" | grep -q '"choices"'; then
            ready=1
            break
        fi
    done
    if [ "$ready" != 1 ]; then
        log "llama.cpp TPS server not ready; tailing logs"
        sudo docker logs --tail 80 "$cont" 2>&1 | tee -a "$LOG" || true
        sudo docker rm -f "$cont" 2>/dev/null || true
        return 0
    fi
    python3 "$P25" \
        --url "http://127.0.0.1:$TPS_PORT/v1" \
        --model "$served" \
        --max-tokens $TPS_MAX_TOKENS \
        --runs "$TPS_RUNS" \
        --out "$out" 2>&1 | tee -a "$LOG"
    sudo docker rm -f "$cont" 2>/dev/null || true
    log "llama.cpp Q4_K_M-imatrix TPS report=$out"
}

wait_until_hhmm "$TARGET_HHMM"
while ! bf16_ready; do
    log "waiting for official BF16 package: $BF16_MODEL"
    sleep "$POLL_SECONDS"
done
wait_for_priority_quality
wait_for_eval_slot
convert_f16
run_imatrix
quant_q4
wait_for_eval_slot
eval_q4
wait_for_eval_slot
bench_q4_tps
log "Q4_K_M-imatrix backstop complete"
