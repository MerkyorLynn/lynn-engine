#!/usr/bin/env bash
# Build Qwen3.6-35B-A3B Q4_K_M imatrix GGUF while preserving embedded APEX-MTP.
#
# This script intentionally uses distinct output filenames from the older
# base-only Q4_K_M backstop so an MTP-capable artifact cannot be confused with a
# plain GGUF. GPU-heavy stages are opt-in by default to avoid disturbing the
# production 35B fallback service and voice services on Spark.

set -euo pipefail

BF16_MODEL="${BF16_MODEL:-/home/merkyor/models/Qwen3.6-35B-A3B-BF16-official-n5}"
OUT_DIR="${OUT_DIR:-/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-Q4KM-imatrix}"
CALIB="${CALIB:-/home/merkyor/calibration/calib_combined.txt}"
LLAMA_CPP="${LLAMA_CPP:-/home/merkyor/build/llama.cpp}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_CPP/build-cuda-sm121/bin}"
CONVERT="${CONVERT:-$LLAMA_CPP/convert_hf_to_gguf.py}"
GGUF_PY="${GGUF_PY:-$LLAMA_CPP/gguf-py}"
LOG_DIR="${LOG_DIR:-/home/merkyor/reports/qwen36_apex_mtp_q4km_$(date '+%Y%m%d_%H%M%S')}"
RUN_GPU_STAGES="${RUN_GPU_STAGES:-0}"
CONVERT_MODE="${CONVERT_MODE:-auto}"
CONVERT_IMAGE="${CONVERT_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-cuda}"
STOP_FALLBACK_FOR_GPU="${STOP_FALLBACK_FOR_GPU:-0}"
FALLBACK_SERVICE="${FALLBACK_SERVICE:-lynn-apex-mtp-llamacpp.service}"
FALLBACK_HEALTH_URL="${FALLBACK_HEALTH_URL:-http://127.0.0.1:18098/health}"
IMATRIX_CHUNKS="${IMATRIX_CHUNKS:-200}"
IMATRIX_CTX="${IMATRIX_CTX:-512}"
IMATRIX_NGL="${IMATRIX_NGL:-99}"

F16="$OUT_DIR/Qwen3.6-35B-A3B-APEX-MTP-F16.gguf"
IMATRIX="$OUT_DIR/Qwen3.6-35B-A3B-APEX-MTP.imatrix"
Q4="$OUT_DIR/Qwen3.6-35B-A3B-APEX-MTP-Q4_K_M-imatrix.gguf"
LOG="$LOG_DIR/build.log"
FALLBACK_WAS_ACTIVE=0

mkdir -p "$OUT_DIR" "$LOG_DIR"

log() {
    printf '[apex-mtp-q4km] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

restore_fallback() {
    if [ "$FALLBACK_WAS_ACTIVE" != "1" ]; then
        return 0
    fi
    log "restoring fallback service: $FALLBACK_SERVICE"
    sudo systemctl start "$FALLBACK_SERVICE" || true
    for _ in $(seq 1 90); do
        if curl -fsS --max-time 5 "$FALLBACK_HEALTH_URL" >/dev/null 2>&1; then
            log "fallback health restored: $FALLBACK_HEALTH_URL"
            return 0
        fi
        sleep 5
    done
    log "fallback health not restored within timeout; inspect: systemctl status $FALLBACK_SERVICE"
    return 1
}

stop_fallback_for_gpu() {
    if [ "$STOP_FALLBACK_FOR_GPU" != "1" ]; then
        log "STOP_FALLBACK_FOR_GPU=$STOP_FALLBACK_FOR_GPU; keeping fallback service running"
        return 0
    fi
    if systemctl is-active --quiet "$FALLBACK_SERVICE"; then
        FALLBACK_WAS_ACTIVE=1
        trap restore_fallback EXIT INT TERM
        log "stopping fallback service for GPU-heavy build stage: $FALLBACK_SERVICE"
        sudo systemctl stop "$FALLBACK_SERVICE"
    else
        log "fallback service already inactive: $FALLBACK_SERVICE"
    fi
}

need_file() {
    local path="$1"
    if [ ! -s "$path" ]; then
        log "missing required file: $path"
        return 2
    fi
}

preflight() {
    need_file "$BF16_MODEL/model.safetensors.index.json"
    need_file "$CONVERT"
    need_file "$LLAMA_BIN/llama-imatrix"
    need_file "$LLAMA_BIN/llama-quantize"
    need_file "$CALIB"
    if ! grep -q '"mtp\.' "$BF16_MODEL/model.safetensors.index.json"; then
        log "BF16 checkpoint does not contain mtp.* tensors; refusing base-only Q4_K_M build"
        return 3
    fi
    local shards
    shards="$(find "$BF16_MODEL" -maxdepth 1 -type f -name 'model-*.safetensors' | wc -l | tr -d ' ')"
    log "preflight ok: bf16=$BF16_MODEL shards=$shards out=$OUT_DIR run_gpu_stages=$RUN_GPU_STAGES"
    df -h "$OUT_DIR" | tee -a "$LOG"
    free -h | tee -a "$LOG"
    if command -v systemctl >/dev/null 2>&1; then
        log "fallback service: $(systemctl is-active lynn-apex-mtp-llamacpp.service 2>/dev/null || true)"
    fi
}

convert_f16() {
    if [ -s "$F16" ]; then
        log "F16 exists: $(du -sh "$F16" | awk '{print $1}')"
        return 0
    fi
    local tmp="$F16.tmp"
    rm -f "$tmp"
    log "converting MTP-enabled HF/BF16 checkpoint to F16 GGUF"
    local mode="$CONVERT_MODE"
    if [ "$mode" = "auto" ]; then
        if python3 - <<'PY' >/dev/null 2>&1
import torch, safetensors
PY
        then
            mode="local"
        else
            mode="docker"
        fi
    fi
    log "convert mode: $mode"
    if [ "$mode" = "docker" ]; then
        case "$BF16_MODEL" in
            /home/merkyor/models/*) ;;
            *) log "docker convert requires BF16_MODEL under /home/merkyor/models"; return 5 ;;
        esac
        case "$OUT_DIR" in
            /home/merkyor/models/*) ;;
            *) log "docker convert requires OUT_DIR under /home/merkyor/models"; return 5 ;;
        esac
        nice -n 10 ionice -c2 -n7 \
            docker run --rm \
            -v /home/merkyor/models:/models \
            -v "$LLAMA_CPP:/src:ro" \
            -v "$LOG_DIR:/log" \
            --entrypoint python3 \
            "$CONVERT_IMAGE" \
            /src/convert_hf_to_gguf.py \
            "/models/$(basename "$BF16_MODEL")" \
            --outfile "/models/$(basename "$OUT_DIR")/$(basename "$tmp")" \
            --outtype f16 2>&1 | tee -a "$LOG"
    else
        PYTHONPATH="$GGUF_PY:${PYTHONPATH:-}" \
            nice -n 10 ionice -c2 -n7 \
            python3 "$CONVERT" "$BF16_MODEL" \
            --outfile "$tmp" \
            --outtype f16 2>&1 | tee -a "$LOG"
    fi
    test -s "$tmp"
    mv "$tmp" "$F16"
    log "F16 ready: $(du -sh "$F16" | awk '{print $1}')"
}

dump_meta() {
    local model="$1"
    local out="$2"
    PYTHONPATH="$GGUF_PY:${PYTHONPATH:-}" \
        python3 -m gguf.scripts.gguf_dump --json "$model" > "$out"
}

verify_mtp_meta() {
    local model="$1"
    local name="$2"
    local meta="$LOG_DIR/${name}.metadata.json"
    dump_meta "$model" "$meta"
    python3 - "$meta" <<'PY' | tee -a "$LOG"
import json, sys
p = sys.argv[1]
d = json.load(open(p))
kv = d["metadata"]
def val(key):
    item = kv.get(key)
    return None if item is None else item.get("value")
for key in (
    "general.file_type",
    "qwen35moe.nextn_predict_layers",
    "qwen35moe.block_count",
    "qwen35moe.expert_count",
    "quantize.imatrix.entries_count",
    "quantize.imatrix.chunks_count",
):
    print(f"{key}={val(key)}")
if val("qwen35moe.nextn_predict_layers") != 1:
    raise SystemExit("missing qwen35moe.nextn_predict_layers=1")
tensors = d.get("tensors", {})
if isinstance(tensors, dict):
    names = list(tensors)
else:
    names = [x.get("name", "") for x in tensors if isinstance(x, dict)]
nextn = [n for n in names if "nextn" in n or "mtp" in n]
print(f"nextn_tensor_count={len(nextn)}")
for n in nextn[:12]:
    print(f"nextn_tensor={n}")
if not nextn:
    raise SystemExit("missing nextn/mtp tensor names in GGUF tensor table")
PY
    log "$name MTP metadata verified"
}

run_imatrix() {
    if [ -s "$IMATRIX" ]; then
        log "imatrix exists: $(du -sh "$IMATRIX" | awk '{print $1}')"
        return 0
    fi
    local tmp="$IMATRIX.tmp"
    rm -f "$tmp"
    log "running imatrix calibration chunks=$IMATRIX_CHUNKS ctx=$IMATRIX_CTX ngl=$IMATRIX_NGL"
    nice -n 10 ionice -c2 -n7 \
        "$LLAMA_BIN/llama-imatrix" \
        -m "$F16" \
        -f "$CALIB" \
        -o "$tmp" \
        --chunks "$IMATRIX_CHUNKS" \
        -ngl "$IMATRIX_NGL" \
        -c "$IMATRIX_CTX" 2>&1 | tee -a "$LOG"
    test -s "$tmp"
    mv "$tmp" "$IMATRIX"
    log "imatrix ready: $(du -sh "$IMATRIX" | awk '{print $1}')"
}

quant_q4() {
    if [ -s "$Q4" ]; then
        log "Q4_K_M exists: $(du -sh "$Q4" | awk '{print $1}')"
        return 0
    fi
    local tmp="$Q4.tmp"
    rm -f "$tmp"
    log "quantizing Q4_K_M with imatrix"
    nice -n 10 ionice -c2 -n7 \
        "$LLAMA_BIN/llama-quantize" \
        --imatrix "$IMATRIX" \
        "$F16" \
        "$tmp" \
        Q4_K_M 2>&1 | tee -a "$LOG"
    test -s "$tmp"
    mv "$tmp" "$Q4"
    log "Q4_K_M ready: $(du -sh "$Q4" | awk '{print $1}')"
}

main() {
    preflight
    convert_f16
    verify_mtp_meta "$F16" "f16"
    if [ "$RUN_GPU_STAGES" != "1" ]; then
        log "RUN_GPU_STAGES=$RUN_GPU_STAGES; stopping after F16/MTP verification"
        log "resume with: RUN_GPU_STAGES=1 bash $0"
        return 0
    fi
    stop_fallback_for_gpu
    run_imatrix
    quant_q4
    verify_mtp_meta "$Q4" "q4km"
    log "complete: $Q4"
}

main "$@"
