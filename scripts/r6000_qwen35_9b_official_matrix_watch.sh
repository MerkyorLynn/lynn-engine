#!/usr/bin/env bash
#
# r6000_qwen35_9b_official_matrix_watch.sh
# Full benchmark harness for Qwen3.5-9B official BF16 / NVFP4 / Q4_K_M.
# Runs MMLU-500 5-shot, GPQA-diamond, and single TPS (128/256/512) per quant.
#
# DRY_RUN=1 by default (prints commands, no execution).
# GPU-aware: checks nvidia-smi; obeys SKIP_GPU=1; never competes with 35B P37.
# Q4_K_M GGUF not on disk -> PENDING, no download.
#
# Env vars:
#   MODEL_ROOT          — base model dir (default: /root/autodl-tmp/models)
#   BF16_MODEL          — BF16 path
#   NVFP4_MODEL         — NVFP4 path
#   Q4KM_GGUF           — Q4_K_M GGUF path
#   REPORT_ROOT         — report output root
#   REPORT_DIR          — per-model report dir
#   DRY_RUN             — 1 = print only (default)
#   SKIP_GPU            — 1 = skip all GPU benchmarks
#   GPU_MEM_THRESHOLD_MB — default 1000
#   PYTHON_BIN          — Python interpreter
#   MMLU_RUNNER         — path to mmlu_runner_v2.py or equivalent
#   GPQA_RUNNER         — path to gpqa_runner_v2.py or equivalent
#   P25_PROBE           — path to p25_server_decode_tps_probe.py
#   SERVER_MODULE       — Python module to start server (default: server.openai_http)
#   SERVER_HOST         — default 127.0.0.1
#   MMLU_DATASET        — MMLU dataset dir
#   GPQA_DATASET        — GPQA diamond CSV path
#   STAMP               — timestamp suffix
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
BF16_MODEL="${BF16_MODEL:-$MODEL_ROOT/Qwen3.5-9B-BF16}"
NVFP4_MODEL="${NVFP4_MODEL:-$MODEL_ROOT/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
Q4KM_GGUF="${Q4KM_GGUF:-$MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/qwen35_9b}"
DRY_RUN="${DRY_RUN:-1}"
SKIP_GPU="${SKIP_GPU:-0}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-1000}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MMLU_RUNNER="${MMLU_RUNNER:-}"
GPQA_RUNNER="${GPQA_RUNNER:-}"
P25_PROBE="${P25_PROBE:-$REPO_ROOT/benchmarks/p25_server_decode_tps_probe.py}"
SERVER_MODULE="${SERVER_MODULE:-server.openai_http}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
MMLU_DATASET="${MMLU_DATASET:-/tmp/datasets/mmlu}"
GPQA_DATASET="${GPQA_DATASET:-/tmp/datasets/gpqa/gpqa_diamond.csv}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="${LOG:-$REPORT_DIR/r6000_qwen35_9b_official_matrix_watch_${STAMP}.log}"
SUMMARIZER="${SCRIPT_DIR}/summarize_qwen35_9b_r6000_reports.py"

BF16_PORT="${BF16_PORT:-18170}"
NVFP4_PORT="${NVFP4_PORT:-18171}"
Q4KM_PORT="${Q4KM_PORT:-18172}"

# The report directory must exist before the final tee opens $LOG.  For local
# dry-runs on macOS the R6000 Python path may not exist, so use python3 for the
# summary-only stages while still printing the configured remote PYTHON_BIN.
mkdir -p "$REPORT_DIR" || true
PY_EXEC="$PYTHON_BIN"
if [[ ! -x "$PY_EXEC" ]]; then
    PY_EXEC="$(command -v python3 || true)"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log_info()  { printf "\033[1;34m[INFO]\033[0m  %s\n" "$*"; }
log_warn()  { printf "\033[1;33m[WARN]\033[0m  %s\n" "$*"; }
log_ok()    { printf "\033[1;32m[OK]\033[0m    %s\n" "$*"; }
log_dry()   { printf "\033[1;36m[DRY]\033[0m   %s\n" "$*"; }
log_err()   { printf "\033[1;31m[ERR]\033[0m   %s\n" "$*" >&2; }
log_block() { printf "\033[1;35m[BLOCKED]\033[0m %s\n" "$*"; }

run_or_dry() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        log_dry "$1"
    else
        log_info "RUN: $1"
        eval "$1"
    fi
}

dir_bytes() {
    local d="$1"
    if [[ -d "$d" ]]; then
        find "$d" -type f -print0 2>/dev/null | du --files0-from=- --total -b 2>/dev/null | tail -1 | awk '{print $1}'
    else
        echo 0
    fi
}

gpu_is_idle() {
    if [[ "${SKIP_GPU}" == "1" ]]; then
        log_info "SKIP_GPU=1 — treating GPU as busy"
        return 1
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_warn "nvidia-smi not found; assuming GPU idle (non-NVIDIA host?)"
        return 0
    fi
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s}')"
    if [[ -z "$used" || "$used" == "0" ]]; then
        log_ok "GPU idle (0 MB used)"
        return 0
    fi
    if [[ "$used" -gt "${GPU_MEM_THRESHOLD_MB}" ]]; then
        log_warn "GPU busy (${used} MB used > ${GPU_MEM_THRESHOLD_MB} MB threshold)"
        return 1
    fi
    log_ok "GPU lightly loaded (${used} MB used <= ${GPU_MEM_THRESHOLD_MB} MB threshold)"
    return 0
}

SERVER_PID=""
cleanup_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup_server EXIT

start_server() {
    local model="$1" served_name="$2" port="$3" server_log="$4" health_json="$5"
    log_info "Starting server: model=$model port=$port"
    if [[ "${DRY_RUN}" == "1" ]]; then
        log_dry "$PYTHON_BIN -m $SERVER_MODULE --model $model --served-name $served_name --host $SERVER_HOST --port $port --dtype bfloat16 > $server_log 2>&1 &"
        return 0
    fi
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    "$PYTHON_BIN" -m "$SERVER_MODULE" \
        --model "$model" --served-name "$served_name" --host "$SERVER_HOST" --port "$port" --dtype bfloat16 > "$server_log" 2>&1 &
    SERVER_PID=$!
    log_info "Server PID=$SERVER_PID log=$server_log"
    local ready=0
    for _ in $(seq 1 300); do
        if curl -fsS "http://${SERVER_HOST}:${port}/health" > "$health_json" 2>/dev/null; then
            ready=1; break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log_err "Server exited before ready"
            tail -120 "$server_log" || true
            return 1
        fi
        sleep 2
    done
    if [[ "$ready" != "1" ]]; then
        log_err "Server not ready in time"
        tail -120 "$server_log" || true
        return 1
    fi
    log_ok "Server ready on port $port"
    return 0
}

stop_server() {
    cleanup_server
    log_info "Server stopped"
}

run_mmlu() {
    local base_url="$1" model_name="$2" out_jsonl="$3" out_summary="$4"
    if [[ -z "$MMLU_RUNNER" || ! -f "$MMLU_RUNNER" ]]; then
        log_block "MMLU_RUNNER not found: ${MMLU_RUNNER:-(not set)}"
        return 1
    fi
    if [[ ! -d "$MMLU_DATASET" ]]; then
        log_block "MMLU dataset dir missing: $MMLU_DATASET"
        return 1
    fi
    run_or_dry "\"$PYTHON_BIN\" \"$MMLU_RUNNER\" --data-dir \"$MMLU_DATASET\" --base-url \"$base_url\" --model \"$model_name\" --out \"$out_jsonl\" --concurrency 4 --shots 5 --sample 500"
    if [[ "${DRY_RUN}" != "1" && -f "$out_summary" ]]; then
        log_ok "MMLU summary: $out_summary"
    fi
    return 0
}

run_gpqa() {
    local base_url="$1" model_name="$2" out_jsonl="$3" out_summary="$4"
    if [[ -z "$GPQA_RUNNER" || ! -f "$GPQA_RUNNER" ]]; then
        log_block "GPQA_RUNNER not found: ${GPQA_RUNNER:-(not set)}"
        return 1
    fi
    if [[ ! -f "$GPQA_DATASET" ]]; then
        log_block "GPQA dataset missing: $GPQA_DATASET"
        return 1
    fi
    run_or_dry "\"$PYTHON_BIN\" \"$GPQA_RUNNER\" --csv \"$GPQA_DATASET\" --base-url \"$base_url\" --model \"$model_name\" --out \"$out_jsonl\" --concurrency 2"
    if [[ "${DRY_RUN}" != "1" && -f "$out_summary" ]]; then
        log_ok "GPQA summary: $out_summary"
    fi
    return 0
}

run_tps() {
    local base_url="$1" model_name="$2" out_json="$3"
    if [[ ! -f "$P25_PROBE" ]]; then
        log_block "P25 probe not found: $P25_PROBE"
        return 1
    fi
    run_or_dry "\"$PYTHON_BIN\" \"$P25_PROBE\" --url \"$base_url\" --model \"$model_name\" --chat --max-tokens 128 256 512 --runs 3 --out \"$out_json\""
    return 0
}

# Per-quant state variables (bash 3.2 compatible — no associative arrays)
reset_quant_state() {
    STATUS_BF16="PENDING";  BLOCKED_BF16=""
    MMLU_SCORE_BF16="null"; MMLU_CORRECT_BF16="null"; MMLU_TOTAL_BF16="500"; MMLU_PATH_BF16=""
    GPQA_SCORE_BF16="null"; GPQA_CORRECT_BF16="null"; GPQA_TOTAL_BF16="null"; GPQA_PATH_BF16=""
    TPS_128_BF16="null"; TPS_256_BF16="null"; TPS_512_BF16="null"; TPS_PATH_BF16=""
    LOAD_SEC_BF16="null"; SIZE_GIB_BF16="null"

    STATUS_NVFP4="PENDING"; BLOCKED_NVFP4=""
    MMLU_SCORE_NVFP4="null"; MMLU_CORRECT_NVFP4="null"; MMLU_TOTAL_NVFP4="500"; MMLU_PATH_NVFP4=""
    GPQA_SCORE_NVFP4="null"; GPQA_CORRECT_NVFP4="null"; GPQA_TOTAL_NVFP4="null"; GPQA_PATH_NVFP4=""
    TPS_128_NVFP4="null"; TPS_256_NVFP4="null"; TPS_512_NVFP4="null"; TPS_PATH_NVFP4=""
    LOAD_SEC_NVFP4="null"; SIZE_GIB_NVFP4="null"

    STATUS_Q4KM="PENDING";  BLOCKED_Q4KM=""
    MMLU_SCORE_Q4KM="null"; MMLU_CORRECT_Q4KM="null"; MMLU_TOTAL_Q4KM="500"; MMLU_PATH_Q4KM=""
    GPQA_SCORE_Q4KM="null"; GPQA_CORRECT_Q4KM="null"; GPQA_TOTAL_Q4KM="null"; GPQA_PATH_Q4KM=""
    TPS_128_Q4KM="null"; TPS_256_Q4KM="null"; TPS_512_Q4KM="null"; TPS_PATH_Q4KM=""
    LOAD_SEC_Q4KM="null"; SIZE_GIB_Q4KM="null"
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
{
echo "═══════════════════════════════════════════════════════════════════════"
echo " Qwen3.5-9B Official Matrix Watch + Benchmark"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  BF16_MODEL:     ${BF16_MODEL}"
echo "  NVFP4_MODEL:    ${NVFP4_MODEL}"
echo "  Q4KM_GGUF:      ${Q4KM_GGUF}"
echo "  REPORT_DIR:     ${REPORT_DIR}"
echo "  DRY_RUN:        ${DRY_RUN}"
echo "  SKIP_GPU:       ${SKIP_GPU}"
echo "  GPU_THRESHOLD:  ${GPU_MEM_THRESHOLD_MB} MB"
echo "  PYTHON_BIN:     ${PYTHON_BIN}"
echo "  MMLU_RUNNER:    ${MMLU_RUNNER:-(not set)}"
echo "  GPQA_RUNNER:    ${GPQA_RUNNER:-(not set)}"
echo "  P25_PROBE:      ${P25_PROBE}"
echo ""

# ---------------------------------------------------------------------------
# Stage 0: Asset completion
# ---------------------------------------------------------------------------
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 0 — Asset completion check"
echo "────────────────────────────────────────────────────────────────────────"

BF16_INDEX="${BF16_MODEL}/model.safetensors.index.json"
NVFP4_MANIFEST="${NVFP4_MODEL}/lynn_quant_manifest.json"

BF16_READY=0; NVFP4_READY=0; Q4KM_READY=0

if [[ -f "$BF16_INDEX" ]]; then
    log_ok "BF16 index found"
    BF16_READY=1
else
    log_warn "BF16 index MISSING"
fi

if [[ -f "$NVFP4_MANIFEST" ]]; then
    log_ok "NVFP4 manifest found"
    NVFP4_READY=1
else
    log_warn "NVFP4 manifest MISSING"
fi

if [[ -f "$Q4KM_GGUF" ]]; then
    log_ok "Q4_K_M GGUF found"
    Q4KM_READY=1
else
    log_warn "Q4_K_M GGUF MISSING (PENDING, no download)"
fi

# ---------------------------------------------------------------------------
# Stage 1: Size / manifest summary
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 1 — Size / manifest summary"
echo "────────────────────────────────────────────────────────────────────────"

BF16_BYTES=0; NVFP4_BYTES=0
QUANTIZED_COUNT=null; KEPT_COUNT=null; OUTPUT_SHARDS=null; PACK_ELAPSED=null

if [[ "$BF16_READY" == "1" ]]; then
    BF16_BYTES="$(dir_bytes "$BF16_MODEL")"
    SIZE_GIB_BF16="$(awk "BEGIN {printf \"%.3f\", $BF16_BYTES/(1024^3)}")"
    log_ok "BF16 size: ${SIZE_GIB_BF16} GiB"
else
    SIZE_GIB_BF16="null"
fi

if [[ "$NVFP4_READY" == "1" ]]; then
    NVFP4_BYTES="$(dir_bytes "$NVFP4_MODEL")"
    SIZE_GIB_NVFP4="$(awk "BEGIN {printf \"%.3f\", $NVFP4_BYTES/(1024^3)}")"
    log_ok "NVFP4 size: ${SIZE_GIB_NVFP4} GiB"

    if [[ -f "$NVFP4_MANIFEST" ]]; then
        MANIFEST_DATA="$("$PY_EXEC" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        m = json.load(f)
    print(json.dumps({
        'quantized_count': m.get('quantized_count'),
        'kept_count': m.get('kept_count'),
        'output_shards': m.get('output_shards'),
        'elapsed_seconds': m.get('elapsed_seconds'),
    }))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" "$NVFP4_MANIFEST")"
        QUANTIZED_COUNT="$(echo "$MANIFEST_DATA" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('quantized_count','null'))")"
        KEPT_COUNT="$(echo "$MANIFEST_DATA" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('kept_count','null'))")"
        OUTPUT_SHARDS="$(echo "$MANIFEST_DATA" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('output_shards','null'))")"
        PACK_ELAPSED="$(echo "$MANIFEST_DATA" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('elapsed_seconds','null'))")"
        log_ok "Manifest: quantized=$QUANTIZED_COUNT kept=$KEPT_COUNT shards=$OUTPUT_SHARDS elapsed=${PACK_ELAPSED}s"
    fi
else
    SIZE_GIB_NVFP4="null"
fi

# ---------------------------------------------------------------------------
# Stage 2: GPU idle check
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 2 — GPU idle check"
echo "────────────────────────────────────────────────────────────────────────"

GPU_IDLE=0
if gpu_is_idle; then
    GPU_IDLE=1
else
    log_warn "GPU not idle — all GPU benchmarks will be SKIPPED/BLOCKED"
fi

# ---------------------------------------------------------------------------
# Stage 3: Benchmark per quant
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 3 — Benchmark per quant"
echo "────────────────────────────────────────────────────────────────────────"

reset_quant_state

benchmark_quant() {
    local quant="$1"
    local model_path="$2"
    local ready="$3"
    local port="$4"
    local served_name="Qwen3.5-9B-${quant}-${STAMP}"
    local server_log="$REPORT_DIR/server_${quant}_${STAMP}.log"
    local health_json="$REPORT_DIR/server_${quant}_${STAMP}_health.json"
    local prefix="$REPORT_DIR/${quant}_${STAMP}"
    local mmlu_jsonl="${prefix}_mmlu_n500.jsonl"
    local mmlu_summary="${prefix}_mmlu_n500.summary.json"
    local gpqa_jsonl="${prefix}_gpqa.jsonl"
    local gpqa_summary="${prefix}_gpqa.summary.json"
    local tps_json="${prefix}_tps.json"

    # Set per-quant path vars
    case "$quant" in
        bf16)
            MMLU_PATH_BF16="$mmlu_summary"; GPQA_PATH_BF16="$gpqa_summary"; TPS_PATH_BF16="$tps_json"
            ;;
        nvfp4)
            MMLU_PATH_NVFP4="$mmlu_summary"; GPQA_PATH_NVFP4="$gpqa_summary"; TPS_PATH_NVFP4="$tps_json"
            ;;
        q4_k_m)
            MMLU_PATH_Q4KM="$mmlu_summary"; GPQA_PATH_Q4KM="$gpqa_summary"; TPS_PATH_Q4KM="$tps_json"
            ;;
    esac

    if [[ "$ready" != "1" ]]; then
        case "$quant" in
            bf16)   STATUS_BF16="PENDING";  BLOCKED_BF16="Model asset not ready" ;;
            nvfp4)  STATUS_NVFP4="PENDING"; BLOCKED_NVFP4="Model asset not ready" ;;
            q4_k_m) STATUS_Q4KM="PENDING";  BLOCKED_Q4KM="Model asset not ready" ;;
        esac
        log_warn "[$quant] PENDING — model asset not ready"
        return 0
    fi

    if [[ "$GPU_IDLE" != "1" ]]; then
        case "$quant" in
            bf16)   STATUS_BF16="BLOCKED";  BLOCKED_BF16="GPU not idle" ;;
            nvfp4)  STATUS_NVFP4="BLOCKED"; BLOCKED_NVFP4="GPU not idle" ;;
            q4_k_m) STATUS_Q4KM="BLOCKED";  BLOCKED_Q4KM="GPU not idle" ;;
        esac
        log_block "[$quant] BLOCKED — GPU not idle"
        return 0
    fi

    if [[ "$quant" == "q4_k_m" ]]; then
        case "$quant" in
            q4_k_m) STATUS_Q4KM="PENDING"; BLOCKED_Q4KM="Q4_K_M eval not in this harness (llama.cpp runner not implemented)" ;;
        esac
        log_warn "[$quant] PENDING — llama.cpp eval not in this harness"
        return 0
    fi

    case "$quant" in
        bf16)   STATUS_BF16="RUNNING" ;;
        nvfp4)  STATUS_NVFP4="RUNNING" ;;
    esac
    log_info "[$quant] Starting benchmark..."

    local server_ok=0
    if start_server "$model_path" "$served_name" "$port" "$server_log" "$health_json"; then
        server_ok=1
    else
        case "$quant" in
            bf16)   STATUS_BF16="BLOCKED";  BLOCKED_BF16="Server failed to start" ;;
            nvfp4)  STATUS_NVFP4="BLOCKED"; BLOCKED_NVFP4="Server failed to start" ;;
        esac
        log_err "[$quant] Server failed"
        return 0
    fi

    local base_url="http://${SERVER_HOST}:${port}/v1"
    local load_start load_end
    load_start=$(date +%s.%N)

    # MMLU
    if run_mmlu "$base_url" "$served_name" "$mmlu_jsonl" "$mmlu_summary" 2>/dev/null; then
        if [[ -f "$mmlu_summary" ]]; then
            local mmlu_data
            mmlu_data="$("$PY_EXEC" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    print(json.dumps({'accuracy': d.get('accuracy'), 'correct': d.get('correct'), 'total': d.get('n')}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" "$mmlu_summary")"
            local ms mc mt
            ms="$(echo "$mmlu_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('accuracy','null'))")"
            mc="$(echo "$mmlu_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('correct','null'))")"
            mt="$(echo "$mmlu_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('total',500))" || echo "500")"
            case "$quant" in
                bf16)   MMLU_SCORE_BF16="$ms"; MMLU_CORRECT_BF16="$mc"; MMLU_TOTAL_BF16="$mt" ;;
                nvfp4)  MMLU_SCORE_NVFP4="$ms"; MMLU_CORRECT_NVFP4="$mc"; MMLU_TOTAL_NVFP4="$mt" ;;
            esac
            log_ok "[$quant] MMLU score=$ms"
        fi
    else
        log_warn "[$quant] MMLU blocked or failed"
    fi

    # GPQA
    if run_gpqa "$base_url" "$served_name" "$gpqa_jsonl" "$gpqa_summary" 2>/dev/null; then
        if [[ -f "$gpqa_summary" ]]; then
            local gpqa_data
            gpqa_data="$("$PY_EXEC" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    print(json.dumps({'accuracy': d.get('accuracy'), 'correct': d.get('correct'), 'total': d.get('n')}))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" "$gpqa_summary")"
            local gs gc gt
            gs="$(echo "$gpqa_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('accuracy','null'))")"
            gc="$(echo "$gpqa_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('correct','null'))")"
            gt="$(echo "$gpqa_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('total','null'))")"
            case "$quant" in
                bf16)   GPQA_SCORE_BF16="$gs"; GPQA_CORRECT_BF16="$gc"; GPQA_TOTAL_BF16="$gt" ;;
                nvfp4)  GPQA_SCORE_NVFP4="$gs"; GPQA_CORRECT_NVFP4="$gc"; GPQA_TOTAL_NVFP4="$gt" ;;
            esac
            log_ok "[$quant] GPQA score=$gs"
        fi
    else
        log_warn "[$quant] GPQA blocked or failed"
    fi

    # TPS
    if run_tps "$base_url" "$served_name" "$tps_json" 2>/dev/null; then
        if [[ -f "$tps_json" ]]; then
            local tps_data
            tps_data="$("$PY_EXEC" -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    out = {}
    if isinstance(d, list):
        from collections import defaultdict
        groups = defaultdict(list)
        for run in d:
            mt = run.get('max_tokens')
            tps = run.get('wall_tps')
            if mt and tps:
                groups[mt].append(tps)
        for mt, vals in groups.items():
            out[f'tps_{mt}'] = sum(vals)/len(vals)
    print(json.dumps(out))
except Exception as e:
    print(json.dumps({'error': str(e)}))
" "$tps_json")"
            local t128 t256 t512
            t128="$(echo "$tps_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('tps_128','null'))")"
            t256="$(echo "$tps_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('tps_256','null'))")"
            t512="$(echo "$tps_data" | "$PY_EXEC" -c "import json,sys; print(json.load(sys.stdin).get('tps_512','null'))")"
            case "$quant" in
                bf16)   TPS_128_BF16="$t128"; TPS_256_BF16="$t256"; TPS_512_BF16="$t512" ;;
                nvfp4)  TPS_128_NVFP4="$t128"; TPS_256_NVFP4="$t256"; TPS_512_NVFP4="$t512" ;;
            esac
            log_ok "[$quant] TPS 128=$t128 256=$t256 512=$t512"
        fi
    else
        log_warn "[$quant] TPS blocked or failed"
    fi

    load_end=$(date +%s.%N)
    local load_sec
    load_sec="$(awk "BEGIN {printf \"%.1f\", $load_end - $load_start}")"
    case "$quant" in
        bf16)   LOAD_SEC_BF16="$load_sec"; STATUS_BF16="DONE" ;;
        nvfp4)  LOAD_SEC_NVFP4="$load_sec"; STATUS_NVFP4="DONE" ;;
    esac
    stop_server
    log_ok "[$quant] Benchmark complete (load+eval ${load_sec}s)"
}

benchmark_quant "bf16"   "$BF16_MODEL"   "$BF16_READY"   "$BF16_PORT"
benchmark_quant "nvfp4"  "$NVFP4_MODEL"  "$NVFP4_READY"  "$NVFP4_PORT"
benchmark_quant "q4_k_m" "$Q4KM_GGUF"    "$Q4KM_READY"   "$Q4KM_PORT"

# ---------------------------------------------------------------------------
# Stage 4: Write unified summary JSON
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 4 — Write unified summary JSON"
echo "────────────────────────────────────────────────────────────────────────"

SUMMARY_JSON="${REPORT_DIR}/r6000_qwen35_9b_official_matrix_summary_${STAMP}.json"

"$PY_EXEC" - \
    "$BF16_READY" "$NVFP4_READY" "$Q4KM_READY" \
    "$BF16_BYTES" "$NVFP4_BYTES" \
    "$QUANTIZED_COUNT" "$KEPT_COUNT" "$OUTPUT_SHARDS" "$PACK_ELAPSED" \
    "$STATUS_BF16" "$BLOCKED_BF16" \
    "$MMLU_SCORE_BF16" "$MMLU_CORRECT_BF16" "$MMLU_TOTAL_BF16" "$MMLU_PATH_BF16" \
    "$GPQA_SCORE_BF16" "$GPQA_CORRECT_BF16" "$GPQA_TOTAL_BF16" "$GPQA_PATH_BF16" \
    "$TPS_128_BF16" "$TPS_256_BF16" "$TPS_512_BF16" "$TPS_PATH_BF16" \
    "$LOAD_SEC_BF16" "$SIZE_GIB_BF16" \
    "$STATUS_NVFP4" "$BLOCKED_NVFP4" \
    "$MMLU_SCORE_NVFP4" "$MMLU_CORRECT_NVFP4" "$MMLU_TOTAL_NVFP4" "$MMLU_PATH_NVFP4" \
    "$GPQA_SCORE_NVFP4" "$GPQA_CORRECT_NVFP4" "$GPQA_TOTAL_NVFP4" "$GPQA_PATH_NVFP4" \
    "$TPS_128_NVFP4" "$TPS_256_NVFP4" "$TPS_512_NVFP4" "$TPS_PATH_NVFP4" \
    "$LOAD_SEC_NVFP4" "$SIZE_GIB_NVFP4" \
    "$STATUS_Q4KM" "$BLOCKED_Q4KM" \
    "$MMLU_SCORE_Q4KM" "$MMLU_CORRECT_Q4KM" "$MMLU_TOTAL_Q4KM" "$MMLU_PATH_Q4KM" \
    "$GPQA_SCORE_Q4KM" "$GPQA_CORRECT_Q4KM" "$GPQA_TOTAL_Q4KM" "$GPQA_PATH_Q4KM" \
    "$TPS_128_Q4KM" "$TPS_256_Q4KM" "$TPS_512_Q4KM" "$TPS_PATH_Q4KM" \
    "$LOAD_SEC_Q4KM" "$SIZE_GIB_Q4KM" \
    "$SUMMARY_JSON" <<'PY'
import json, sys

def to_num(v):
    if v in (None, "null", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def to_int(v):
    if v in (None, "null", ""):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None

bf16_ready = sys.argv[1] == "1"
nvfp4_ready = sys.argv[2] == "1"
q4km_ready = sys.argv[3] == "1"

bf16_bytes = int(sys.argv[4] or 0)
nvfp4_bytes = int(sys.argv[5] or 0)

qc = sys.argv[6] if sys.argv[6] != "null" else None
kc = sys.argv[7] if sys.argv[7] != "null" else None
os_ = sys.argv[8] if sys.argv[8] != "null" else None
pe = sys.argv[9] if sys.argv[9] != "null" else None

def to_gib(b):
    return round(b / (1024 ** 3), 3) if b else None

def make_quant(status, blocked, mmlu_s, mmlu_c, mmlu_t, mmlu_p, gpqa_s, gpqa_c, gpqa_t, gpqa_p, tps128, tps256, tps512, tps_p, load_sec, size_gib):
    mmlu_status = "DONE" if to_num(mmlu_s) is not None else ("BLOCKED" if blocked else "PENDING")
    gpqa_status = "DONE" if to_num(gpqa_s) is not None else ("BLOCKED" if blocked else "PENDING")
    tps_status = "DONE" if to_num(tps128) is not None else ("BLOCKED" if blocked else "PENDING")
    return {
        "status": status,
        "blocked_reason": blocked or None,
        "size_gib": to_num(size_gib),
        "load_seconds": to_num(load_sec),
        "mmlu_500_5shot": {
            "score": to_num(mmlu_s),
            "correct": to_int(mmlu_c),
            "total": to_int(mmlu_t) or 500,
            "report_path": mmlu_p if status != "PENDING" else None,
            "status": mmlu_status,
            "blocked_reason": blocked or None,
        },
        "gpqa_diamond": {
            "score": to_num(gpqa_s),
            "correct": to_int(gpqa_c),
            "total": to_int(gpqa_t),
            "report_path": gpqa_p if status != "PENDING" else None,
            "status": gpqa_status,
            "blocked_reason": blocked or None,
        },
        "single_tps": {
            "tps_128": to_num(tps128),
            "tps_256": to_num(tps256),
            "tps_512": to_num(tps512),
            "report_path": tps_p if status != "PENDING" else None,
            "status": tps_status,
            "blocked_reason": blocked or None,
        },
    }

summary = {
    "schema": "lynn-qwen35-9b-official-matrix-summary-v1",
    "model_id": "Qwen3.5-9B",
    "arch": "dense",
    "stamp": sys.argv[-1].rsplit("_", 1)[-1].replace(".json", ""),
    "assets": {
        "bf16": {
            "ready": bf16_ready,
            "bytes": bf16_bytes,
            "gib": to_gib(bf16_bytes),
        },
        "nvfp4": {
            "ready": nvfp4_ready,
            "bytes": nvfp4_bytes,
            "gib": to_gib(nvfp4_bytes),
            "quantized_count": int(qc) if qc is not None else None,
            "kept_count": int(kc) if kc is not None else None,
            "output_shards": int(os_) if os_ is not None else None,
            "pack_elapsed_seconds": float(pe) if pe is not None else None,
        },
        "q4_k_m": {
            "ready": q4km_ready,
            "status": "PENDING",
            "note": "GGUF not downloaded if missing; no download in this watcher.",
        },
    },
    "results": {
        "bf16": make_quant(sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15],
                           sys.argv[16], sys.argv[17], sys.argv[18], sys.argv[19],
                           sys.argv[20], sys.argv[21], sys.argv[22], sys.argv[23], sys.argv[24], sys.argv[25]),
        "nvfp4": make_quant(sys.argv[26], sys.argv[27], sys.argv[28], sys.argv[29], sys.argv[30], sys.argv[31],
                            sys.argv[32], sys.argv[33], sys.argv[34], sys.argv[35],
                            sys.argv[36], sys.argv[37], sys.argv[38], sys.argv[39], sys.argv[40], sys.argv[41]),
        "q4_k_m": make_quant(sys.argv[42], sys.argv[43], sys.argv[44], sys.argv[45], sys.argv[46], sys.argv[47],
                             sys.argv[48], sys.argv[49], sys.argv[50], sys.argv[51],
                             sys.argv[52], sys.argv[53], sys.argv[54], sys.argv[55], sys.argv[56], sys.argv[57]),
    },
}

out_path = sys.argv[-1]
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[matrix] Wrote summary JSON: {out_path}")
PY

# ---------------------------------------------------------------------------
# Stage 5: Summarize to Markdown
# ---------------------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────────────────────────"
echo " Stage 5 — Summarize to Markdown"
echo "────────────────────────────────────────────────────────────────────────"

MD_OUT="${REPORT_DIR}/QWEN35_9B_R6000_NVFP4_PIPELINE_${STAMP}.md"
if [[ -f "$SUMMARIZER" ]]; then
    run_or_dry "\"$PY_EXEC\" \"$SUMMARIZER\" --summary \"$SUMMARY_JSON\" --out \"$MD_OUT\""
else
    log_warn "Summarizer not found: $SUMMARIZER"
fi

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo " Watch complete"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Summary JSON:  $SUMMARY_JSON"
echo "  Markdown:      $MD_OUT"
echo "  Log:           $LOG"
echo "  DRY_RUN:       $DRY_RUN"
echo ""
echo "  Per-quant results:"
echo "    BF16:   $STATUS_BF16   MMLU=$MMLU_SCORE_BF16 GPQA=$GPQA_SCORE_BF16"
echo "    NVFP4:  $STATUS_NVFP4  MMLU=$MMLU_SCORE_NVFP4 GPQA=$GPQA_SCORE_NVFP4"
echo "    Q4KM:   $STATUS_Q4KM   MMLU=$MMLU_SCORE_Q4KM GPQA=$GPQA_SCORE_Q4KM"
echo ""
echo "  Next steps:"
echo "    - If assets incomplete: wait for packer to finish."
echo "    - If GPU busy: retry when 35B P37 is idle."
echo "    - To execute for real: DRY_RUN=0 SKIP_GPU=0"
echo "    - Set MMLU_RUNNER=/path/to/mmlu_runner_v2.py GPQA_RUNNER=/path/to/gpqa_runner_v2.py"
echo ""
} 2>&1 | tee "$LOG" || true

echo "$LOG"
