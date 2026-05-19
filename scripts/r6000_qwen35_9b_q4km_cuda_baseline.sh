#!/usr/bin/env bash
# r6000_qwen35_9b_q4km_cuda_baseline.sh
#
# Qwen3.5-9B Q4_K_M-imatrix llama.cpp CUDA baseline on R6000 (RTX PRO 6000 Blackwell).
#
# This script MUST use the CUDA build of llama-server:
#   /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server
#
# The CPU-only build (/root/autodl-tmp/llama.cpp/build/bin/llama-server) is
# NOT suitable for this baseline. If the CUDA binary is missing, the script
# exits with an error rather than silently falling back to CPU.
#
# Usage:
#   bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh          # full suite
#   GPQA=1 bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh   # GPQA 32K thinking only
#   PERF=1 bash scripts/r6000_qwen35_9b_q4km_cuda_baseline.sh   # PP+TG throughput only
#
# Branch: qwen/qwen35-9b-q4km-cuda-baseline-20260519
# Stream: 9B-C / Qwen(MIMO)

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen35_9b}"
PORT="${PORT:-18197}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-8}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

# Test matrix
SINGLE_MAX_TOKENS="${SINGLE_MAX_TOKENS:-128 256 512}"
CONCURRENCY="${CONCURRENCY:-2 4 8}"
LONG_CONTEXT_CHARS="${LONG_CONTEXT_CHARS:-4096 16384 32768}"
LONG_DECODE_TOKENS="${LONG_DECODE_TOKENS:-64}"
LONG_CONCURRENCY="${LONG_CONCURRENCY:-4}"

# GPQA
GPQA_TIMEOUT="${GPQA_TIMEOUT:-1800}"
GPQA_SEED="${GPQA_SEED:-42}"

# Toggle: set GPQA=1 to run only GPQA, PERF=1 to run only perf
RUN_GPQA="${GPQA:-0}"
RUN_PERF="${PERF:-0}"
if [[ "$RUN_GPQA" == "0" && "$RUN_PERF" == "0" ]]; then
  RUN_GPQA=1; RUN_PERF=1
fi

# Temp dir for intermediate results
WORK_DIR=""
cleanup() {
  [[ -n "$WORK_DIR" && -d "$WORK_DIR" ]] && rm -rf "$WORK_DIR"
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Helpers ──────────────────────────────────────────────────────────────────
ts()  { date +%Y%m%d_%H%M; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

now_ns() { date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))"; }
elapsed_s() {
  local start="$1" end="$2"
  python3 -c "print(round(($end - $start) / 1e9, 3))"
}

# ── Locate CUDA binary ──────────────────────────────────────────────────────
discover_server() {
  if [[ -n "${LLAMA_SERVER:-}" ]]; then
    [[ -x "$LLAMA_SERVER" ]] || die "LLAMA_SERVER=$LLAMA_SERVER not executable"
    echo "$LLAMA_SERVER"; return
  fi
  local candidates=(
    /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server
  )
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then echo "$c"; return; fi
  done
  die "CUDA llama-server not found.
Expected: /root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server
Build with:
  cd /root/autodl-tmp/llama.cpp
  cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
  cmake --build build-cuda -j
Do NOT use the CPU-only build (build/bin/llama-server)."
}

# ── Locate GGUF ─────────────────────────────────────────────────────────────
discover_gguf() {
  if [[ -n "${GGUF:-}" && -f "$GGUF" ]]; then echo "$GGUF"; return; fi
  local candidates=(
    "$MODEL_ROOT/Qwen3.5-9B-Q4_K_M-imatrix.gguf"
    "$MODEL_ROOT/Qwen3.5-9B-Q4_K_M.gguf"
  )
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then echo "$c"; return; fi
  done
  # Glob fallback
  local match
  match=$(find "$MODEL_ROOT" -maxdepth 2 -name '*Qwen3.5*9B*Q4*K*M*.gguf' -type f 2>/dev/null | head -1)
  if [[ -n "$match" ]]; then echo "$match"; return; fi
  echo ""
}

# ── GGUF download ───────────────────────────────────────────────────────────
download_gguf() {
  local target="$MODEL_ROOT/Qwen3.5-9B-Q4_K_M-imatrix.gguf"
  log "GGUF not found — attempting download → $target"
  mkdir -p "$MODEL_ROOT"
  if [[ -n "${HF_REPO:-}" ]]; then
    huggingface-cli download "$HF_REPO" --include "*q4_k_m*imatrix*" --local-dir "$MODEL_ROOT"
  elif [[ -n "${MS_REPO:-}" ]]; then
    modelscope download "$MS_REPO" --include "*q4_k_m*imatrix*" --local_dir "$MODEL_ROOT"
  elif [[ -n "${GGUF_URL:-}" ]]; then
    curl -fSL "$GGUF_URL" -o "$target"
  else
    log "No download source configured. Set HF_REPO, MS_REPO, or GGUF_URL."
    return 1
  fi
  [[ -f "$target" ]] && echo "$target" && return
  echo ""
}

# ── Verify CUDA binary ──────────────────────────────────────────────────────
verify_cuda_binary() {
  local server="$1"
  if strings "$server" 2>/dev/null | grep -q "ggml-cuda"; then
    log "CUDA binary verified: $server (ggml-cuda symbols found)"
  elif strings "$server" 2>/dev/null | grep -q "CUDA"; then
    log "CUDA binary likely OK: $server (CUDA string found)"
  else
    log "WARNING: Cannot confirm CUDA support in $server — proceeding anyway"
  fi
}

# ── Start server ─────────────────────────────────────────────────────────────
SERVER_PID=""
SERVER_LOG=""

start_server() {
  local server="$1" model="$2"
  SERVER_LOG="/tmp/llama_server_cuda_$$.log"
  log "Starting server: $server"
  log "  model=$model ctx=$CTX_SIZE parallel=$PARALLEL gpu_layers=$N_GPU_LAYERS"
  log "  log=$SERVER_LOG"

  "$server" \
    --model "$model" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --flash-attn auto \
    --cont-batching \
    --log-disable \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  log "Waiting for server to become ready (PID=$SERVER_PID)..."
  local attempts=0
  while ! curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      die "Server process died. Check $SERVER_LOG"
    fi
    if (( attempts >= 240 )); then
      die "Server did not become ready within 120s. Check $SERVER_LOG"
    fi
    sleep 0.5
    ((++attempts))
  done
  log "Server ready (took $((attempts / 2))s)"

  if grep -qi "CUDA" "$SERVER_LOG" 2>/dev/null; then
    log "Server log confirms CUDA backend"
  else
    log "WARNING: No CUDA mention in server log — check $SERVER_LOG"
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping server PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

# ── Probe: single-stream decode ──────────────────────────────────────────────
probe_single() {
  local max_tokens="$1" out_file="$2"
  local prompt="Explain the concept of gradient descent in machine learning in exactly $max_tokens words."
  local resp_file="${WORK_DIR}/single_resp_${max_tokens}.json"

  local start_ns end_ns
  start_ns=$(now_ns)

  curl -sf --max-time 300 -X POST "http://$HOST:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"qwen3.5-9b\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
      \"max_tokens\": $max_tokens,
      \"temperature\": 0.0,
      \"stream\": false
    }" > "$resp_file" 2>/dev/null || {
    echo '{"ok":false,"error":"curl_failed","max_tokens_req":'"$max_tokens"'}' > "$out_file"
    return
  }

  end_ns=$(now_ns)
  local es
  es=$(elapsed_s "$start_ns" "$end_ns")

  python3 -c "
import json
with open('$resp_file') as f:
    resp = json.load(f)
pt = resp['usage']['prompt_tokens']
ct = resp['usage']['completion_tokens']
es = $es
result = {
    'ok': True,
    'max_tokens_req': $max_tokens,
    'prompt_tokens': pt,
    'completion_tokens': ct,
    'elapsed_s': es,
    'wall_tps': round(ct / es, 2) if es > 0 else 0
}
with open('$out_file', 'w') as f:
    json.dump(result, f, indent=2)
print(f'  {result[\"wall_tps\"]} TPS ({ct} tokens in {es}s)')
" 2>/dev/null || echo '{"ok":false,"error":"parse_failed","max_tokens_req":'"$max_tokens"'}' > "$out_file"
}

# ── Probe: concurrent decode ─────────────────────────────────────────────────
probe_concurrent() {
  local concurrency="$1" out_file="$2"
  local max_tokens=256
  local prompt="Write a detailed analysis of renewable energy trends in 2026 covering solar wind and hydrogen."

  local start_ns end_ns
  start_ns=$(now_ns)

  local pids=()
  for (( i=0; i<concurrency; i++ )); do
    curl -sf --max-time 300 -X POST "http://$HOST:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"qwen3.5-9b\",
        \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
        \"max_tokens\": $max_tokens,
        \"temperature\": 0.7,
        \"stream\": false
      }" > "${WORK_DIR}/cc_${i}.json" 2>/dev/null &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done

  end_ns=$(now_ns)
  local es
  es=$(elapsed_s "$start_ns" "$end_ns")

  python3 -c "
import json, os
total_ct = 0
for i in range($concurrency):
    try:
        with open('${WORK_DIR}/cc_' + str(i) + '.json') as f:
            total_ct += json.load(f)['usage']['completion_tokens']
    except: pass
es = $es
result = {
    'ok': total_ct > 0,
    'concurrency': $concurrency,
    'max_tokens_req': $max_tokens,
    'total_completion_tokens': total_ct,
    'elapsed_s': es,
    'batch_wall_tps': round(total_ct / es, 2) if es > 0 else 0
}
with open('$out_file', 'w') as f:
    json.dump(result, f, indent=2)
print(f'  {result[\"batch_wall_tps\"]} TPS ({total_ct} tokens in {es}s)')
" 2>/dev/null || echo '{"ok":false,"error":"parse_failed","concurrency":'"$concurrency"'}' > "$out_file"
}

# ── Probe: long-context prefill+decode ───────────────────────────────────────
probe_long_context() {
  local chars="$1" out_file="$2"
  local decode_tokens="${LONG_DECODE_TOKENS}"
  local concurrency="${LONG_CONCURRENCY}"

  # Write filler to temp file to avoid shell heredoc size limits
  local filler_file="${WORK_DIR}/filler_${chars}.txt"
  python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * (($chars // 46) + 1))[:$chars])" > "$filler_file"

  local start_ns end_ns
  start_ns=$(now_ns)

  local pids=()
  for (( i=0; i<concurrency; i++ )); do
    python3 -c "
import json
with open('$filler_file') as f:
    filler = f.read()
payload = {
    'model': 'qwen3.5-9b',
    'messages': [
        {'role': 'system', 'content': filler},
        {'role': 'user', 'content': 'Summarize the above in $decode_tokens tokens.'}
    ],
    'max_tokens': $decode_tokens,
    'temperature': 0.0,
    'stream': False
}
print(json.dumps(payload))
" | curl -sf --max-time 600 -X POST "http://$HOST:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d @- > "${WORK_DIR}/lc_${i}.json" 2>/dev/null &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done

  end_ns=$(now_ns)
  local es
  es=$(elapsed_s "$start_ns" "$end_ns")

  python3 -c "
import json
total_pt = 0; total_ct = 0
for i in range($concurrency):
    try:
        with open('${WORK_DIR}/lc_' + str(i) + '.json') as f:
            d = json.load(f)
            total_pt += d['usage']['prompt_tokens']
            total_ct += d['usage']['completion_tokens']
    except: pass
es = $es
result = {
    'ok': total_ct > 0,
    'chars': $chars,
    'concurrency': $concurrency,
    'total_prompt_tokens': total_pt,
    'total_completion_tokens': total_ct,
    'elapsed_s': es,
    'wall_tps': round(total_ct / es, 2) if es > 0 else 0
}
with open('$out_file', 'w') as f:
    json.dump(result, f, indent=2)
status = 'OK' if result['ok'] else 'FAILED'
print(f'  {result[\"wall_tps\"]} TPS, {status}')
" 2>/dev/null || echo '{"ok":false,"error":"parse_failed","chars":'"$chars"'}' > "$out_file"
}

# ── GPQA 32K Thinking ───────────────────────────────────────────────────────
run_gpqa_thinking() {
  local server="$1" model="$2"
  log "=== GPQA Diamond 32K Thinking (CUDA) ==="

  local gpqa_out="$REPORT_ROOT/gpqa_q4km_cuda_$(ts)_gpqa.json"

  # Generate 32K thinking filler
  log "Generating 32K thinking context..."
  local thinking_filler
  thinking_filler=$(python3 -c "
import random, string
random.seed($GPQA_SEED)
words = []
topics = ['quantum computing', 'neural networks', 'protein folding',
          'transformer architectures', 'reinforcement learning',
          'attention mechanisms', 'gradient optimization', 'loss landscapes',
          'batch normalization', 'dropout regularization']
for _ in range(4000):
    words.append(random.choice(topics) + ' ' + ''.join(random.choices(string.ascii_lowercase, k=3)))
print(' '.join(words)[:32768])
")

  log "Starting GPQA eval with 32K thinking context..."
  local gpqa_start gpqa_end
  gpqa_start=$(date +%s)

  python3 scripts/openai_gpqa_diamond_eval.py \
    --base-url "http://$HOST:$PORT/v1" \
    --model "qwen3.5-9b" \
    --temperature 0.0 \
    --max-tokens 2048 \
    --gpqa-n 198 \
    --seed "$GPQA_SEED" \
    --system-prompt "You are a helpful assistant. Think step by step before answering. Context: $thinking_filler" \
    > "$gpqa_out" 2>&1 || {
    log "GPQA eval script failed — saving error"
    echo '{"status":"FAILED","reason":"eval_script_error"}' > "$gpqa_out"
  }

  gpqa_end=$(date +%s)
  log "GPQA eval completed in $(( gpqa_end - gpqa_start ))s"
  log "GPQA results: $gpqa_out"
}

# ── Assemble final report ────────────────────────────────────────────────────
assemble_report() {
  local report_file="$1" model="$2" server="$3" size_gib="$4" git_rev="$5" report_ts="$6"
  shift 6
  # Remaining args: list of (section_name, json_file) pairs
  python3 - "$report_file" "$model" "$server" "$size_gib" "$git_rev" "$report_ts" "$@" <<'PYEOF'
import json, sys

report_file = sys.argv[1]
model_path = sys.argv[2]
binary = sys.argv[3]
size_gib = sys.argv[4]
git_rev = sys.argv[5]
report_ts = sys.argv[6]

report = {
    "schema": "lynn-qwen35-9b-q4km-cuda-baseline-v1",
    "status": "DONE",
    "model_id": "Qwen3.5-9B",
    "quant": "Q4_K_M-imatrix",
    "engine": "llama.cpp CUDA",
    "engine_detail": "CUDA 12.8, CMAKE_CUDA_ARCHITECTURES=120, flash-attn",
    "model_path": model_path,
    "size_gib": size_gib,
    "llama_cpp_binary": binary,
    "git_rev": git_rev,
    "n_gpu_layers": 99,
    "ctx_size": 32768,
    "parallel": 8,
    "timestamp": report_ts,
    "single_tps": {},
    "concurrent_tps": {},
    "long_context": {},
    "errors": []
}

# Parse section_name=filepath pairs
args = sys.argv[7:]
for i in range(0, len(args), 2):
    section = args[i]
    filepath = args[i + 1]
    try:
        with open(filepath) as f:
            data = json.load(f)
        key = data.get("_key", section)
        if section in ("single_tps", "concurrent_tps", "long_context"):
            report[section][key] = data
        else:
            report[section] = data
    except Exception as e:
        report["errors"].append(f"{section}/{filepath}: {e}")

with open(report_file, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report written: {report_file}")
PYEOF
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
  local report_ts
  report_ts=$(ts)

  WORK_DIR=$(mktemp -d)
  log "Work dir: $WORK_DIR"

  # Discover binary
  local server
  server=$(discover_server)
  log "CUDA server: $server"
  verify_cuda_binary "$server"

  # Discover model
  local model
  model=$(discover_gguf)
  if [[ -z "$model" ]]; then
    log "GGUF not found — attempting download..."
    model=$(download_gguf) || true
  fi
  if [[ -z "$model" || ! -f "$model" ]]; then
    local pending="$REPORT_ROOT/r6000_qwen35_9b_q4km_cuda_baseline_${report_ts}_PENDING_DOWNLOAD.json"
    mkdir -p "$REPORT_ROOT"
    python3 -c "
import json
report = {
    'schema': 'lynn-qwen35-9b-q4km-cuda-baseline-v1',
    'status': 'PENDING_DOWNLOAD',
    'model_id': 'Qwen3.5-9B',
    'quant': 'Q4_K_M-imatrix',
    'engine': 'llama.cpp CUDA',
    'gguf_search_root': '$MODEL_ROOT',
    'timestamp': '$report_ts',
    'download_commands': {
        'huggingface': 'huggingface-cli download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m-imatrix.gguf --local-dir $MODEL_ROOT',
        'modelscope':  'modelscope download Qwen/Qwen3.5-9B-GGUF qwen3.5-9b-q4_k_m-imatrix.gguf --local_dir $MODEL_ROOT',
        'imatrix':     'curl -fSL https://huggingface.co/Qwen/Qwen3.5-9B-GGUF/resolve/main/qwen3.5-9b-q4_k_m-imatrix.gguf -o $MODEL_ROOT/Qwen3.5-9B-Q4_K_M-imatrix.gguf'
    },
    'errors': ['GGUF not found']
}
print(json.dumps(report, indent=2))
" > "$pending"
    die "GGUF not found. Download instructions: $pending"
  fi
  log "Model: $model"

  # GGUF size
  local size_gib
  size_gib=$(python3 -c "import os; print(round(os.path.getsize('$model') / (1024**3), 2))")
  log "Model size: ${size_gib} GiB"

  # CUDA binary git revision
  local git_rev
  git_rev=$(cd /root/autodl-tmp/llama.cpp 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

  mkdir -p "$REPORT_ROOT"

  # Start server (needed for both perf and GPQA)
  start_server "$server" "$model"

  # Collect report assembly args
  local report_args=()

  # ── Performance suite ────────────────────────────────────────────────────
  if [[ "$RUN_PERF" == "1" ]]; then
    log "=== Performance Suite ==="

    # Single-stream decode
    for mt in $SINGLE_MAX_TOKENS; do
      log "Single-stream decode: max_tokens=$mt"
      local sfile="${WORK_DIR}/single_${mt}.json"
      probe_single "$mt" "$sfile"
      # Add _key for report assembly
      python3 -c "
import json
with open('$sfile') as f: d = json.load(f)
d['_key'] = '$mt'
with open('$sfile', 'w') as f: json.dump(d, f, indent=2)
" 2>/dev/null || true
      report_args+=("single_tps" "$sfile")
    done

    # Concurrent decode
    for cc in $CONCURRENCY; do
      log "Concurrent decode: concurrency=$cc"
      local cfile="${WORK_DIR}/concurrent_${cc}.json"
      probe_concurrent "$cc" "$cfile"
      python3 -c "
import json
with open('$cfile') as f: d = json.load(f)
d['_key'] = '$cc'
with open('$cfile', 'w') as f: json.dump(d, f, indent=2)
" 2>/dev/null || true
      report_args+=("concurrent_tps" "$cfile")
    done

    # Long-context
    for lc in $LONG_CONTEXT_CHARS; do
      log "Long-context: chars=$lc"
      local lfile="${WORK_DIR}/long_${lc}.json"
      probe_long_context "$lc" "$lfile"
      python3 -c "
import json
with open('$lfile') as f: d = json.load(f)
d['_key'] = '$lc'
with open('$lfile', 'w') as f: json.dump(d, f, indent=2)
" 2>/dev/null || true
      report_args+=("long_context" "$lfile")
    done
  fi

  # ── GPQA suite ───────────────────────────────────────────────────────────
  if [[ "$RUN_GPQA" == "1" ]]; then
    run_gpqa_thinking "$server" "$model"
    local gpqa_file
    gpqa_file=$(ls -t "$REPORT_ROOT"/gpqa_q4km_cuda_*_gpqa.json 2>/dev/null | head -1)
    if [[ -n "$gpqa_file" ]]; then
      report_args+=("gpqa" "$gpqa_file")
    fi
  fi

  # ── Assemble report ─────────────────────────────────────────────────────
  local report_file="$REPORT_ROOT/r6000_qwen35_9b_q4km_cuda_baseline_${report_ts}.json"
  assemble_report "$report_file" "$model" "$server" "$size_gib" "$git_rev" "$report_ts" "${report_args[@]}"

  # Done
  stop_server
  log "=== CUDA Baseline Complete ==="
  log "Report: $report_file"
  log "Summary: python3 scripts/summarize_qwen35_9b_q4km_cuda_baseline.py $report_file"
}

main "$@"
