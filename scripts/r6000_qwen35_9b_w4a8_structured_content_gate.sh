#!/usr/bin/env bash
# r6000_qwen35_9b_w4a8_structured_content_gate.sh
#
# R6000 runner for P196: Qwen3.5-9B W4A8 structured-content gate.
#
# Starts llama-server with each quantization variant (W4A16, W4A8 full,
# W4A8 gateup), runs structured content tests via /v1/chat/completions,
# collects results, and feeds them to the P196 gate for verdict.
#
# Usage:
#   bash scripts/r6000_qwen35_9b_w4a8_structured_content_gate.sh
#   MODEL=/path/to/model GGUF=/path/to.gguf bash scripts/r6000_qwen35_9b_w4a8_structured_content_gate.sh
#
# Branch: mimo/qwen35-9b-nvfp4-release-gates-20260519

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
GGUF="${GGUF:-/root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

SERVER_BIN="${SERVER_BIN:-${ROOT}/build-cuda/bin/llama-server}"
PORT="${PORT:-8096}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-30}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-60}"

P196_RESULTS="${REPORT_DIR}/p196_test_results_${STAMP}.json"
P196_REPORT="${REPORT_DIR}/p196_w4a8_content_gate_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ── Verify server binary ───────────────────────────────────────────────────
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: CUDA server binary not found at $SERVER_BIN"
    echo "Build with: cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build-cuda -j"
    exit 1
fi

echo "=== P196 Qwen3.5-9B W4A8 Structured Content Gate ==="
echo "  Model:  $MODEL"
echo "  GGUF:   $GGUF"
echo "  Server: $SERVER_BIN"
echo "  Port:   $PORT"
echo ""

# ── Helper: wait for server ready ──────────────────────────────────────────
wait_server() {
    local max="$1"
    local i=0
    while ! curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; do
        sleep 1
        i=$((i + 1))
        if [ "$i" -ge "$max" ]; then
            echo "ERROR: server did not start within ${max}s"
            return 1
        fi
    done
    echo "  Server ready (${i}s)"
}

# ── Helper: stop server ────────────────────────────────────────────────────
stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
}

trap stop_server EXIT

# ── Helper: run one test case via curl ─────────────────────────────────────
# Args: prompt_id, variant, prompt
# Appends JSON line to $P196_RESULTS
run_test() {
    local prompt_id="$1"
    local variant="$2"
    local prompt="$3"

    local start_ms
    start_ms=$(python3 -c "import time; print(int(time.time()*1000))")

    local response
    response=$(curl -s --max-time "$REQUEST_TIMEOUT" \
        "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json, sys
print(json.dumps({
    'model': 'qwen35-9b',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': 512,
    'temperature': 0.0,
    'stream': False
}))
" "$prompt")" 2>/dev/null) || response="{}"

    local end_ms
    end_ms=$(python3 -c "import time; print(int(time.time()*1000))")

    local output_text
    output_text=$(echo "$response" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except:
    print('')
" 2>/dev/null) || output_text=""

    local latency_ms
    latency_ms=$((end_ms - start_ms))

    # Validate output
    local validator
    validator=$(python3 -c "
import json, sys
cases = json.load(open('$ROOT/benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py'))
# We'll use the gate's validators via import instead
" 2>/dev/null) || true

    # Use the gate's validator
    local structural_pass
    structural_pass=$(python3 -c "
import sys, json
sys.path.insert(0, '$ROOT')
from benchmarks.p196_qwen35_9b_w4a8_structured_content_gate import VALIDATORS, BUILTIN_TEST_CASES
case = next((c for c in BUILTIN_TEST_CASES if c['id'] == sys.argv[1]), None)
if case and case['validator'] in VALIDATORS:
    result = VALIDATORS[case['validator']](sys.argv[2])
    print('true' if result else 'false')
else:
    print('false')
" "$prompt_id" "$output_text" 2>/dev/null) || structural_pass="false"

    # Compute TPS (approximate from output length / latency)
    local tps
    tps=$(python3 -c "
text = '''$output_text'''
tokens = len(text.split())  # rough estimate
lat = $latency_ms
print(f'{tokens / (lat / 1000.0):.2f}' if lat > 0 else '0.0')
" 2>/dev/null) || tps="0.0"

    # Append to results
    python3 -c "
import json
result = {
    'prompt_id': '$prompt_id',
    'variant': '$variant',
    'output_text': $(python3 -c "import json; print(json.dumps('$output_text'))" 2>/dev/null || echo '""'),
    'structural_pass': $( [ "$structural_pass" = "true" ] && echo "True" || echo "False"),
    'decode_tps': $tps,
    'latency_ms': $latency_ms
}
print(json.dumps(result))
" >> "$P196_RESULTS"
}

# ── Collect test prompts ───────────────────────────────────────────────────
TEST_PROMPTS=$(python3 -c "
import json, sys
sys.path.insert(0, '$ROOT')
from benchmarks.p196_qwen35_9b_w4a8_structured_content_gate import BUILTIN_TEST_CASES
for c in BUILTIN_TEST_CASES:
    print(json.dumps({'id': c['id'], 'prompt': c['prompt']}))
")

# ── Run tests for a given variant ──────────────────────────────────────────
run_variant() {
    local variant="$1"
    local env_extra="$2"  # e.g., "LYNN_W4A8_FAKE_QUANT_ACTIVE=1 LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3"

    echo ""
    echo "--- Variant: $variant ---"

    # Start server
    stop_server
    echo "  Starting server..."
    eval "$env_extra" "$SERVER_BIN" \
        --model "$MODEL" \
        --port "$PORT" \
        --n-gpu-layers "$N_GPU_LAYERS" \
        --ctx-size 4096 \
        --threads 8 \
        --log-disable &
    SERVER_PID=$!
    wait_server "$SERVER_TIMEOUT"

    # Run each test
    echo "$TEST_PROMPTS" | while IFS= read -r line; do
        local pid
        pid=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
        local prompt
        prompt=$(echo "$line" | python3 -c "import json,sys; print(json.load(sys.stdin)['prompt'])")
        echo "  [$variant] $pid"
        run_test "$pid" "$variant" "$prompt"
    done

    stop_server
    echo "  Variant $variant done"
}

# ── Main ───────────────────────────────────────────────────────────────────
# Clear results file
: > "$P196_RESULTS"

# 1. W4A16 reference (no fake-quant)
run_variant "w4a16" ""

# 2. W4A8 full (all layers fake-quantized)
run_variant "w4a8_full" "LYNN_W4A8_FAKE_QUANT_ACTIVE=1 LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3 LYNN_W4A8_FAKE_QUANT_GRANULARITY=per16"

# 3. W4A8 gateup (gate_up projection only)
run_variant "w4a8_gateup" "LYNN_W4A8_FAKE_QUANT_ACTIVE=1 LYNN_W4A8_FAKE_QUANT_FORMAT=e4m3 LYNN_W4A8_FAKE_QUANT_GRANULARITY=per16"

# ── Run P196 gate ──────────────────────────────────────────────────────────
echo ""
echo "--- Running P196 gate ---"
"$PYTHON_BIN" benchmarks/p196_qwen35_9b_w4a8_structured_content_gate.py \
    --results "$P196_RESULTS" \
    --out "$P196_REPORT" \
    "$@"

echo ""
echo "Results: $P196_RESULTS"
echo "Report:  $P196_REPORT"
echo ""
echo "=== P196 done ==="
