#!/usr/bin/env bash
set -euo pipefail

# R6000 llama.cpp baseline for official Qwen3.6-35B-A3B Q4_K_M-imatrix GGUF.
# Measures:
#   1. warm single-stream wall TPS
#   2. true OpenAI HTTP concurrency throughput
#   3. long-context prefill+decode wall behavior

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
GGUF=${GGUF:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-GGUF-imatrix/Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports/qwen36_35b}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18196}
SERVED_NAME=${SERVED_NAME:-Qwen3.6-35B-A3B-Q4KM-imatrix-r6000}
CTX_SIZE=${CTX_SIZE:-32768}
THREADS=${THREADS:-12}
N_GPU_LAYERS=${N_GPU_LAYERS:-99}
PARALLEL=${PARALLEL:-8}
CONCURRENCY=${CONCURRENCY:-"2 4"}
SINGLE_MAX_TOKENS=${SINGLE_MAX_TOKENS:-"128 256 512"}
LONG_CONTEXT_CHARS=${LONG_CONTEXT_CHARS:-"8192 32768 65536"}
CONCURRENT_MAX_TOKENS=${CONCURRENT_MAX_TOKENS:-256}
LONG_CONTEXT_MAX_TOKENS=${LONG_CONTEXT_MAX_TOKENS:-128}
LLAMA_SERVER=${LLAMA_SERVER:-}
LLAMA_EXTRA_ARGS=${LLAMA_EXTRA_ARGS:-}

mkdir -p "$REPORT_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

OUT="$REPORT_ROOT/r6000_qwen36_q4km_llamacpp_matrix_${STAMP}.json"
SERVER_LOG="$REPORT_ROOT/r6000_qwen36_q4km_llamacpp_server_${STAMP}.log"

if [[ ! -s "$GGUF" ]]; then
  echo "[q4km-bench] missing GGUF: $GGUF" >&2
  exit 2
fi

if [[ -z "$LLAMA_SERVER" ]]; then
  for candidate in \
    /root/autodl-tmp/llama.cpp/build/bin/llama-server \
    /root/autodl-tmp/llama.cpp/build/tools/server/llama-server \
    /root/autodl-tmp/llama.cpp/build/tools/server; do
    if [[ -x "$candidate" ]]; then
      LLAMA_SERVER="$candidate"
      break
    fi
  done
fi
if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  echo "[q4km-bench] missing llama-server binary; build /root/autodl-tmp/llama.cpp target llama-server first" >&2
  exit 3
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[q4km-bench] server=$LLAMA_SERVER"
echo "[q4km-bench] gguf=$GGUF"
echo "[q4km-bench] out=$OUT"

"$LLAMA_SERVER" \
  --model "$GGUF" \
  --host "$HOST" \
  --port "$PORT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --parallel "$PARALLEL" \
  --jinja \
  -a "$SERVED_NAME" \
  $LLAMA_EXTRA_ARGS > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 180); do
  sleep 5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[q4km-bench] server exited before ready" >&2
    tail -120 "$SERVER_LOG" || true
    exit 4
  fi
  smoke="$(curl -s -m 30 -H 'Content-Type: application/json' \
    -d '{"model":"'"$SERVED_NAME"'","prompt":"A","max_tokens":4,"temperature":0}' \
    "http://${HOST}:${PORT}/v1/completions" 2>&1 || true)"
  if echo "$smoke" | grep -q '"choices"'; then
    ready=1
    break
  fi
done
if [[ "$ready" != "1" ]]; then
  echo "[q4km-bench] server not ready" >&2
  tail -120 "$SERVER_LOG" || true
  exit 5
fi

"$PY" benchmarks/openai_serving_matrix_probe.py \
  --url "http://${HOST}:${PORT}/v1" \
  --model "$SERVED_NAME" \
  --single-max-tokens $SINGLE_MAX_TOKENS \
  --concurrency $CONCURRENCY \
  --concurrent-max-tokens "$CONCURRENT_MAX_TOKENS" \
  --long-context-chars $LONG_CONTEXT_CHARS \
  --long-context-max-tokens "$LONG_CONTEXT_MAX_TOKENS" \
  --runs 1 \
  --timeout 1800 \
  --out "$OUT"

"$PY" - "$OUT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))

def single_line():
    rows = report["single"]["rows"]
    return {
        str(r["max_tokens"]): {
            "ok": r["ok"],
            "wall_tps": r.get("wall_tps"),
            "prompt_tokens": r.get("prompt_tokens"),
        }
        for r in rows
    }

def concurrency_line():
    out = {}
    for r in report["concurrency"]["rows"]:
        c = str(r["concurrency"])
        out.setdefault(c, r.get("batch_wall_tps"))
    return out

def long_line():
    return {
        str(r["target_prompt_chars"]): {
            "ok": r["ok"],
            "prompt_tokens": r.get("prompt_tokens"),
            "wall_tps": r.get("wall_tps"),
            "error": r.get("error"),
        }
        for r in report["long_context"]["rows"]
    }

print(json.dumps({
    "report": str(path),
    "single": single_line(),
    "concurrency_batch_wall_tps": concurrency_line(),
    "long_context": long_line(),
}, ensure_ascii=False, indent=2))
PY
