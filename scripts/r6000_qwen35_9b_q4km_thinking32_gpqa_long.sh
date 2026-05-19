#!/usr/bin/env bash
# Run a long Qwen3.5-9B Q4_K_M llama.cpp CUDA GPQA Diamond thinking-on eval on R6000.
#
# This is intentionally a long detached-friendly job. It starts a dedicated
# llama-server with Qwen thinking enabled, runs the shared OpenAI-compatible
# 32K thinking MCQ evaluator, writes JSONL + summary artifacts, then stops the
# server.

set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine-main}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-Q4_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen35_9b}"
GPQA_CSV="${GPQA_CSV:-$REPORT_ROOT/gpqa_diamond.csv}"
PORT="${PORT:-18197}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
CONCURRENCY="${CONCURRENCY:-1}"
LIMIT="${LIMIT:-198}"
TIMEOUT="${TIMEOUT:-3600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

OUT_DIR="$REPORT_ROOT/thinking32"
SERVER_LOG="$OUT_DIR/qwen35_9b_q4km_llama_server_${STAMP}.log"
EVAL_OUT="$OUT_DIR/qwen35_9b_q4km_gpqa_thinking32_${STAMP}.jsonl"
STATUS_JSON="$OUT_DIR/qwen35_9b_q4km_gpqa_thinking32_${STAMP}.status.json"
PID_FILE="$OUT_DIR/qwen35_9b_q4km_gpqa_thinking32_${STAMP}.pid"

SERVER_PID=""

log() {
  echo "[$(date '+%F %T')] $*"
}

write_status() {
  local status="$1"
  local note="${2:-}"
  python3 - "$STATUS_JSON" "$status" "$note" "$STAMP" "$EVAL_OUT" "$SERVER_LOG" "$PID_FILE" <<'PY'
import json
import sys
from pathlib import Path

path, status, note, stamp, eval_out, server_log, pid_file = sys.argv[1:]
payload = {
    "schema": "lynn-qwen35-9b-q4km-thinking32-gpqa-long-v1",
    "status": status,
    "note": note,
    "stamp": stamp,
    "eval_out": eval_out,
    "summary_json": str(Path(eval_out).with_suffix(".summary.json")),
    "server_log": server_log,
    "pid_file": pid_file,
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping llama-server PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

main() {
  mkdir -p "$OUT_DIR"
  cd "$ROOT"

  [[ -x "$LLAMA_SERVER" ]] || { write_status FAILED "missing llama-server: $LLAMA_SERVER"; exit 1; }
  [[ -f "$MODEL" ]] || { write_status FAILED "missing model: $MODEL"; exit 1; }
  [[ -f "$GPQA_CSV" ]] || { write_status FAILED "missing gpqa csv: $GPQA_CSV"; exit 1; }
  [[ -f scripts/openai_mcq_thinking32_eval.py ]] || { write_status FAILED "missing evaluator"; exit 1; }

  write_status STARTING "launching llama-server"
  log "Starting Qwen3.5-9B Q4_K_M llama-server on $HOST:$PORT"
  "$LLAMA_SERVER" \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --parallel "$PARALLEL" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --flash-attn auto \
    --cont-batching \
    --jinja \
    --reasoning on \
    --reasoning-budget -1 \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "$$ $SERVER_PID" > "$PID_FILE"

  log "Waiting for llama-server readiness"
  for _ in $(seq 1 300); do
    if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      write_status FAILED "llama-server died before readiness"
      exit 1
    fi
    sleep 1
  done
  curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 || {
    write_status FAILED "llama-server readiness timeout"
    exit 1
  }

  write_status RUNNING "running GPQA Diamond thinking-on eval"
  log "Running GPQA Diamond: limit=$LIMIT max_tokens=$MAX_TOKENS concurrency=$CONCURRENCY"
  /root/autodl-tmp/conda-envs/r6000-eval/bin/python scripts/openai_mcq_thinking32_eval.py \
    --task gpqa \
    --base-url "http://$HOST:$PORT/v1" \
    --model qwen3.5-9b-q4km \
    --out "$EVAL_OUT" \
    --gpqa-csv "$GPQA_CSV" \
    --limit "$LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --timeout "$TIMEOUT"

  write_status DONE "eval completed"
  log "DONE: $EVAL_OUT"
  log "SUMMARY: ${EVAL_OUT%.jsonl}.summary.json"
}

main "$@"
