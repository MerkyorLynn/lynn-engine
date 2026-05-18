#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18192}"
HOST="${HOST:-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-qwen35-9b-nvfp4-linear-graph-matrix}"
SERVER_LOG="${SERVER_LOG:-${REPORT_DIR}/p151_qwen35_9b_nvfp4_linear_graph_matrix_server_${STAMP}.log}"
MATRIX_OUT="${MATRIX_OUT:-${REPORT_DIR}/p151_qwen35_9b_nvfp4_linear_graph_matrix_${STAMP}.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${REPORT_DIR}/p151_qwen35_9b_nvfp4_linear_graph_matrix_summary_${STAMP}.json}"

cd "$ROOT"
mkdir -p "$REPORT_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[p151] root=$ROOT"
echo "[p151] model=$MODEL"
echo "[p151] matrix_out=$MATRIX_OUT"

"$PYTHON_BIN" -m server.openai_http \
  --model "$MODEL" \
  --served-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[p151][error] server exited before ready" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "[p151][error] server not ready" >&2
  tail -120 "$SERVER_LOG" >&2 || true
  exit 1
fi

"$PYTHON_BIN" benchmarks/openai_serving_matrix_probe.py \
  --url "http://${HOST}:${PORT}/v1" \
  --model "$SERVED_NAME" \
  --single-max-tokens 128 256 512 \
  --runs 1 \
  --concurrency 2 4 8 \
  --concurrent-max-tokens 256 \
  --long-context-chars 4096 16384 32768 \
  --long-context-max-tokens 128 \
  --out "$MATRIX_OUT"

"$PYTHON_BIN" - "$MATRIX_OUT" "$SUMMARY_OUT" <<'PY'
import json
import pathlib
import sys

matrix_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
matrix = json.loads(matrix_path.read_text())
single = {}
for row in (matrix.get("single") or {}).get("rows") or []:
    single[str(row.get("max_tokens"))] = row.get("wall_tps")
concurrent = {}
for key, row in ((matrix.get("concurrency") or {}).get("batch_summary") or {}).items():
    concurrent[key] = row.get("batch_wall_tps")
long_ctx = {}
for row in (matrix.get("long_context") or {}).get("rows") or []:
    long_ctx[str(row.get("target_prompt_chars"))] = row.get("wall_tps")
summary = {
    "schema": "lynn-qwen35-9b-nvfp4-linear-graph-serving-matrix-v1",
    "matrix": str(matrix_path),
    "single_wall_tps": single,
    "concurrent_total_tps": concurrent,
    "long_context_wall_tps": long_ctx,
    "verdict": "MATRIX_DONE",
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[p151] summary=$SUMMARY_OUT"
