#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18191}"
HOST="${HOST:-127.0.0.1}"
SERVED_NAME="${SERVED_NAME:-qwen35-9b-nvfp4-linear-graph}"
SERVER_LOG="${SERVER_LOG:-${REPORT_DIR}/p150_qwen35_9b_nvfp4_linear_graph_server_${STAMP}.log}"
P25_OUT="${P25_OUT:-${REPORT_DIR}/p150_qwen35_9b_nvfp4_linear_graph_p25_${STAMP}.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${REPORT_DIR}/p150_qwen35_9b_nvfp4_linear_graph_summary_${STAMP}.json}"

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

echo "[p150] root=$ROOT"
echo "[p150] model=$MODEL"
echo "[p150] port=$PORT"
echo "[p150] server_log=$SERVER_LOG"
echo "[p150] p25_out=$P25_OUT"

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
    echo "[p150][error] server exited before ready" >&2
    tail -120 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "[p150][error] server not ready" >&2
  tail -120 "$SERVER_LOG" >&2 || true
  exit 1
fi
echo "[p150] server ready"

"$PYTHON_BIN" benchmarks/p25_server_decode_tps_probe.py \
  --url "http://${HOST}:${PORT}/v1" \
  --model "$SERVED_NAME" \
  --chat \
  --max-tokens 128 256 512 \
  --runs 3 \
  --out "$P25_OUT"

"$PYTHON_BIN" - "$P25_OUT" "$SUMMARY_OUT" <<'PY'
import json
import pathlib
import sys

p25_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
p25 = json.loads(p25_path.read_text())
by = p25.get("summary_by_max_tokens", {})
decode_512 = ((by.get("512") or {}).get("decode_tps") or {}).get("mean")
decode_256 = ((by.get("256") or {}).get("decode_tps") or {}).get("mean")
decode_128 = ((by.get("128") or {}).get("decode_tps") or {}).get("mean")
rows = p25.get("results") or []
graph_reused = [r.get("linear_block_graph_reused") for r in rows if r.get("linear_block_graph_reused") is not None]
summary = {
    "schema": "lynn-qwen35-9b-nvfp4-linear-graph-serving-gate-v1",
    "p25": str(p25_path),
    "decode_tps": {
        "128": decode_128,
        "256": decode_256,
        "512": decode_512,
    },
    "linear_block_graph_reused_values": graph_reused,
    "verdict": "P25_READY" if decode_512 is not None and decode_512 >= 55.0 else "P25_LOW_OR_MISSING",
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[p150] summary=$SUMMARY_OUT"
