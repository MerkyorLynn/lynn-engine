#!/usr/bin/env bash
set -euo pipefail

# R6000 Qwen3.5-9B Q4_K_M-imatrix gate.
#
# This is intentionally queued behind the current official 9B runs.  It builds
# the missing llama.cpp imatrix/quantize binaries if needed, converts the
# official BF16 HF package to F16 GGUF, computes an imatrix from the Lynn
# calibration corpus, quantizes Q4_K_M with that imatrix, and then runs the same
# quality/speed gate used for the official Q4_K_M comparison.

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
MODEL_ROOT="${MODEL_ROOT:-/root/autodl-tmp/models}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"

LLAMA_CPP="${LLAMA_CPP:-/root/autodl-tmp/llama.cpp}"
LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_CPP/build-cuda}"
BF16_MODEL="${BF16_MODEL:-$MODEL_ROOT/Qwen3.5-9B-BF16}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/Qwen3.5-9B-GGUF-imatrix}"
F16_GGUF="${F16_GGUF:-$OUT_DIR/Qwen3.5-9B-F16.gguf}"
IMATRIX="${IMATRIX:-$OUT_DIR/Qwen3.5-9B.imatrix}"
Q4_IMATRIX="${Q4_IMATRIX:-$OUT_DIR/Qwen3.5-9B-Q4_K_M-imatrix.gguf}"

CALIB_JSONL="${CALIB_JSONL:-$ROOT/pruning/calibration/calibration_set_v1.1.jsonl}"
CALIB_TXT="${CALIB_TXT:-$OUT_DIR/qwen35_9b_imatrix_calib.txt}"
CALIB_LINES="${CALIB_LINES:-768}"
IMATRIX_CHUNKS="${IMATRIX_CHUNKS:-256}"
IMATRIX_CTX="${IMATRIX_CTX:-512}"

SERVER_PORT="${SERVER_PORT:-18198}"
SERVED_NAME="${SERVED_NAME:-Qwen3.5-9B-Q4KM-imatrix}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-16}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"

MMLU_DATA_DIR="${MMLU_DATA_DIR:-/tmp/datasets}"
GPQA_CSV="${GPQA_CSV:-/tmp/datasets/gpqa/gpqa_diamond.csv}"
THINK_GPQA_LIMIT="${THINK_GPQA_LIMIT:-50}"
RUN_MMLU="${RUN_MMLU:-1}"
RUN_GPQA="${RUN_GPQA:-1}"
RUN_THINK_GPQA="${RUN_THINK_GPQA:-1}"
RUN_TPS="${RUN_TPS:-1}"

LOG="$REPORT_DIR/q4km_imatrix_gate_${STAMP}.log"
SUMMARY="$REPORT_DIR/q4km_imatrix_gate_${STAMP}.summary.json"
mkdir -p "$REPORT_DIR" "$OUT_DIR"

log() {
  printf '[q4km-imatrix-9b] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

is_alive_pid_file() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local pid
  pid="$(cat "$f" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

wait_for_current_9b_queue() {
  local watched=(
    "$REPORT_DIR/q4km_cuda_gpqa50.pid"
    "$REPORT_DIR/run_official_9b_round_after_q4km_20260519.pid"
  )
  while true; do
    local active=0
    for f in "${watched[@]}"; do
      if is_alive_pid_file "$f"; then
        active=1
        log "waiting for existing 9B job pid=$(cat "$f") file=$f"
      fi
    done
    if [[ "$active" == "0" ]]; then
      return 0
    fi
    sleep 120
  done
}

find_exe() {
  local name="$1"
  shift
  for cand in "$@"; do
    if [[ -x "$cand" ]]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  find "$LLAMA_BUILD" "$LLAMA_CPP/build" -type f -name "$name" -perm -111 2>/dev/null | head -1
}

build_llama_tools() {
  log "building llama.cpp imatrix/quantize/bench tools if missing"
  cmake --build "$LLAMA_BUILD" --target llama-imatrix llama-quantize llama-bench -j "$(nproc)" 2>&1 | tee -a "$LOG"
  LLAMA_IMATRIX="$(find_exe llama-imatrix \
    "$LLAMA_BUILD/bin/llama-imatrix" \
    "$LLAMA_BUILD/tools/imatrix/llama-imatrix")"
  LLAMA_QUANTIZE="$(find_exe llama-quantize \
    "$LLAMA_BUILD/bin/llama-quantize" \
    "$LLAMA_CPP/build/bin/llama-quantize")"
  LLAMA_SERVER="$(find_exe llama-server \
    "$LLAMA_BUILD/bin/llama-server" \
    "$LLAMA_CPP/build/bin/llama-server")"
  if [[ -z "${LLAMA_IMATRIX:-}" || -z "${LLAMA_QUANTIZE:-}" || -z "${LLAMA_SERVER:-}" ]]; then
    log "missing llama.cpp tool(s): imatrix=$LLAMA_IMATRIX quantize=$LLAMA_QUANTIZE server=$LLAMA_SERVER"
    exit 2
  fi
  log "tools: imatrix=$LLAMA_IMATRIX quantize=$LLAMA_QUANTIZE server=$LLAMA_SERVER"
}

convert_f16() {
  if [[ -s "$F16_GGUF" ]]; then
    log "F16 GGUF exists: $(du -h "$F16_GGUF" | awk '{print $1}')"
    return 0
  fi
  log "converting official BF16 HF package to F16 GGUF"
  (cd "$LLAMA_CPP" && "$PYTHON_BIN" convert_hf_to_gguf.py "$BF16_MODEL" --outfile "$F16_GGUF" --outtype f16) 2>&1 | tee -a "$LOG"
  test -s "$F16_GGUF"
  log "F16 GGUF ready: $(du -h "$F16_GGUF" | awk '{print $1}')"
}

build_calib_txt() {
  if [[ -s "$CALIB_TXT" ]]; then
    log "calibration text exists: $CALIB_TXT"
    return 0
  fi
  log "building calibration text from $CALIB_JSONL"
  "$PYTHON_BIN" - "$CALIB_JSONL" "$CALIB_TXT" "$CALIB_LINES" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
limit = int(sys.argv[3])
texts = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        text = (obj.get("text") or obj.get("prompt") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n\n".join(texts) + "\n", encoding="utf-8")
print(json.dumps({"source": str(src), "out": str(dst), "lines": len(texts)}, ensure_ascii=False))
PY
  test -s "$CALIB_TXT"
}

run_imatrix() {
  if [[ -s "$IMATRIX" ]]; then
    log "imatrix exists: $(du -h "$IMATRIX" | awk '{print $1}')"
    return 0
  fi
  log "running imatrix calibration chunks=$IMATRIX_CHUNKS ctx=$IMATRIX_CTX"
  "$LLAMA_IMATRIX" \
    -m "$F16_GGUF" \
    -f "$CALIB_TXT" \
    -o "$IMATRIX" \
    --chunks "$IMATRIX_CHUNKS" \
    -ngl "$N_GPU_LAYERS" \
    -c "$IMATRIX_CTX" 2>&1 | tee -a "$LOG"
  test -s "$IMATRIX"
  log "imatrix ready: $(du -h "$IMATRIX" | awk '{print $1}')"
}

quantize_q4km() {
  if [[ -s "$Q4_IMATRIX" ]]; then
    log "Q4_K_M-imatrix GGUF exists: $(du -h "$Q4_IMATRIX" | awk '{print $1}')"
    return 0
  fi
  log "quantizing Q4_K_M with imatrix"
  "$LLAMA_QUANTIZE" \
    --imatrix "$IMATRIX" \
    "$F16_GGUF" \
    "$Q4_IMATRIX" \
    Q4_K_M 2>&1 | tee -a "$LOG"
  test -s "$Q4_IMATRIX"
  log "Q4_K_M-imatrix ready: $(du -h "$Q4_IMATRIX" | awk '{print $1}')"
}

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

start_server() {
  local server_log="$REPORT_DIR/q4km_imatrix_server_${STAMP}.log"
  log "starting llama-server for imatrix gate port=$SERVER_PORT"
  "$LLAMA_SERVER" \
    --model "$Q4_IMATRIX" \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --threads "$THREADS" \
    --jinja \
    -a "$SERVED_NAME" > "$server_log" 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 180); do
    sleep 5
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "server exited before ready"
      tail -120 "$server_log" | tee -a "$LOG" || true
      exit 3
    fi
    if curl -s -m 30 -H 'Content-Type: application/json' \
      -d '{"model":"'"$SERVED_NAME"'","prompt":"A","max_tokens":4,"temperature":0}' \
      "http://127.0.0.1:$SERVER_PORT/v1/completions" | grep -q '"choices"'; then
      log "server ready"
      return 0
    fi
  done
  log "server not ready"
  tail -120 "$server_log" | tee -a "$LOG" || true
  exit 4
}

run_quality() {
  local prefix="$REPORT_DIR/q4km_imatrix_${STAMP}"
  if [[ "$RUN_MMLU" == "1" ]]; then
    log "running MMLU500 5-shot"
    "$PYTHON_BIN" "$ROOT/scripts/openai_mmlu_500_5shot_eval.py" \
      --data-dir "$MMLU_DATA_DIR" \
      --base-url "http://127.0.0.1:$SERVER_PORT/v1" \
      --model "$SERVED_NAME" \
      --out "${prefix}_mmlu_n500.jsonl" \
      --concurrency 6 \
      --shots 5 \
      --sample 500 \
      --timeout 180 2>&1 | tee -a "$LOG"
  fi
  if [[ "$RUN_GPQA" == "1" ]]; then
    log "running GPQA Diamond thinking-off style"
    "$PYTHON_BIN" "$ROOT/scripts/openai_gpqa_diamond_eval.py" \
      --csv "$GPQA_CSV" \
      --base-url "http://127.0.0.1:$SERVER_PORT/v1" \
      --model "$SERVED_NAME" \
      --out "${prefix}_gpqa.jsonl" \
      --concurrency 2 \
      --timeout 180 2>&1 | tee -a "$LOG"
  fi
  if [[ "$RUN_THINK_GPQA" == "1" ]]; then
    log "running GPQA thinking32 limit=$THINK_GPQA_LIMIT"
    "$PYTHON_BIN" "$ROOT/scripts/openai_mcq_thinking32_eval.py" \
      --task gpqa \
      --gpqa-csv "$GPQA_CSV" \
      --base-url "http://127.0.0.1:$SERVER_PORT/v1" \
      --model "$SERVED_NAME" \
      --out "${prefix}_thinking32_gpqa${THINK_GPQA_LIMIT}.jsonl" \
      --limit "$THINK_GPQA_LIMIT" \
      --max-tokens 32768 \
      --concurrency 1 \
      --timeout 3600 2>&1 | tee -a "$LOG"
  fi
}

run_tps() {
  if [[ "$RUN_TPS" != "1" ]]; then
    return 0
  fi
  cleanup_server
  SERVER_PID=""
  log "running Q4_K_M-imatrix llama.cpp TPS baseline"
  GGUF="$Q4_IMATRIX" \
  PORT=18199 \
  SERVED_NAME="${SERVED_NAME}-TPS" \
  STAMP="${STAMP}_imatrix_tps" \
  QUALITY=0 \
  "$ROOT/scripts/r6000_qwen35_9b_q4km_llamacpp_baseline.sh" 2>&1 | tee -a "$LOG"
}

write_summary() {
  "$PYTHON_BIN" - "$SUMMARY" "$STAMP" "$Q4_IMATRIX" "$F16_GGUF" "$IMATRIX" "$REPORT_DIR" <<'PY'
import glob
import json
import os
import sys

out, stamp, q4, f16, imx, report_dir = sys.argv[1:]
def load(path):
    if path and os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception as exc:
            return {"error": repr(exc), "path": path}
    return None

summary = {
    "schema": "lynn-qwen35-9b-q4km-imatrix-gate-v1",
    "stamp": stamp,
    "q4_imatrix": q4,
    "f16_gguf": f16,
    "imatrix": imx,
    "mmlu": load(f"{report_dir}/q4km_imatrix_{stamp}_mmlu_n500.summary.json"),
    "gpqa": load(f"{report_dir}/q4km_imatrix_{stamp}_gpqa.summary.json"),
    "thinking32_gpqa": load(f"{report_dir}/q4km_imatrix_{stamp}_thinking32_gpqa50.summary.json"),
    "tps_reports": sorted(glob.glob(f"{report_dir}/r6000_qwen35_9b_q4km_baseline_{stamp}_imatrix_tps*.json")),
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False))
PY
  log "summary written: $SUMMARY"
}

main() {
  cd "$ROOT"
  log "start stamp=$STAMP"
  wait_for_current_9b_queue
  build_llama_tools
  convert_f16
  build_calib_txt
  run_imatrix
  quantize_q4km
  start_server
  run_quality
  run_tps
  write_summary
  log "done"
}

main "$@"
