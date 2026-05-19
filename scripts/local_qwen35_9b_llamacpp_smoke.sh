#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18197}"
SERVED_NAME="${SERVED_NAME:-qwen35-9b-q4km}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-}"
PARALLEL="${PARALLEL:-1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
REASONING="${REASONING:-auto}"
TIMEOUT="${TIMEOUT:-300}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
REPORT="${REPORT:-$ROOT/reports/qwen35_9b/mac_smoke_${STAMP}.json}"
LOG="${LOG:-$ROOT/reports/qwen35_9b/mac_smoke_${STAMP}.server.log}"
MODEL="${MODEL:-}"
LLAMA_SERVER="${LLAMA_SERVER:-}"
EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local_qwen35_9b_llamacpp_smoke.sh [options]

Runs the Mac Qwen3.5-9B Q4_K_M llama.cpp smoke chain:
  1. discover GGUF and llama-server
  2. start an OpenAI-compatible llama-server endpoint
  3. check /health
  4. run one chat completion
  5. run one 256-token decode TPS smoke
  6. write reports/qwen35_9b/mac_smoke_<stamp>.json

Options:
  --model PATH          Q4_K_M GGUF path; same as MODEL env.
  --llama-server PATH   llama-server binary; same as LLAMA_SERVER env.
  --host HOST           Bind host (default: 127.0.0.1).
  --port PORT           Server port (default: 18197); same as PORT env.
  --served-name NAME    Served model name (default: qwen35-9b-q4km).
  --ctx-size N          llama.cpp context size (default: 32768).
  --threads N           llama.cpp thread count (default: auto-detect).
  --parallel N          llama.cpp parallel slots (default: 1).
  --gpu-layers N        llama.cpp GPU layers (default: 999).
  --reasoning auto|on   llama.cpp reasoning mode (default: auto).
  --report PATH         JSON report path.
  --log PATH            server log path.
  --timeout SECONDS     server/test timeout (default: 300).
  --dry-run             Print resolved config and command without starting.
  -h, --help            Show help.

Discovery order:
  GGUF:          $MODEL, ./models, ~/models, /Users/lynn/Downloads/Lynn/models
  llama-server:  $LLAMA_SERVER, llama-server in PATH, common llama.cpp build/bin paths
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --llama-server) LLAMA_SERVER="${2:-}"; shift 2 ;;
    --host) HOST="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --served-name) SERVED_NAME="${2:-}"; shift 2 ;;
    --ctx-size|--ctx) CTX_SIZE="${2:-}"; shift 2 ;;
    --threads) THREADS="${2:-}"; shift 2 ;;
    --parallel) PARALLEL="${2:-}"; shift 2 ;;
    --gpu-layers) N_GPU_LAYERS="${2:-}"; shift 2 ;;
    --reasoning) REASONING="${2:-}"; shift 2 ;;
    --report) REPORT="${2:-}"; shift 2 ;;
    --log) LOG="${2:-}"; shift 2 ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[mac-smoke] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$REASONING" in
  auto|on) ;;
  *) echo "[mac-smoke] --reasoning must be auto or on" >&2; exit 2 ;;
esac

if [[ -z "$THREADS" ]]; then
  THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
fi

find_model() {
  if [[ -n "$MODEL" ]]; then
    if [[ -s "$MODEL" ]]; then
      printf '%s\n' "$MODEL"
      return 0
    fi
    echo "[mac-smoke] MODEL is set but file is missing or empty: $MODEL" >&2
    return 1
  fi

  local roots=(
    "$ROOT/models"
    "$PWD/models"
    "$HOME/models"
    "/Users/lynn/Downloads/Lynn/models"
  )
  local root
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r candidate; do
      [[ -s "$candidate" ]] || continue
      printf '%s\n' "$candidate"
      return 0
    done < <(find "$root" -maxdepth 5 -type f \( \
      -iname '*Qwen3.5*9B*Q4*K*M*.gguf' -o \
      -iname '*qwen3.5*9b*q4*k*m*.gguf' -o \
      -iname '*Qwen3.5*9B*Q4_K_M*.gguf' -o \
      -iname '*qwen3.5*9b*q4_k_m*.gguf' \
    \) 2>/dev/null | sort)
  done
  return 1
}

find_llama_server() {
  if [[ -n "$LLAMA_SERVER" ]]; then
    if [[ -x "$LLAMA_SERVER" ]]; then
      printf '%s\n' "$LLAMA_SERVER"
      return 0
    fi
    echo "[mac-smoke] LLAMA_SERVER is set but not executable: $LLAMA_SERVER" >&2
    return 1
  fi

  local candidates=(
    "$(command -v llama-server 2>/dev/null || true)"
    "/opt/homebrew/bin/llama-server"
    "/usr/local/bin/llama-server"
    "$HOME/llama.cpp/build/bin/llama-server"
    "$HOME/llama.cpp/build/tools/server/llama-server"
    "$HOME/src/llama.cpp/build/bin/llama-server"
    "$HOME/dev/llama.cpp/build/bin/llama-server"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate:-}" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

port_in_use() {
  python3 - "$HOST" "$PORT" <<'PY'
import socket
import sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    ok = sock.connect_ex((host, port)) == 0
finally:
    sock.close()
raise SystemExit(0 if ok else 1)
PY
}

MODEL_PATH="$(find_model || true)"
if [[ -z "$MODEL_PATH" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    MODEL_PATH="${MODEL:-/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf}"
  else
  cat >&2 <<EOF
[mac-smoke] ERROR: Qwen3.5-9B Q4_K_M GGUF not found.

Searched in order:
  1. MODEL env / --model
  2. $ROOT/models
  3. $PWD/models
  4. $HOME/models
  5. /Users/lynn/Downloads/Lynn/models

Fix:
  export MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf
  bash scripts/local_qwen35_9b_llamacpp_smoke.sh
EOF
  exit 4
  fi
fi

LLAMA_SERVER_BIN="$(find_llama_server || true)"
if [[ -z "$LLAMA_SERVER_BIN" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    LLAMA_SERVER_BIN="${LLAMA_SERVER:-/absolute/path/to/llama-server}"
  else
  cat >&2 <<'EOF'
[mac-smoke] ERROR: llama-server not found.

Fix with Homebrew:
  brew install llama.cpp

Or build llama.cpp with Metal:
  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_METAL=ON
  cmake --build ~/llama.cpp/build -j
  export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server

Then rerun:
  bash scripts/local_qwen35_9b_llamacpp_smoke.sh
EOF
  exit 5
  fi
fi

if [[ "$DRY_RUN" != "1" ]] && port_in_use; then
  cat >&2 <<EOF
[mac-smoke] ERROR: $HOST:$PORT is already in use.

Fix options:
  lsof -nP -iTCP:$PORT -sTCP:LISTEN
  kill <pid>

Or choose another port:
  PORT=18198 bash scripts/local_qwen35_9b_llamacpp_smoke.sh
EOF
  exit 6
fi

mkdir -p "$(dirname "$REPORT")" "$(dirname "$LOG")"

CMD=(
  "$LLAMA_SERVER_BIN"
  --model "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX_SIZE"
  --threads "$THREADS"
  --parallel "$PARALLEL"
  --n-gpu-layers "$N_GPU_LAYERS"
  --jinja
  --reasoning "$REASONING"
  -a "$SERVED_NAME"
)

if [[ -n "$EXTRA_ARGS" ]]; then
  read -ra EXTRA_ARR <<< "$EXTRA_ARGS"
  CMD+=("${EXTRA_ARR[@]}")
fi

cat <<EOF
[mac-smoke] model=$MODEL_PATH
[mac-smoke] llama_server=$LLAMA_SERVER_BIN
[mac-smoke] endpoint=http://$HOST:$PORT/v1
[mac-smoke] served_name=$SERVED_NAME
[mac-smoke] reasoning=$REASONING
[mac-smoke] report=$REPORT
[mac-smoke] log=$LOG
EOF

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[mac-smoke] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"${CMD[@]}" >"$LOG" 2>&1 &
SERVER_PID=$!
echo "[mac-smoke] server_pid=$SERVER_PID"

ready=0
for _ in $(seq 1 "$TIMEOUT"); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[mac-smoke] ERROR: llama-server exited before /health became ready" >&2
    echo "[mac-smoke] server log tail:" >&2
    tail -120 "$LOG" >&2 || true
    exit 7
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "[mac-smoke] ERROR: /health did not become ready within ${TIMEOUT}s" >&2
  echo "[mac-smoke] server log tail:" >&2
  tail -120 "$LOG" >&2 || true
  exit 8
fi

echo "[mac-smoke] /health OK; running chat and 256-token decode TPS smoke"
python3 - "$HOST" "$PORT" "$SERVED_NAME" "$MODEL_PATH" "$LLAMA_SERVER_BIN" "$REPORT" "$LOG" "$TIMEOUT" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

host, port, model, model_path, server_bin, report_path, log_path, timeout_raw = sys.argv[1:9]
timeout = float(timeout_raw)
base_url = f"http://{host}:{port}/v1"
health_url = f"http://{host}:{port}/health"


def get_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", "replace")
        return resp.status, raw


def post_chat(content: str, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    elapsed = time.time() - t0
    data = json.loads(raw)
    text = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = max(1, len(text.split()))
    tps = float(completion_tokens) / elapsed if elapsed > 0 else None
    return {
        "ok": bool(text.strip()),
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "decode_tps": tps,
        "text_preview": text[:240],
        "response_id": data.get("id"),
    }

started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
errors: list[str] = []
checks: dict[str, object] = {}

try:
    status, raw = get_json(health_url)
    checks["health"] = {"ok": status == 200, "http_status": status, "body_preview": raw[:200]}
    if status != 200:
        errors.append(f"health returned HTTP {status}")
except Exception as exc:
    checks["health"] = {"ok": False, "error": str(exc)}
    errors.append(f"health error: {exc}")

try:
    chat = post_chat("Say OK in one short sentence.", 64)
    checks["chat"] = chat
    if not chat["ok"]:
        errors.append("chat returned empty content")
except Exception as exc:
    checks["chat"] = {"ok": False, "error": str(exc)}
    errors.append(f"chat error: {exc}")

try:
    tps = post_chat(
        "Write a compact numbered checklist for validating a local coding model endpoint. Keep going until the token budget ends.",
        256,
    )
    checks["decode_tps_256"] = tps
    if not tps["ok"]:
        errors.append("256-token decode smoke returned empty content")
except Exception as exc:
    checks["decode_tps_256"] = {"ok": False, "error": str(exc)}
    errors.append(f"256-token decode smoke error: {exc}")

report = {
    "schema": "lynn-qwen35-9b-mac-llamacpp-smoke-v1",
    "started_utc": started,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ok": not errors,
    "base_url": base_url,
    "model": model,
    "model_path": model_path,
    "llama_server": server_bin,
    "server_log": log_path,
    "checks": checks,
    "errors": errors,
}

Path(report_path).parent.mkdir(parents=True, exist_ok=True)
Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "ok": report["ok"],
    "report": report_path,
    "chat_ok": (checks.get("chat") or {}).get("ok") if isinstance(checks.get("chat"), dict) else None,
    "decode_tps_256": (checks.get("decode_tps_256") or {}).get("decode_tps") if isinstance(checks.get("decode_tps_256"), dict) else None,
    "errors": errors,
}, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["ok"] else 9)
PY

echo "[mac-smoke] report written: $REPORT"
