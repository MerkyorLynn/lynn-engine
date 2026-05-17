#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
ENV_FILE=${ENV_FILE:-/tmp/lynn_p16_env.sh}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports/p16_155}
PORT=${PORT:-18162}
HOST=${HOST:-127.0.0.1}
MAX_TOKENS=${MAX_TOKENS:-"128 256 512"}
RUNS=${RUNS:-2}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}

# Format: "name:down_backend name2:down_backend2".
CONFIGS=${CONFIGS:-"configD:triton cudaTileDown:cuda_tile"}

mkdir -p "$REPORT_ROOT"
cd "$REPO"

current_pid=""
cleanup() {
  if [[ -n "$current_pid" ]]; then
    kill "$current_pid" 2>/dev/null || true
    wait "$current_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_config() {
  local name="$1"
  local down_backend="$2"
  local served="Lynn-27B-A3B-W4A8-V2-${name}-${TS}"
  local server_log="$REPORT_ROOT/r6000_server_${name}_${TS}.log"
  local out="$REPORT_ROOT/r6000_p25_server_decode_${name}_${TS}.json"
  local health="$REPORT_ROOT/r6000_server_${name}_${TS}_health.json"

  echo "[r6000-sweep] start name=${name} down=${down_backend} ts=${TS}"
  set +u
  source "$ENV_FILE"
  set -u
  export PYTHONPATH="$REPO"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export LYNN_LINEAR_BLOCK_GRAPH=1
  export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
  export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
  export LYNN_FULL_TOKEN_GRAPH_SLOT=0
  export LYNN_PACKED_DECODE=0
  export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
  export LYNN_NATIVE_DOWN_BACKEND="$down_backend"
  export LYNN_MOE_FAST_FIXED=1

  "$PY" -m server.openai_http \
    --model "$MODEL" \
    --served-name "$served" \
    --host "$HOST" \
    --port "$PORT" > "$server_log" 2>&1 &
  current_pid=$!
  echo "[r6000-sweep] server pid=${current_pid} log=${server_log}"

  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "http://${HOST}:${PORT}/health" > "$health" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$current_pid" 2>/dev/null; then
      echo "[r6000-sweep] server exited before ready"
      tail -80 "$server_log" || true
      return 1
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "[r6000-sweep] server not ready in time"
    return 1
  fi

  echo "[r6000-sweep] health ready ${health}"
  "$PY" benchmarks/p25_server_decode_tps_probe.py \
    --url "http://${HOST}:${PORT}/v1" \
    --model "$served" \
    --max-tokens $MAX_TOKENS \
    --runs "$RUNS" \
    --out "$out"
  echo "[r6000-sweep] report ${out}"

  kill "$current_pid" 2>/dev/null || true
  wait "$current_pid" 2>/dev/null || true
  current_pid=""
  sleep 5
}

for item in $CONFIGS; do
  name=${item%%:*}
  down_backend=${item#*:}
  run_config "$name" "$down_backend"
done

"$PY" - "$REPORT_ROOT" "$TS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
ts = sys.argv[2]
paths = sorted(root.glob(f"r6000_p25_server_decode_*_{ts}.json"))
summary = {}
quality_flags = {}
for path in paths:
    data = json.loads(path.read_text())
    name = path.name.replace("r6000_p25_server_decode_", "").replace(f"_{ts}.json", "")
    summary[name] = {
        key: value.get("decode_tps", {}).get("mean")
        for key, value in data.get("summary_by_max_tokens", {}).items()
    }
    previews = [str(row.get("preview", "")) for row in data.get("results", [])]
    quality_flags[name] = {
        "all_preview_exclamation_loop": bool(previews) and all(set(p.strip()) <= {"!"} for p in previews),
        "preview_samples": previews[:2],
    }
out = root / f"r6000_p25_server_decode_sweep_{ts}.json"
out.write_text(
    json.dumps(
        {
            "schema_version": "r6000-p25-server-decode-env-sweep-v1",
            "timestamp": ts,
            "reports": [str(p) for p in paths],
            "decode_tps_mean_by_config": summary,
            "quality_flags": quality_flags,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(out)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
