#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/lynn-27b-a3b-w4a8-nvfp4-v2}
ENV_FILE=${ENV_FILE:-/tmp/lynn_p16_env.sh}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports/p16_155}
PORT=${PORT:-18163}
HOST=${HOST:-127.0.0.1}
MAX_TOKENS=${MAX_TOKENS:-"256 512"}
RUNS=${RUNS:-1}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}

# Format:
#   name:graph:reuse:prewarm:native_lm_head:down_backend
CONFIGS=${CONFIGS:-"configD:1:1:1:1:triton noLmHead:1:1:1:0:triton noGraphLmHead:0:0:0:1:triton noGraphNoLmHead:0:0:0:0:triton graphNoPrewarm:1:1:0:1:triton"}

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
  local graph="$2"
  local reuse="$3"
  local prewarm="$4"
  local native_lm_head="$5"
  local down_backend="$6"
  local served="Lynn-27B-A3B-W4A8-V2-ablation-${name}-${TS}"
  local server_log="$REPORT_ROOT/r6000_server_ablation_${name}_${TS}.log"
  local out="$REPORT_ROOT/r6000_p25_server_decode_ablation_${name}_${TS}.json"
  local health="$REPORT_ROOT/r6000_server_ablation_${name}_${TS}_health.json"

  echo "[r6000-ablation] start name=${name} graph=${graph} reuse=${reuse} prewarm=${prewarm} lm_head=${native_lm_head} down=${down_backend} ts=${TS}"
  set +u
  source "$ENV_FILE"
  set -u
  export PYTHONPATH="$REPO"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export LYNN_LINEAR_BLOCK_GRAPH="$graph"
  export LYNN_LINEAR_BLOCK_GRAPH_REUSE="$reuse"
  export LYNN_LINEAR_BLOCK_GRAPH_PREWARM="$prewarm"
  export LYNN_FULL_TOKEN_GRAPH_SLOT=0
  export LYNN_PACKED_DECODE=0
  export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
  export LYNN_NATIVE_FP4_LM_HEAD="$native_lm_head"
  export LYNN_NATIVE_DOWN_BACKEND="$down_backend"
  export LYNN_MOE_FAST_FIXED=1

  "$PY" -m server.openai_http \
    --model "$MODEL" \
    --served-name "$served" \
    --host "$HOST" \
    --port "$PORT" > "$server_log" 2>&1 &
  current_pid=$!
  echo "[r6000-ablation] server pid=${current_pid} log=${server_log}"

  local ready=0
  for _ in $(seq 1 300); do
    if curl -fsS "http://${HOST}:${PORT}/health" > "$health" 2>/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$current_pid" 2>/dev/null; then
      echo "[r6000-ablation] server exited before ready"
      tail -80 "$server_log" || true
      return 1
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "[r6000-ablation] server not ready in time"
    return 1
  fi

  echo "[r6000-ablation] health ready ${health}"
  "$PY" benchmarks/p25_server_decode_tps_probe.py \
    --url "http://${HOST}:${PORT}/v1" \
    --model "$served" \
    --max-tokens $MAX_TOKENS \
    --runs "$RUNS" \
    --out "$out"
  echo "[r6000-ablation] report ${out}"

  kill "$current_pid" 2>/dev/null || true
  wait "$current_pid" 2>/dev/null || true
  current_pid=""
  sleep 5
}

for item in $CONFIGS; do
  IFS=: read -r name graph reuse prewarm native_lm_head down_backend <<< "$item"
  run_config "$name" "$graph" "$reuse" "$prewarm" "$native_lm_head" "$down_backend"
done

"$PY" - "$REPORT_ROOT" "$TS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
ts = sys.argv[2]
paths = sorted(root.glob(f"r6000_p25_server_decode_ablation_*_{ts}.json"))
summary = {}
for path in paths:
    data = json.loads(path.read_text())
    name = path.name.replace("r6000_p25_server_decode_ablation_", "").replace(f"_{ts}.json", "")
    rows = data.get("results", [])
    first = rows[0] if rows else {}
    summary[name] = {
        "decode_tps_mean_by_max_tokens": {
            key: value.get("decode_tps", {}).get("mean")
            for key, value in data.get("summary_by_max_tokens", {}).items()
        },
        "wall_tps_mean_by_max_tokens": {
            key: value.get("wall_tps", {}).get("mean")
            for key, value in data.get("summary_by_max_tokens", {}).items()
        },
        "decode_step_ms_median_by_max_tokens": {
            key: value.get("decode_step_ms_median", {}).get("mean")
            for key, value in data.get("summary_by_max_tokens", {}).items()
        },
        "linear_block_graph_reused": first.get("linear_block_graph_reused"),
        "linear_block_graph_capture_seconds": first.get("linear_block_graph_capture_seconds"),
        "linear_block_graph_prewarm_seconds": first.get("linear_block_graph_prewarm_seconds"),
        "native_fp4_lm_head_enabled": first.get("native_fp4_lm_head_enabled"),
        "preview_sample": first.get("preview"),
    }
out = root / f"r6000_p25_server_decode_ablation_summary_{ts}.json"
out.write_text(
    json.dumps(
        {
            "schema_version": "r6000-p25-server-decode-ablation-summary-v1",
            "timestamp": ts,
            "reports": [str(p) for p in paths],
            "summary": summary,
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
