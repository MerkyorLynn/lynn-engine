#!/usr/bin/env bash
# R6000 watcher for official Qwen3.6-35B-A3B W4A16 Lynn-native server TPS.
# It waits for the end-to-end pack/generation/MTP pipeline to finish, then
# starts the OpenAI-compatible Lynn server and runs P25 decode TPS probes.

set -euo pipefail

REPO="${REPO:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen36_35b_native_w4a16_mtp}"
PORT="${PORT:-18164}"
HOST="${HOST:-127.0.0.1}"
MAX_TOKENS="${MAX_TOKENS:-128 256}"
RUNS="${RUNS:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
POLL_SECONDS="${POLL_SECONDS:-120}"

mkdir -p "$REPORT_ROOT"
cd "$REPO"

LOG="$REPORT_ROOT/r6000_qwen36_w4a16_server_tps_watch_${TS}.log"
SERVED="Qwen3.6-35B-A3B-W4A16-NVFP4-server-${TS}"
SERVER_LOG="$REPORT_ROOT/r6000_qwen36_w4a16_server_${TS}.log"
OUT="$REPORT_ROOT/r6000_qwen36_w4a16_p25_server_decode_tps_${TS}.json"
HEALTH="$REPORT_ROOT/r6000_qwen36_w4a16_server_${TS}_health.json"

log() {
  printf '[qwen36-w4a16-server-tps] %s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"
}

artifact_ready() {
  MODEL_DIR="$MODEL" "$PY" - <<'PY'
import json
import os
import pathlib

d = pathlib.Path(os.environ["MODEL_DIR"])
index_path = d / "model.safetensors.index.json"
manifest_path = d / "lynn_quant_manifest.json"
if not index_path.exists() or not manifest_path.exists():
    raise SystemExit(1)
index = json.loads(index_path.read_text())
manifest = json.loads(manifest_path.read_text())
files = set(index.get("weight_map", {}).values())
if not files:
    raise SystemExit(2)
missing = [name for name in files if not (d / name).exists() or (d / name).stat().st_size <= 0]
if missing:
    raise SystemExit(3)
if int(manifest.get("quantized_count", 0)) <= 0:
    raise SystemExit(4)
PY
}

heavy_pipeline_active() {
  pgrep -af 'r6000_qwen36_35b_native_w4a16_mtp_pipeline.sh|a100_pack_lynn_native_nvfp4.py|p2_resident_logit_analysis.py|v4_w4a16_w4a8_generation_matrix.py|a100_mtp_forward_smoke.py|a100_mtp_iterative_accept_probe.py|p107_mtp_shadow_serving_credit_probe.py' >/dev/null 2>&1
}

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log "watching model=$MODEL"
while ! artifact_ready; do
  log "artifact not ready"
  sleep "$POLL_SECONDS"
done
log "artifact ready; waiting for end-to-end pipeline/GPU slot"

while heavy_pipeline_active; do
  log "pipeline or MTP/generation probe still active; waiting"
  sleep "$POLL_SECONDS"
done

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LYNN_MOE_IMPL="${LYNN_MOE_IMPL:-packed_nvfp4}"
export LYNN_PACKED_DECODE="${LYNN_PACKED_DECODE:-0}"
export LYNN_PACKED_DECODE_BACKEND="${LYNN_PACKED_DECODE_BACKEND:-native_fast_2d}"
export LYNN_PACKED_DECODE_FULL_ATTN="${LYNN_PACKED_DECODE_FULL_ATTN:-0}"
export LYNN_PACKED_DECODE_LINEAR_ATTN="${LYNN_PACKED_DECODE_LINEAR_ATTN:-0}"
export LYNN_PACKED_DECODE_PREPARE_NATIVE="${LYNN_PACKED_DECODE_PREPARE_NATIVE:-0}"
export LYNN_PACKED_SHARED_EXPERT="${LYNN_PACKED_SHARED_EXPERT:-0}"
export LYNN_NATIVE_FP4_LM_HEAD="${LYNN_NATIVE_FP4_LM_HEAD:-1}"
export LYNN_LINEAR_ATTN_INPROJ_FUSED="${LYNN_LINEAR_ATTN_INPROJ_FUSED:-1}"
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4="${LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4:-1}"
# Quality-safe default for official Qwen3.6-35B-A3B W4A16. The graph path is
# fast, but the 2026-05-18 R6000 long-generation probe showed low-entropy
# repetition with graph reuse enabled; keep it opt-in until parity is fixed.
export LYNN_LINEAR_BLOCK_GRAPH="${LYNN_LINEAR_BLOCK_GRAPH:-0}"
export LYNN_LINEAR_BLOCK_GRAPH_REUSE="${LYNN_LINEAR_BLOCK_GRAPH_REUSE:-0}"
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM="${LYNN_LINEAR_BLOCK_GRAPH_PREWARM:-0}"
if [[ "$LYNN_LINEAR_BLOCK_GRAPH" == "1" ]]; then
  export LYNN_LINEAR_STATE_UPDATE="${LYNN_LINEAR_STATE_UPDATE:-inplace}"
else
  export LYNN_LINEAR_STATE_UPDATE="${LYNN_LINEAR_STATE_UPDATE:-assign}"
fi
export LYNN_FULL_TOKEN_GRAPH_SLOT=0
export LYNN_ROUTER_TOPK_SORTED="${LYNN_ROUTER_TOPK_SORTED:-0}"
export LYNN_MOE_FAST_FIXED="${LYNN_MOE_FAST_FIXED:-0}"
export LYNN_NATIVE_DOWN_BACKEND="${LYNN_NATIVE_DOWN_BACKEND:-triton}"
export LYNN_NATIVE_ACTIVE_MOE="${LYNN_NATIVE_ACTIVE_MOE:-1}"
export LYNN_NATIVE_ACTIVE_MOE_BACKEND="${LYNN_NATIVE_ACTIVE_MOE_BACKEND:-triton}"

log "starting server served=$SERVED port=$PORT"
"$PY" -m server.openai_http \
  --model "$MODEL" \
  --served-name "$SERVED" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 > "$SERVER_LOG" 2>&1 &
server_pid=$!
log "server pid=$server_pid log=$SERVER_LOG"

ready=0
for _ in $(seq 1 300); do
  if curl -fsS "http://${HOST}:${PORT}/health" > "$HEALTH" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    log "server exited before ready"
    tail -120 "$SERVER_LOG" | tee -a "$LOG" || true
    exit 2
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  log "server not ready in time"
  tail -120 "$SERVER_LOG" | tee -a "$LOG" || true
  exit 2
fi

log "server ready; running P25 decode TPS"
"$PY" benchmarks/p25_server_decode_tps_probe.py \
  --url "http://${HOST}:${PORT}/v1" \
  --model "$SERVED" \
  --chat \
  --max-tokens $MAX_TOKENS \
  --runs "$RUNS" \
  --out "$OUT" 2>&1 | tee -a "$LOG"

log "report=$OUT"
log "done"
