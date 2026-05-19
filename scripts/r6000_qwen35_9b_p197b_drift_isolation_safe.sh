#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# R6000: P197b drift isolation with GPU-busy guard
#
# Checks whether the R6000 GPU is occupied by long evals
# (openai_mcq_thinking32_eval, llama-server, etc.) before
# running the isolation probe. If busy → REFUSE_RUN + status JSON.
# ─────────────────────────────────────────────────────────────

ROOT="${ROOT:-/root/autodl-tmp/lynn-engine}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPORT_DIR:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
MAX_NEW="${MAX_NEW:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
LIMIT="${LIMIT:-5}"
MODES="${MODES:-scalar_full,scalar_gateup_only,scalar_down_only}"
FORCE="${FORCE:-0}"

# CUDA build env
export LYNN_NATIVE_CUDA_ARCH="${LYNN_NATIVE_CUDA_ARCH:-sm_120a}"
export LYNN_ENABLE_SM120A_FP4_MMA="${LYNN_ENABLE_SM120A_FP4_MMA:-1}"

OUT_JSON="${REPORT_DIR}/p197b_drift_isolation_${STAMP}.json"
STATUS_JSON="${REPORT_DIR}/p197b_drift_isolation_status_${STAMP}.json"

cd "$ROOT"
mkdir -p "$REPORT_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────
# GPU-busy guard
# ─────────────────────────────────────────────────────────────
gpu_busy_reason=""

# Check 1: llama-server on known eval ports
for port in 18197 18198 18099; do
  if ss -tlnp 2>/dev/null | grep -q ":${port} " || \
     netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
    pid="$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
    if [[ -n "$pid" ]]; then
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 200 || echo 'unknown')"
      gpu_busy_reason="port $port occupied by pid=$pid ($cmdline)"
    else
      gpu_busy_reason="port $port is listening (unknown process)"
    fi
    break
  fi
done

# Check 2: GPU memory usage > 80%
if [[ -z "$gpu_busy_reason" ]] && command -v nvidia-smi &>/dev/null; then
  gpu_mem_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  gpu_mem_total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [[ -n "$gpu_mem_used" && -n "$gpu_mem_total" && "$gpu_mem_total" -gt 0 ]]; then
    pct=$((gpu_mem_used * 100 / gpu_mem_total))
    if [[ "$pct" -gt 80 ]]; then
      gpu_busy_reason="GPU memory ${pct}% used (${gpu_mem_used}/${gpu_mem_total} MiB)"
    fi
  fi
fi

# Check 3: Known eval processes
if [[ -z "$gpu_busy_reason" ]]; then
  for pattern in "openai_mcq_thinking32" "llama-server.*18197" "sglang.*serve" "vllm.*serve"; do
    pid="$(pgrep -f "$pattern" 2>/dev/null | head -1 || true)"
    if [[ -n "$pid" ]]; then
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | head -c 200 || echo 'unknown')"
      gpu_busy_reason="process matches '$pattern': pid=$pid ($cmdline)"
      break
    fi
  done
fi

# ─────────────────────────────────────────────────────────────
# REFUSE_RUN if busy (unless FORCE=1)
# ─────────────────────────────────────────────────────────────
if [[ -n "$gpu_busy_reason" && "$FORCE" != "1" ]]; then
  echo "[p197b] REFUSE_RUN: GPU is busy" >&2
  echo "[p197b] Reason: $gpu_busy_reason" >&2
  echo "[p197b] Set FORCE=1 to override." >&2
  echo "[p197b] Writing status to: $STATUS_JSON" >&2

  cat > "$STATUS_JSON" <<ENDJSON
{
  "schema": "lynn-p197b-status-v1",
  "created": "$(date -Iseconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S%z)",
  "status": "REFUSE_RUN",
  "reason": "$gpu_busy_reason",
  "model": "$MODEL",
  "modes_requested": "$MODES",
  "advice": "Wait for the running eval to complete, then re-run this script.",
  "gpu_mem_used_mib": ${gpu_mem_used:-null},
  "gpu_mem_total_mib": ${gpu_mem_total:-null}
}
ENDJSON

  echo ""
  echo "[p197b] Status JSON:"
  cat "$STATUS_JSON"
  exit 2
fi

# ─────────────────────────────────────────────────────────────
# GPU is free — run isolation probe
# ─────────────────────────────────────────────────────────────
echo "[p197b] GPU is free — starting isolation probe"
echo "[p197b] Model: $MODEL"
echo "[p197b] Modes: $MODES"
echo "[p197b] Output: $OUT_JSON"
echo ""

"$PYTHON_BIN" benchmarks/p197b_qwen35_9b_w4a8_drift_isolation.py \
  --model "$MODEL" \
  --max-new "$MAX_NEW" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --limit "$LIMIT" \
  --modes "$MODES" \
  --out "$OUT_JSON"

echo "[p197b] done report=$OUT_JSON"
