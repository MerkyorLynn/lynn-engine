#!/usr/bin/env bash
set -euo pipefail

# Unified promotion gate for Qwen3.6-35B-A3B Lynn-native W4A16 runtime candidates.
#
# Example:
#   CANDIDATE_NAME=amber_shared_gate \
#   CANDIDATE_ENV='LYNN_SHARED_EXPERT_GATE_BACKEND=triton LYNN_LINEAR_ATTN_CONV_BACKEND=triton_inplace' \
#   scripts/r6000_qwen36_candidate_promotion_gate.sh
#
# Safer remote example that avoids shell quoting mistakes:
#   CANDIDATE_NAME=amber_shared_gate \
#   CANDIDATE_ENV_FILE=scripts/qwen36_candidate_env_amber_sharedgate_convinplace.env \
#   scripts/r6000_qwen36_candidate_promotion_gate.sh
#
# The script is designed for R6000, but paths are overridable for local smoke.

REPO=${REPO:-/root/autodl-tmp/lynn-engine}
PY=${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}
MODEL=${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}
REPORT_ROOT=${REPORT_ROOT:-/root/autodl-tmp/reports/qwen36_35b}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18169}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
CANDIDATE_NAME=${CANDIDATE_NAME:-candidate}
CANDIDATE_ENV=${CANDIDATE_ENV:-}
CANDIDATE_ENV_FILE=${CANDIDATE_ENV_FILE:-}
PROMPT_SPECS=${PROMPT_SPECS:-scripts/qwen36_structured_hard_prompts.json}
P37_MAX_NEW=${P37_MAX_NEW:-128}
P25_MAX_TOKENS=${P25_MAX_TOKENS:-"128 256 512"}
STRUCTURED_REQUESTS=${STRUCTURED_REQUESTS:-40}
STRUCTURED_MAX_TOKENS=${STRUCTURED_MAX_TOKENS:-128}
SAFE_DEFAULT_TPS=${SAFE_DEFAULT_TPS:-107.0}
DEFAULT_P25_TPS=${DEFAULT_P25_TPS:-108.0}
AMBER_P25_TPS=${AMBER_P25_TPS:-118.0}
DEFAULT_MARGIN=${DEFAULT_MARGIN:-1.01}
AMBER_MARGIN=${AMBER_MARGIN:-1.05}
RUN_PROFILES=${RUN_PROFILES:-0}
STOP_ON_P37_FAIL=${STOP_ON_P37_FAIL:-0}

mkdir -p "$REPORT_ROOT"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SERVED_NAME="Qwen36-35B-W4A16-${CANDIDATE_NAME}-${STAMP}"
PREFIX="$REPORT_ROOT/r6000_qwen36_w4a16_${CANDIDATE_NAME}_${STAMP}"
P37_OUT="${PREFIX}_p37.json"
P25_OUT="${PREFIX}_p25.json"
STRUCTURED_OUT="${PREFIX}_hard_structured.json"
P26_OUT="${PREFIX}_p26.json"
P28_OUT="${PREFIX}_p28.json"
SUMMARY_OUT="${PREFIX}_promotion_summary.json"
SERVER_LOG="${PREFIX}_server.log"
HEALTH_OUT="${PREFIX}_health.json"

P37_BASELINE_ARGS=(
  --baseline LYNN_FULL_ATTN_ROPE_CACHE=1
  --baseline LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
)
P37_CANDIDATE_ARGS=(
  --candidate LYNN_FULL_ATTN_ROPE_CACHE=1
  --candidate LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
)

apply_safe_default_env() {
  export LYNN_PREFILL_WARMUP=1
  export LYNN_MOE_IMPL=packed_nvfp4
  export LYNN_MOE_FAST_FIXED=1
  export LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
  export LYNN_NATIVE_DOWN_BACKEND=triton
  export LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton
  export LYNN_PACKED_DECODE=0
  export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
  export LYNN_PACKED_SHARED_EXPERT=0
  export LYNN_NATIVE_FP4_LM_HEAD=1
  export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
  export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
  export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
  export LYNN_LINEAR_ATTN_GQA_RECURRENT=1
  export LYNN_LINEAR_ATTN_CONV_BACKEND=triton_torch_silu
  export LYNN_LINEAR_BLOCK_GRAPH=1
  export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
  export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
  export LYNN_LINEAR_STATE_UPDATE=inplace
  export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
  export LYNN_FULL_ATTN_ROPE_CACHE=1
  export LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
  export LYNN_RMSNORM_GATED_BACKEND=triton
  export LYNN_DECODE_FAST_DISPATCH=1
  export LYNN_SHARED_EXPERT_GATE_BACKEND=torch
}

candidate_pairs() {
  if [[ -n "$CANDIDATE_ENV_FILE" ]]; then
    "$PY" - "$CANDIDATE_ENV_FILE" <<'PY'
import pathlib
import shlex
import sys

path = pathlib.Path(sys.argv[1])
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export "):
        line = line[len("export "):].strip()
    for item in shlex.split(line):
        if "=" in item:
            print(item)
PY
  fi
  if [[ -n "$CANDIDATE_ENV" ]]; then
    "$PY" - "$CANDIDATE_ENV" <<'PY'
import shlex
import sys

for item in shlex.split(sys.argv[1]):
    if "=" in item:
        print(item)
PY
  fi
}

mapfile -t CANDIDATE_PAIRS < <(candidate_pairs)
for pair in "${CANDIDATE_PAIRS[@]}"; do
  P37_CANDIDATE_ARGS+=(--candidate "$pair")
done

apply_candidate_env() {
  local pair key value
  for pair in "${CANDIDATE_PAIRS[@]}"; do
    key=${pair%%=*}
    value=${pair#*=}
    export "$key=$value"
  done
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

echo "[promotion-gate] candidate=${CANDIDATE_NAME} stamp=${STAMP}"
printf '[promotion-gate] candidate env:'
printf ' %s' "${CANDIDATE_PAIRS[@]}"
printf '\n'

set +e
"$PY" benchmarks/p37_moe_config_generate_gate.py \
  --model "$MODEL" \
  --out "$P37_OUT" \
  --max-new "$P37_MAX_NEW" \
  "${P37_BASELINE_ARGS[@]}" \
  "${P37_CANDIDATE_ARGS[@]}"
P37_RC=$?
set -e

if [[ "$P37_RC" -ne 0 && "$STOP_ON_P37_FAIL" == "1" ]]; then
  "$PY" - "$SUMMARY_OUT" "$P37_OUT" "$P37_RC" "$CANDIDATE_NAME" "$STAMP" "$SAFE_DEFAULT_TPS" "$DEFAULT_MARGIN" "$AMBER_MARGIN" "$DEFAULT_P25_TPS" "$AMBER_P25_TPS" "$P25_OUT" "$STRUCTURED_OUT" "$P26_OUT" "$P28_OUT" "${CANDIDATE_PAIRS[@]}" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
p37 = pathlib.Path(sys.argv[2])
safe_default = float(sys.argv[6])
default_margin = float(sys.argv[7])
amber_margin = float(sys.argv[8])
default_p25_raw = sys.argv[9]
amber_p25_raw = sys.argv[10]
default_threshold = float(default_p25_raw) if default_p25_raw else safe_default * default_margin
amber_threshold = float(amber_p25_raw) if amber_p25_raw else safe_default * amber_margin
report = {
    "schema_version": "lynn-qwen36-candidate-promotion-summary-v1",
    "candidate_name": sys.argv[4],
    "stamp": sys.argv[5],
    "candidate_env": sys.argv[15:],
    "thresholds": {
        "safe_default_tps": safe_default,
        "default_threshold": default_threshold,
        "amber_threshold": amber_threshold,
        "default_requires_structured": "40/40",
        "amber_requires_structured": "70/70",
    },
    "p37": json.loads(p37.read_text(encoding="utf-8")) if p37.exists() else None,
    "p37_returncode": int(sys.argv[3]),
    "p25": None,
    "hard_structured": None,
    "metrics": {
        "p37_exact": False,
        "p25_512_decode_tps": None,
        "hard_structured_ok": None,
        "hard_structured_decode_tps_mean": None,
        "structured_prompt_file": None,
        "structured_prompt_count": None,
        "structured_request_count": None,
        "structured_pass_count": None,
        "structured_pass_rate": None,
        "structured_pass_rate_40": None,
        "structured_pass_rate_70": None,
        "structured_40_exact": False,
        "structured_70_exact": False,
    },
    "promote_default": False,
    "promote_amber": False,
    "decision": "CLOSED: P37 failed and STOP_ON_P37_FAIL=1.",
}
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out)
print(json.dumps({"decision": report["decision"]}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

apply_safe_default_env
apply_candidate_env

"$PY" -m server.openai_http \
  --model "$MODEL" \
  --served-name "$SERVED_NAME" \
  --host "$HOST" \
  --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 300); do
  if curl -fsS "http://${HOST}:${PORT}/health" > "$HEALTH_OUT" 2>/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[promotion-gate] server exited before ready"
    tail -80 "$SERVER_LOG" || true
    exit 1
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "[promotion-gate] server not ready"
  tail -80 "$SERVER_LOG" || true
  exit 1
fi

set +e
"$PY" benchmarks/p25_server_decode_tps_probe.py \
  --url "http://${HOST}:${PORT}/v1" \
  --model "$SERVED_NAME" \
  --max-tokens $P25_MAX_TOKENS \
  --runs 1 \
  --out "$P25_OUT"
P25_RC=$?

"$PY" scripts/openai_structured_tps_gate.py \
  --base-url "http://${HOST}:${PORT}" \
  --model "$SERVED_NAME" \
  --prompt-specs-file "$PROMPT_SPECS" \
  --requests "$STRUCTURED_REQUESTS" \
  --max-tokens "$STRUCTURED_MAX_TOKENS" \
  --target-decode-tps "$SAFE_DEFAULT_TPS" \
  --out "$STRUCTURED_OUT"
STRUCTURED_RC=$?
set -e

if [[ "$RUN_PROFILES" == "1" ]]; then
  # P26/P28 instantiate their own resident runner. Stop the OpenAI server first
  # so the profiler does not OOM while a 90GB resident model is still live.
  stop_server
  sleep 5
  "$PY" benchmarks/p26_decode_phase_profile.py --model "$MODEL" --out "$P26_OUT" || true
  "$PY" benchmarks/p28_hybrid_block_timing_profile.py --model "$MODEL" --out "$P28_OUT" || true
fi

"$PY" - "$SUMMARY_OUT" "$P37_OUT" "$P37_RC" "$CANDIDATE_NAME" "$STAMP" "$SAFE_DEFAULT_TPS" "$DEFAULT_MARGIN" "$AMBER_MARGIN" "$DEFAULT_P25_TPS" "$AMBER_P25_TPS" "$P25_OUT" "$STRUCTURED_OUT" "$P25_RC" "$STRUCTURED_RC" "$P26_OUT" "$P28_OUT" "${CANDIDATE_PAIRS[@]}" <<'PY'
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
p37_path = pathlib.Path(sys.argv[2])
p37_rc = int(sys.argv[3])
candidate_name = sys.argv[4]
stamp = sys.argv[5]
safe_default = float(sys.argv[6])
default_margin = float(sys.argv[7])
amber_margin = float(sys.argv[8])
default_p25_raw = sys.argv[9]
amber_p25_raw = sys.argv[10]
p25_path = pathlib.Path(sys.argv[11])
structured_path = pathlib.Path(sys.argv[12])
p25_rc = int(sys.argv[13])
structured_rc = int(sys.argv[14])
p26_path = pathlib.Path(sys.argv[15])
p28_path = pathlib.Path(sys.argv[16])
candidate_env = sys.argv[17:]

def load(path: pathlib.Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

p37 = load(p37_path)
p25 = load(p25_path)
structured = load(structured_path)
p26 = load(p26_path)
p28 = load(p28_path)

p37_exact = bool(p37 and p37.get("new_ids_all_match"))
p37_speedup = p37.get("median_speedup") if p37 else None
p25_512 = None
if p25:
    p25_512 = ((p25.get("summary_by_max_tokens") or {}).get("512") or {}).get("decode_tps", {}).get("mean")
hard_ok = bool(structured and structured.get("summary", {}).get("all_format_ok") and structured.get("summary", {}).get("all_finish_stop"))
hard_decode = structured.get("summary", {}).get("decode_tps_mean") if structured else None
structured_summary = structured.get("summary", {}) if structured else {}
structured_prompt_file = structured.get("prompt_specs_file") if structured else None
structured_prompt_count = structured.get("prompt_spec_count") if structured else None
structured_request_count = structured_summary.get("request_count") if structured_summary else None
structured_pass_count = structured_summary.get("format_ok") if structured_summary else None
structured_pass_rate = (
    (float(structured_pass_count) / float(structured_request_count))
    if structured_pass_count is not None and structured_request_count
    else None
)
structured_pass_rate_40 = structured_pass_rate if structured_request_count == 40 else None
structured_pass_rate_70 = structured_pass_rate if structured_request_count == 70 else None
structured_40_exact = bool(structured_request_count == 40 and structured_pass_count == 40 and hard_ok)
structured_70_exact = bool(structured_request_count == 70 and structured_pass_count == 70 and hard_ok)

default_threshold = float(default_p25_raw) if default_p25_raw else safe_default * default_margin
amber_threshold = float(amber_p25_raw) if amber_p25_raw else safe_default * amber_margin
promote_default = bool(p37_exact and structured_40_exact and p25_512 is not None and p25_512 >= default_threshold)
promote_amber = bool(structured_70_exact and p25_512 is not None and p25_512 >= amber_threshold)

if promote_default:
    decision = "DEFAULT_CANDIDATE: exact-greedy, 40/40 hard structured, and P25 threshold passed."
elif promote_amber:
    decision = "AMBER_CANDIDATE: 70/70 hard structured and P25 threshold passed, but exact-greedy default gate did not pass."
elif not hard_ok:
    decision = "CLOSED: hard structured gate failed."
elif p25_512 is None or p25_512 < safe_default:
    decision = "CLOSED: P25 512 decode TPS is below safe default."
elif p25_512 >= amber_threshold and not structured_70_exact:
    decision = "RESEARCH_ONLY: AMBER speed may pass, but 70/70 hard structured gate is missing or failed."
elif not p37_exact:
    decision = "RESEARCH_ONLY: speed/format may pass, but exact-greedy drift blocks default."
else:
    decision = "RESEARCH_ONLY: below promotion margin."

report = {
    "schema_version": "lynn-qwen36-candidate-promotion-summary-v1",
    "candidate_name": candidate_name,
    "stamp": stamp,
    "candidate_env": candidate_env,
    "thresholds": {
        "safe_default_tps": safe_default,
        "default_threshold": default_threshold,
        "amber_threshold": amber_threshold,
        "default_requires_structured": "40/40",
        "amber_requires_structured": "70/70",
    },
    "returncodes": {
        "p37": p37_rc,
        "p25": p25_rc,
        "hard_structured": structured_rc,
    },
    "metrics": {
        "p37_exact": p37_exact,
        "p37_median_speedup": p37_speedup,
        "p25_512_decode_tps": p25_512,
        "hard_structured_ok": hard_ok,
        "hard_structured_decode_tps_mean": hard_decode,
        "structured_prompt_file": structured_prompt_file,
        "structured_prompt_count": structured_prompt_count,
        "structured_request_count": structured_request_count,
        "structured_pass_count": structured_pass_count,
        "structured_pass_rate": structured_pass_rate,
        "structured_pass_rate_40": structured_pass_rate_40,
        "structured_pass_rate_70": structured_pass_rate_70,
        "structured_40_exact": structured_40_exact,
        "structured_70_exact": structured_70_exact,
    },
    "reports": {
        "p37": str(p37_path),
        "p25": str(p25_path),
        "hard_structured": str(structured_path),
        "p26": str(p26_path) if p26 else None,
        "p28": str(p28_path) if p28 else None,
    },
    "promote_default": promote_default,
    "promote_amber": promote_amber,
    "decision": decision,
}
summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(summary_path)
print(json.dumps(report["metrics"] | {"decision": decision, "promote_default": promote_default, "promote_amber": promote_amber}, ensure_ascii=False, indent=2))
PY

echo "[promotion-gate] summary ${SUMMARY_OUT}"
