#!/usr/bin/env bash
set -Eeuo pipefail

# Spark-side closed loop for the Qwen3.5-9B W4A8 / FP8-MMA path.
#
# This intentionally does not stop or modify the live APEX-MTP Brain V2
# fallback service. By default it refuses to run if that service is actively
# processing requests, so TPS numbers are not polluted by a concurrent 35B
# eval/user request.

REPO_DIR="${REPO_DIR:-/home/merkyor/lynn-engine}"
W4A16_MODEL="${W4A16_MODEL:-/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
W4A8_MODEL="${W4A8_MODEL:-/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a8-fp8}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/home/merkyor/reports/qwen35_9b/w4a8_fp8_mma_closed_loop_${STAMP}}"

PYTHON="${PYTHON:-}"
MAX_NEW_VALUES="${MAX_NEW_VALUES:-64 128}"
FORCE_REPACK="${FORCE_REPACK:-0}"
RUN_QUALITY="${RUN_QUALITY:-1}"
ALLOW_BUSY_APEX="${ALLOW_BUSY_APEX:-0}"
APEX_BASE_URL="${APEX_BASE_URL:-http://127.0.0.1:18098}"

choose_python() {
  if [[ -n "$PYTHON" ]]; then
    echo "$PYTHON"
    return
  fi
  for cand in \
    /home/merkyor/comfyui/ComfyUI/.venv/bin/python \
    /home/merkyor/.venv/bin/python \
    python3
  do
    if command -v "$cand" >/dev/null 2>&1 || [[ -x "$cand" ]]; then
      echo "$cand"
      return
    fi
  done
  echo "python3"
}

json_get_requests_processing() {
  "$PY" - "$APEX_BASE_URL" <<'PY' || true
import json
import sys
import urllib.request

base = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(base + "/metrics", timeout=5) as resp:
        text = resp.read().decode("utf-8", "replace")
except Exception:
    print("")
    raise SystemExit(0)

for line in text.splitlines():
    if line.startswith("llamacpp:requests_processing "):
        print(line.split()[-1])
        break
PY
}

preflight_live_apex() {
  echo "[qwen35-w4a8] live APEX health (${APEX_BASE_URL})"
  curl -fsS --max-time 5 "${APEX_BASE_URL}/health" || true
  echo

  local processing
  processing="$(json_get_requests_processing)"
  if [[ -n "$processing" ]]; then
    echo "[qwen35-w4a8] live APEX requests_processing=${processing}"
    if [[ "$ALLOW_BUSY_APEX" != "1" && "$processing" != "0" && "$processing" != "0.0" ]]; then
      echo "[qwen35-w4a8] refusing to run while APEX is busy; set ALLOW_BUSY_APEX=1 to override" >&2
      exit 75
    fi
  else
    echo "[qwen35-w4a8] live APEX metrics unavailable; continuing with caution"
  fi
}

main() {
  PY="$(choose_python)"
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

  mkdir -p "$OUT_ROOT"
  exec > >(tee -a "${OUT_ROOT}/run.log") 2>&1

  echo "[qwen35-w4a8] start $(date -Is)"
  echo "[qwen35-w4a8] repo=${REPO_DIR}"
  echo "[qwen35-w4a8] python=${PY}"
  echo "[qwen35-w4a8] w4a16=${W4A16_MODEL}"
  echo "[qwen35-w4a8] w4a8=${W4A8_MODEL}"
  echo "[qwen35-w4a8] out=${OUT_ROOT}"
  echo "[qwen35-w4a8] max_new=${MAX_NEW_VALUES}"

  cd "$REPO_DIR"

  if [[ ! -d "$W4A16_MODEL" ]]; then
    echo "[qwen35-w4a8] missing W4A16 input model dir: ${W4A16_MODEL}" >&2
    exit 2
  fi

  echo
  echo "========== preflight =========="
  preflight_live_apex
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sort -n || true

  echo
  echo "========== packer self-test =========="
  "$PY" scripts/spark_pack_w4a8_fp8.py self-test

  echo
  echo "========== repack w4a16 nvfp4 -> w4a8 fp8 =========="
  if [[ "$FORCE_REPACK" == "1" || ! -f "${W4A8_MODEL}/lynn_quant_manifest.json" || ! -f "${W4A8_MODEL}/model.safetensors.index.json" ]]; then
    mkdir -p "$(dirname "$W4A8_MODEL")"
    "$PY" scripts/spark_pack_w4a8_fp8.py full-dir \
      --input "$W4A16_MODEL" \
      --output "$W4A8_MODEL" \
      --scale-granularity per_row \
      --verify-cos-threshold 0.999
  else
    echo "[qwen35-w4a8] existing W4A8 FP8 model dir found; set FORCE_REPACK=1 to rebuild"
    "$PY" - "$W4A8_MODEL/repack_summary.json" <<'PY' || true
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if p.exists():
    d = json.loads(p.read_text())
    print(json.dumps({
        "schema_version": d.get("schema_version"),
        "totals": d.get("totals"),
        "cos_vs_bf16": {k: v for k, v in (d.get("cos_vs_bf16") or {}).items() if k != "failures_below_threshold"},
    }, indent=2))
PY
  fi

  echo
  echo "========== e2e tps smoke =========="
  "$PY" scripts/spark_fp8_e2e_tps_smoke.py \
    --w4a16-model "$W4A16_MODEL" \
    --w4a8-fp8-model "$W4A8_MODEL" \
    --out "${OUT_ROOT}/spark_fp8_e2e_tps_smoke.json" \
    --max-new ${MAX_NEW_VALUES}

  if [[ "$RUN_QUALITY" == "1" ]]; then
    echo
    echo "========== quality regression =========="
    "$PY" scripts/spark_w4a8_vs_w4a16_quality_regression.py \
      --model-a "$W4A16_MODEL" \
      --model-b "$W4A8_MODEL" \
      --out-dir "${OUT_ROOT}/quality_regression"
  else
    echo "[qwen35-w4a8] RUN_QUALITY=0, skipping quality regression"
  fi

  echo
  echo "========== final artifacts =========="
  find "$OUT_ROOT" -maxdepth 2 -type f | sort
  echo "[qwen35-w4a8] end $(date -Is)"
}

main "$@"
