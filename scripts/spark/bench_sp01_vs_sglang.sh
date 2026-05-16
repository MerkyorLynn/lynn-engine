#!/bin/bash
# Spark sm_121 — bench Lynn 27B NVFP4 with SP-01 Triton autotune against the
# SGLang FP8+MTP baseline (49.97 mean / 62.51 peak TPS).
#
# This script:
#   1. Stops any running Lynn container
#   2. Starts a fresh Lynn server with LYNN_SP_TRITON_AUTOTUNE=1
#   3. Waits for /health = ok
#   4. Runs the kernel-level microbench (parity + isolated speedup)
#   5. Runs the 20-prompt mixed + 3-run single-stream TPS bench (SGLang-matched)
#   6. Writes a JSON report comparable with the SGLang baseline output
#
# Run on Spark inside the project root (where scripts/spark/* lives).
# Usage:   bash scripts/spark/bench_sp01_vs_sglang.sh [model_dir]

set -euo pipefail

MODEL=${1:-/models/lynn-27b-variable-recovery-step5000-nvfp4-final}
PORT=18099
TS=$(date +%Y%m%d_%H%M)
REPORT_DIR=reports/sp01_autotune
mkdir -p "${REPORT_DIR}"

ENGINE_VENV=${LYNN_ENGINE_VENV:-python3}

echo "[bench] target model: ${MODEL}"
echo "[bench] report dir:   ${REPORT_DIR}"
echo "[bench] timestamp:    ${TS}"
echo

# --- 1. Microbench (kernel parity + isolated speedup) ---
echo "[bench] step 1/3 — microbench kernel parity + isolated speedup"
${ENGINE_VENV} benchmarks/sp01_sm121_autotune_microbench.py \
    --model "${MODEL}" \
    --layer 6 --iters 1000 \
    --out "${REPORT_DIR}/sp01_microbench_${TS}.json"

MICRO_OK=$(python3 -c "import json,sys; r=json.load(open('${REPORT_DIR}/sp01_microbench_${TS}.json')); sys.exit(0 if r['promotion_gate']['cosine_min_ok'] else 1)" && echo "yes" || echo "no")
if [ "${MICRO_OK}" = "no" ]; then
    echo "[bench] WARN microbench cosine gate failed — autotune output drifts from static." >&2
    echo "[bench] not aborting; continuing to TPS bench so we can see whether the drift" >&2
    echo "[bench] actually shows up at greedy-token level." >&2
fi

# --- 2. Start Lynn server with SP-01 ON ---
echo
echo "[bench] step 2/3 — starting Lynn 27B NVFP4 server with SP-01 autotune"
LYNN_SP_TRITON_AUTOTUNE=1 bash scripts/spark/run_27b_nvfp4_server.sh

# Sanity check via /health
HEALTH=$(curl -sf -m 5 "http://127.0.0.1:${PORT}/health" || echo "FAIL")
if [ "${HEALTH}" = "FAIL" ] || ! echo "${HEALTH}" | grep -q -iE "ok|ready"; then
    echo "[bench] FAIL: server /health did not return OK" >&2
    echo "[bench] last health: ${HEALTH}" >&2
    exit 2
fi
echo "[bench] server READY"

# --- 3. TPS bench (SGLang-matched harness) ---
echo
echo "[bench] step 3/3 — TPS bench (3-run single + 20-prompt mixed)"
${ENGINE_VENV} benchmarks/lynn_27b_vs_35b.py \
    --target lynn-27b-sp01 \
    --runs-single 3 --tokens-single 300 \
    --runs-mixed 20 --tokens-mixed 200 \
    --endpoint "http://127.0.0.1:${PORT}" \
    --out "${REPORT_DIR}/sp01_tps_${TS}.json" || {
        echo "[bench] WARN: bench harness returned non-zero" >&2
    }

echo
echo "[bench] DONE"
echo "[bench] microbench: ${REPORT_DIR}/sp01_microbench_${TS}.json"
echo "[bench] tps:        ${REPORT_DIR}/sp01_tps_${TS}.json"
echo
echo "[bench] SGLang FP8+MTP baseline on Spark sm_121:"
echo "[bench]   3-run single mean    43.44 / median 46.39 / peak 47.30"
echo "[bench]   20-prompt mixed mean 49.97 / max 62.51 / stddev 6.22"
echo
echo "[bench] Compare against the JSON above."
echo "[bench] To revert to no-SP-01 production baseline:"
echo "[bench]   bash scripts/spark/run_27b_nvfp4_server.sh"
