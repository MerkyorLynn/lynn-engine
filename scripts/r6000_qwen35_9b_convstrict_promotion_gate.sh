#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.5-9B W4A16 NVFP4 Convstrict Promotion Gate
# ─────────────────────────────────────────────────────────────────────────────
#
# Serial gate: P183 (isolation) → P184 (70/70 exact) → P150 (service TPS)
# Outputs summary JSON with pass/fail verdict.
#
# Prerequisites:
#   - Model at $MODEL_DIR
#   - Env: scripts/qwen35_9b_candidate_env_convstrict.env
#
# Usage:
#   bash scripts/r6000_qwen35_9b_convstrict_promotion_gate.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/lynn-engine}"
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0}"
REPORT_DIR="${REPO_DIR}/reports/qwen35_9b"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTHON="${PYTHON:-/root/miniconda3/bin/python3.12}"

echo "═══════════════════════════════════════════════════════════════"
echo " Qwen3.5-9B NVFP4 Convstrict Promotion Gate"
echo " Model: ${MODEL_DIR}"
echo " Time:  ${TIMESTAMP}"
echo "═══════════════════════════════════════════════════════════════"

cd "${REPO_DIR}"

# ── Step 1: P183 Exact-Fast Isolation ──
echo ""
echo "── Step 1: P183 Exact-Fast Isolation ──"
P183_OUT="${REPORT_DIR}/p183_qwen35_9b_nvfp4_exact_fast_isolation_${TIMESTAMP}.json"
${PYTHON} benchmarks/p183_qwen35_9b_nvfp4_exact_fast_isolation.py \
    --model "${MODEL_DIR}" \
    --out "${P183_OUT}" || true
echo "  P183 output: ${P183_OUT}"

# ── Step 2: P184 Convstrict Exact Gate (70 hard prompts) ──
echo ""
echo "── Step 2: P184 Exact Gate (70/70 hard structured) ──"
P184_OUT="${REPORT_DIR}/p184_qwen35_9b_nvfp4_convstrict_exact_gate_${TIMESTAMP}.json"
${PYTHON} benchmarks/p184_qwen35_9b_nvfp4_convstrict_exact_gate.py \
    --model "${MODEL_DIR}" \
    --prompts-json scripts/qwen36_structured_hard_prompts_70.json \
    --limit 70 \
    --max-new 64 \
    --out "${P184_OUT}" || true
echo "  P184 output: ${P184_OUT}"

# ── Step 3: P150 Service TPS (128/256/512) ──
echo ""
echo "── Step 3: P150 Service TPS (requires server running) ──"
P150_OUT="${REPORT_DIR}/p150_qwen35_9b_nvfp4_linear_graph_summary_${TIMESTAMP}_convstrict.json"
if command -v curl &>/dev/null && curl -s http://127.0.0.1:18191/v1/models >/dev/null 2>&1; then
    ${PYTHON} benchmarks/openai_http_smoke.py \
        --url http://127.0.0.1:18191/v1 \
        --max-tokens 128,256,512 \
        --out-summary "${P150_OUT}" || true
    echo "  P150 output: ${P150_OUT}"
else
    echo "  SKIP: server not running on :18191"
    P150_OUT=""
fi

# ── Step 4: Summary Verdict ──
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PROMOTION VERDICT"
echo "═══════════════════════════════════════════════════════════════"

SUMMARY_OUT="${REPORT_DIR}/qwen35_9b_convstrict_promotion_summary_${TIMESTAMP}.json"
${PYTHON} -c "
import json, os, sys
from pathlib import Path

p184_path = '${P184_OUT}'
p150_path = '${P150_OUT}' if '${P150_OUT}' else None

# P184 check
p184_exact = False
p184_count = 0
if os.path.exists(p184_path):
    with open(p184_path) as f:
        p184 = json.load(f)
    p184_exact = p184.get('comparison', {}).get('all_exact', False)
    p184_count = p184.get('comparison', {}).get('exact_count', 0)

# P150 check
p150_tps_512 = 0.0
p150_ready = False
if p150_path and os.path.exists(p150_path):
    with open(p150_path) as f:
        p150 = json.load(f)
    p150_tps_512 = p150.get('decode_tps', {}).get('512', 0.0)
    p150_ready = p150.get('verdict') == 'P25_READY'

# Verdict
if p184_exact and p150_tps_512 >= 62.0:
    verdict = 'DEFAULT_PROMOTABLE'
elif p184_exact and p150_tps_512 >= 58.0:
    verdict = 'AMBER_FAST_EXACT'
elif not p184_exact:
    verdict = 'BLOCKED_EXACT_DRIFT'
else:
    verdict = 'BLOCKED_TPS_LOW'

summary = {
    'candidate': 'qwen35_9b_convstrict',
    'env_file': 'scripts/qwen35_9b_candidate_env_convstrict.env',
    'p184_exact_70_70': p184_exact,
    'p184_exact_count': p184_count,
    'p150_decode_tps_512': p150_tps_512,
    'p150_ready': p150_ready,
    'verdict': verdict,
    'promotion_criteria': {
        'P184_exact_70_70': p184_exact,
        'P150_512_tps_ge_62': p150_tps_512 >= 62.0,
    },
}
Path('${SUMMARY_OUT}').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
"

echo ""
echo "  Summary: ${SUMMARY_OUT}"
echo "═══════════════════════════════════════════════════════════════"
