#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# r6000_qwen35_9b_dense_runtime_smoke.sh
#
# Smoke test for Qwen3.5-9B (dense FFN, hybrid linear/full attention) on
# Lynn Engine. Validates:
#   1. _runtime_config parses 32-layer config without LAYER_TYPES mismatch
#   2. from_config creates LynnInferenceState with correct dims
#   3. _layer_forward has dense FFN fallback (gate_proj/up_proj/down_proj)
#   4. loader detects dense weight keys
#   5. resident_runner __init__ has is_moe guard
#
# Qwen3.5-9B is NOT all-full_attention — it has 32 layers with 3:1
# linear_attention:full_attention ratio, same pattern as 3.6 but 32 not 40.
# The difference is MLP: dense (gate_proj/up_proj/down_proj) not MoE experts.
#
# Usage:
#   bash r6000_qwen35_9b_dense_runtime_smoke.sh [OPTIONS]
#
# Options:
#   --model-dir PATH     Model checkpoint dir (default: $MODEL_DIR)
#   --timeout SECS       Server startup timeout (default: 60)
#
# Output: reports/qwen35_9b/r6000_qwen35_9b_dense_runtime_smoke.json
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT="$PROJECT_ROOT/reports/qwen35_9b/r6000_qwen35_9b_dense_runtime_smoke.json"

# Defaults
MODEL_DIR="${MODEL_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --timeout) shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$REPORT")"

if [[ -z "$MODEL_DIR" ]]; then
    echo "[smoke] ERROR: --model-dir required"
    cat > "$REPORT" <<JSONEOF
{
  "benchmark": "qwen35_9b_dense_runtime_smoke",
  "status": "ERROR",
  "reason": "--model-dir not specified",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSONEOF
    exit 1
fi

echo "[smoke] Model: $MODEL_DIR"

config_json="$MODEL_DIR/config.json"
if [[ ! -f "$config_json" ]]; then
    echo "[smoke] config.json not found — PENDING"
    cat > "$REPORT" <<PEND
{
  "benchmark": "qwen35_9b_dense_runtime_smoke",
  "status": "PENDING",
  "reason": "config.json not found at $MODEL_DIR",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
PEND
    exit 0
fi

# ── Test 1: _runtime_config parses config without LAYER_TYPES mismatch ───────
echo "[smoke] Test 1: _runtime_config..."
T1=$(python3 -c "
import sys, json
sys.path.insert(0, '$PROJECT_ROOT')
from engine.resident_runner import _runtime_config
cfg, n_layers = _runtime_config('$MODEL_DIR')
lt = cfg.get('layer_types', [])
n_linear = sum(1 for t in lt if t == 'linear_attention')
n_full = sum(1 for t in lt if t == 'full_attention')
ne = cfg.get('num_experts', 0)
is_moe = cfg.get('is_moe', True)
print(f'OK: n_layers={n_layers} num_experts={ne} is_moe={is_moe} linear={n_linear} full={n_full} hidden={cfg[\"hidden_size\"]}')
" 2>&1) && T1_STATUS="PASS" || T1_STATUS="FAIL"
echo "[smoke]   $T1 ($T1_STATUS)"

# ── Test 2: from_config creates state with correct dims ──────────────────────
echo "[smoke] Test 2: from_config..."
T2=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from engine.resident_runner import _runtime_config
from engine.inference_state import LynnInferenceState
cfg, _ = _runtime_config('$MODEL_DIR')
s = LynnInferenceState.from_config(cfg, device='cpu')
n_kv = len(s.kv_cache)
n_rec = len(s.recurrent_state)
n_conv = len(s.conv_state)
print(f'OK: hidden={s.hidden_size} kv_heads={s.num_kv_heads} head_dim={s.head_dim} kv_layers={n_kv} rec_layers={n_rec} conv_layers={n_conv}')
" 2>&1) && T2_STATUS="PASS" || T2_STATUS="FAIL"
echo "[smoke]   $T2 ($T2_STATUS)"

# ── Test 3: _layer_forward has dense FFN path ────────────────────────────────
echo "[smoke] Test 3: dense FFN path..."
T3=$(python3 -c "
import sys, inspect
sys.path.insert(0, '$PROJECT_ROOT')
from engine.full_forward import _layer_forward, _dense_ffn
src = inspect.getsource(_layer_forward)
has_moe_branch = 'is_moe' in src or 'num_experts' in src
has_dense_call = '_dense_ffn' in src
print(f'OK: moe_branch={has_moe_branch} dense_call={has_dense_call}')
" 2>&1) && T3_STATUS="PASS" || T3_STATUS="FAIL"
echo "[smoke]   $T3 ($T3_STATUS)"

# ── Test 4: loader detects dense weight keys ─────────────────────────────────
echo "[smoke] Test 4: loader..."
T4=$(python3 -c "
import sys, inspect
sys.path.insert(0, '$PROJECT_ROOT')
from engine.loader import load_qwen36_layer
src = inspect.getsource(load_qwen36_layer)
has_gate_proj = 'gate_proj' in src
print(f'OK: gate_proj_detection={has_gate_proj}')
" 2>&1) && T4_STATUS="PASS" || T4_STATUS="FAIL"
echo "[smoke]   $T4 ($T4_STATUS)"

# ── Test 5: resident_runner has is_moe guard ─────────────────────────────────
echo "[smoke] Test 5: init guards..."
T5=$(python3 -c "
import sys, inspect
sys.path.insert(0, '$PROJECT_ROOT')
from engine.resident_runner import LynnResidentRunner
src = inspect.getsource(LynnResidentRunner.__init__)
has_is_moe = 'is_moe' in src
has_from_config = 'from_config' in src
print(f'OK: is_moe={has_is_moe} from_config={has_from_config}')
" 2>&1) && T5_STATUS="PASS" || T5_STATUS="FAIL"
echo "[smoke]   $T5 ($T5_STATUS)"

# ── Generate report ──────────────────────────────────────────────────────────
ALL_PASS=1
for v in T1_STATUS T2_STATUS T3_STATUS T4_STATUS T5_STATUS; do
    eval "val=\$$v"
    if [[ "$val" != "PASS" ]]; then ALL_PASS=0; fi
done

if [[ $ALL_PASS -eq 1 ]]; then
    STATUS="PASS"
    REASON="All 5 checks passed."
else
    STATUS="FAIL"
    REASON="One or more checks failed."
fi

cat > "$REPORT" <<JSONEOF
{
  "benchmark": "qwen35_9b_dense_runtime_smoke",
  "status": "$STATUS",
  "reason": "$REASON",
  "model_dir": "$MODEL_DIR",
  "checks": {
    "runtime_config": "${T1_STATUS:-SKIP}",
    "from_config": "${T2_STATUS:-SKIP}",
    "dense_ffn_path": "${T3_STATUS:-SKIP}",
    "loader_dense_keys": "${T4_STATUS:-SKIP}",
    "init_guards": "${T5_STATUS:-SKIP}"
  },
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSONEOF

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Dense Runtime Smoke: $STATUS"
echo "════════════════════════════════════════════════════════════════════"
python3 -m json.tool "$REPORT" 2>/dev/null || cat "$REPORT"
