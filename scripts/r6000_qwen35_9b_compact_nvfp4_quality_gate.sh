#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# R6000: Qwen3.5-9B Compact NVFP4 Quality Gate
#
# Runs MMLU-500 + GPQA Diamond + structured smoke + 32K context + TPS
# on the compact (embed+lm_head quantized) NVFP4 candidate.
#
# DRY_RUN=1 (default): prints all commands without executing.
# DRY_RUN=0: runs everything sequentially against a running Lynn server.
#
# The compact artifact MUST already exist on disk. This script does NOT
# create the compact model; it only evaluates one.
#
# Promote criteria (relative to stable W4A16 8.25 GiB):
#   - MMLU 500: ≤ 1pp regression (stable=75.2%, floor=74.2%)
#   - GPQA Diamond: ≤ 1pp regression (stable=42.9%, floor=41.9%)
#   - Structured: GREEN (all prompts parse correctly)
#   - 32K context: no crash, non-empty response
#   - TPS: ≥ 90% of stable (stable≈61 TPS, floor≈55 TPS)
#
# Usage:
#   DRY_RUN=1 bash scripts/r6000_qwen35_9b_compact_nvfp4_quality_gate.sh
#   DRY_RUN=0 bash scripts/r6000_qwen35_9b_compact_nvfp4_quality_gate.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR="${MODEL_DIR:-/root/autodl-tmp/models/Qwen3.5-9B-lynn-native-compact-nvfp4-candidate}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen35_9b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"

# Server config (assumes Lynn server already running or will be started)
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-18241}"
SERVER_ROOT="http://${SERVER_HOST}:${SERVER_PORT}"
BASE_URL="${SERVER_ROOT}/v1"
SERVED_MODEL="${SERVED_MODEL:-qwen35-9b-compact-nvfp4}"

# Data paths
MMLU_DATA_DIR="${MMLU_DATA_DIR:-/tmp/datasets/mmlu}"
GPQA_CSV="${GPQA_CSV:-${REPORT_ROOT}/gpqa_diamond.csv}"
STRUCTURED_PROMPTS="${STRUCTURED_PROMPTS:-${REPO_ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"

# Eval scripts (verified to exist in this repo)
MMLU_RUNNER="${MMLU_RUNNER:-${REPO_ROOT}/scripts/openai_mmlu_500_5shot_eval.py}"
GPQA_RUNNER="${GPQA_RUNNER:-${REPO_ROOT}/scripts/openai_gpqa_diamond_eval.py}"
STRUCTURED_RUNNER="${STRUCTURED_RUNNER:-${REPO_ROOT}/scripts/openai_structured_tps_gate.py}"
P25_PROBE="${P25_PROBE:-${REPO_ROOT}/benchmarks/p25_server_decode_tps_probe.py}"

# Thresholds (relative to stable W4A16)
STABLE_MMLU="0.752"
STABLE_GPQA="0.4293"
STABLE_TPS="61.0"
MAX_MMLU_DROP="0.01"
MAX_GPQA_DROP="0.01"
MIN_TPS_RATIO="0.90"
MMLU_FLOOR="$("$PYTHON_BIN" -c "print('%.4f' % (${STABLE_MMLU} - ${MAX_MMLU_DROP}))")"
GPQA_FLOOR="$("$PYTHON_BIN" -c "print('%.4f' % (${STABLE_GPQA} - ${MAX_GPQA_DROP}))")"
TPS_FLOOR="$("$PYTHON_BIN" -c "print('%.2f' % (${STABLE_TPS} * ${MIN_TPS_RATIO}))")"

# Output
OUT_DIR="${REPORT_ROOT}/compact_quality_gate_${STAMP}"
MMLU_OUT="${OUT_DIR}/mmlu_500_5shot.jsonl"
GPQA_OUT="${OUT_DIR}/gpqa_diamond.jsonl"
P25_OUT="${OUT_DIR}/p25_decode_tps.json"
STRUCTURED_OUT="${OUT_DIR}/structured_smoke.jsonl"
LONG_CTX_OUT="${OUT_DIR}/long_context_32k.json"
GATE_SUMMARY="${OUT_DIR}/compact_quality_gate_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────────────────────
MISSING=()

if [[ ! -f "$MMLU_RUNNER" ]]; then
  MISSING+=("MMLU runner: $MMLU_RUNNER")
fi
if [[ ! -f "$GPQA_RUNNER" ]]; then
  MISSING+=("GPQA runner: $GPQA_RUNNER")
fi
if [[ ! -f "$STRUCTURED_RUNNER" ]]; then
  MISSING+=("Structured runner: $STRUCTURED_RUNNER")
fi
if [[ ! -f "$P25_PROBE" ]]; then
  MISSING+=("P25 TPS probe: $P25_PROBE")
fi
if [[ ! -f "$STRUCTURED_PROMPTS" ]]; then
  MISSING+=("Structured prompts: $STRUCTURED_PROMPTS")
fi

# Model dir check (only fail if not DRY_RUN)
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[compact-gate] ERROR: Model directory does not exist: $MODEL_DIR" >&2
    echo "[compact-gate] The compact NVFP4 candidate must be created first." >&2
    exit 1
  fi
  if [[ ! -f "$GPQA_CSV" ]]; then
    MISSING+=("GPQA CSV: $GPQA_CSV")
  fi
  if [[ ! -d "$MMLU_DATA_DIR" ]]; then
    MISSING+=("MMLU dataset: $MMLU_DATA_DIR")
  fi
fi

if [[ ${#MISSING[@]} -gt 0 && "$DRY_RUN" != "1" ]]; then
  echo "[compact-gate] ERROR: Missing dependencies:" >&2
  for m in "${MISSING[@]}"; do
    echo "  - $m" >&2
  done
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  Qwen3.5-9B Compact NVFP4 Quality Gate                          │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Model:        $MODEL_DIR"
echo "  DRY_RUN:      $DRY_RUN"
echo "  Output:       $OUT_DIR"
echo "  Server:       $BASE_URL (model=$SERVED_MODEL)"
echo ""
echo "  Thresholds:"
echo "    MMLU floor:   $MMLU_FLOOR (stable=$STABLE_MMLU)"
echo "    GPQA floor:   $GPQA_FLOOR (stable=$STABLE_GPQA)"
echo "    TPS floor:    $TPS_FLOOR (stable=$STABLE_TPS)"
echo ""

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "  [WARN] Missing dependencies (OK for DRY_RUN):"
  for m in "${MISSING[@]}"; do
    echo "    - $m"
  done
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Command builder
# ─────────────────────────────────────────────────────────────────────────────
run_or_print() {
  local label="$1"
  shift
  echo "━━━ $label ━━━"
  echo "  CMD: $*"
  echo ""
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY_RUN] skipped"
    echo ""
    return 0
  fi
  "$@"
  echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Gate steps
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUT_DIR"
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 1: MMLU 500 5-shot"
echo "═══════════════════════════════════════════════════════════════════"
run_or_print "MMLU-500" \
  "$PYTHON_BIN" "$MMLU_RUNNER" \
    --data-dir "$MMLU_DATA_DIR" \
    --base-url "$BASE_URL" \
    --model "$SERVED_MODEL" \
    --out "$MMLU_OUT" \
    --concurrency 4 \
    --shots 5 \
    --sample 500 \
    --seed 20260519

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 2: GPQA Diamond (198 questions)"
echo "═══════════════════════════════════════════════════════════════════"
run_or_print "GPQA-Diamond" \
  "$PYTHON_BIN" "$GPQA_RUNNER" \
    --csv "$GPQA_CSV" \
    --base-url "$BASE_URL" \
    --model "$SERVED_MODEL" \
    --out "$GPQA_OUT" \
    --concurrency 2

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 3: Structured prompt smoke (70 prompts)"
echo "═══════════════════════════════════════════════════════════════════"
run_or_print "Structured-70" \
  "$PYTHON_BIN" "$STRUCTURED_RUNNER" \
    --base-url "$SERVER_ROOT" \
    --model "$SERVED_MODEL" \
    --out "$STRUCTURED_OUT" \
    --prompt-specs-file "$STRUCTURED_PROMPTS" \
    --requests 70 \
    --max-tokens 96 \
    --target-decode-tps 1 \
    --timeout 240

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 4: 32K long context smoke"
echo "═══════════════════════════════════════════════════════════════════"
if [[ -f "$P25_PROBE" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "━━━ 32K-Context ━━━"
    echo "  CMD: <generate 32K needle prompt> | $PYTHON_BIN $P25_PROBE --url $BASE_URL --model $SERVED_MODEL --prompt <generated> --max-tokens 128 --runs 1 --chat --out $LONG_CTX_OUT"
    echo ""
    echo "  [DRY_RUN] skipped"
    echo ""
  else
    LONG_PROMPT="$("$PYTHON_BIN" - <<'PY'
filler = "The quick brown fox jumps over the lazy dog. " * 730
needle = "The compact NVFP4 release gate secret is PINEAPPLE-7749."
mid = len(filler) // 2
print(filler[:mid] + " " + needle + " " + filler[mid:] + "\n\nReturn only the secret code.")
PY
)"
    run_or_print "32K-Context" \
      "$PYTHON_BIN" "$P25_PROBE" \
        --url "$BASE_URL" \
        --model "$SERVED_MODEL" \
        --prompt "$LONG_PROMPT" \
        --max-tokens 128 \
        --runs 1 \
        --chat \
        --timeout 900 \
        --out "$LONG_CTX_OUT"
  fi
else
  echo "  [SKIP] P25 probe not found at: $P25_PROBE"
  echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  STEP 5: Decode TPS (128/256/512)"
echo "═══════════════════════════════════════════════════════════════════"
if [[ -f "$P25_PROBE" ]]; then
  run_or_print "TPS-512" \
    "$PYTHON_BIN" "$P25_PROBE" \
      --url "$BASE_URL" \
      --model "$SERVED_MODEL" \
      --max-tokens 128 256 512 \
      --runs 3 \
      --chat \
      --timeout 900 \
      --out "$P25_OUT"
else
  echo "  [SKIP] P25 probe not found"
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "  Output dir:  $OUT_DIR"
echo "  DRY_RUN:     $DRY_RUN"
echo ""

if [[ "$DRY_RUN" == "1" ]]; then
  echo "  [DRY_RUN] No actual execution. Set DRY_RUN=0 to run for real."
  echo ""
  echo "  Prerequisites for real execution:"
  echo "    1. Compact NVFP4 model at: $MODEL_DIR"
  echo "    2. Lynn server started on port $SERVER_PORT serving $SERVED_MODEL"
  echo "    3. MMLU dataset at: $MMLU_DATA_DIR"
  echo "    4. GPQA CSV at: $GPQA_CSV"
  echo ""
  echo "  After execution, run the gate judge (manual or script) with thresholds:"
  echo "    MMLU ≥ $MMLU_FLOOR"
  echo "    GPQA ≥ $GPQA_FLOOR"
  echo "    TPS  ≥ $TPS_FLOOR"
fi
