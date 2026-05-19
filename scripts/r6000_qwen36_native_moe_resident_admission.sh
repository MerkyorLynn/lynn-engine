#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# R6000: Qwen3.6-35B Native MoE Resident Admission Gate
#
# Sequentially gates a Native MoE candidate from fixture AMBER to resident:
#
#   Stage 1: P136 fixture contract (slot repack strict numeric)
#   Stage 2: P139/P141 packed contract (NVFP4 dequant exact-match)
#   Stage 3: P37 graph-on exact (3 prompts × 8 tokens greedy identical)
#   Stage 4: P25 decode TPS (128/256/512 single-stream)
#   Stage 5: Structured 40/70 (parse correctness + content)
#
# FAIL-LOUD: P37 not exact → CLOSED immediately, P25/structured not run.
#
# Usage:
#   # Dry-run (print all commands, no execution):
#   DRY_RUN=1 bash scripts/r6000_qwen36_native_moe_resident_admission.sh
#
#   # With candidate env file:
#   CANDIDATE_NAME=v31_graphsafe \
#   CANDIDATE_ENV_FILE=scripts/qwen36_candidate_env_moe_repack_scratch.env \
#   DRY_RUN=0 bash scripts/r6000_qwen36_native_moe_resident_admission.sh
#
#   # Override P37 behavior:
#   STOP_ON_P37_FAIL=1 DRY_RUN=0 bash scripts/r6000_qwen36_native_moe_resident_admission.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REPO="${REPO:-/root/autodl-tmp/lynn-engine}"
PY="${PY:-/root/autodl-tmp/conda-envs/r6000-eval/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0}"
REPORT_ROOT="${REPORT_ROOT:-/root/autodl-tmp/reports/qwen36_35b}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-1}"

# Candidate identity
CANDIDATE_NAME="${CANDIDATE_NAME:-native_moe_unnamed}"
CANDIDATE_ENV="${CANDIDATE_ENV:-}"
CANDIDATE_ENV_FILE="${CANDIDATE_ENV_FILE:-}"

# Server (for P25/structured stages)
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18169}"
BASE_URL="http://${HOST}:${PORT}/v1"
SERVED_MODEL="${SERVED_MODEL:-Qwen36-35B-W4A16}"

# P37 config
P37_MAX_NEW="${P37_MAX_NEW:-128}"
P37_PROMPTS="${P37_PROMPTS:-3}"

# P25 config
P25_MAX_TOKENS="${P25_MAX_TOKENS:-128 256 512}"

# Structured config
STRUCTURED_PROMPTS="${STRUCTURED_PROMPTS:-${REPO_ROOT}/scripts/qwen36_structured_hard_prompts_70.json}"
STRUCTURED_40="${STRUCTURED_40:-40}"
STRUCTURED_70="${STRUCTURED_70:-70}"

# Fixture paths
FIXTURE_DIR="${FIXTURE_DIR:-${REPORT_ROOT}/p135_repacked_fixtures}"

# Stop policy
STOP_ON_P37_FAIL="${STOP_ON_P37_FAIL:-1}"

# Probes
P136_PROBE="${REPO_ROOT}/benchmarks/p136_moe_slot_repack_contract.py"
P139_PROBE="${REPO_ROOT}/benchmarks/p139_moe_slot_packed_contract.py"
P37_PROBE="${REPO_ROOT}/benchmarks/p37_moe_config_generate_gate.py"
P25_PROBE="${REPO_ROOT}/benchmarks/p25_server_decode_tps_probe.py"

# Output
OUT_DIR="${REPORT_ROOT}/native_moe_admission_${CANDIDATE_NAME}_${STAMP}"
SUMMARY_JSON="${OUT_DIR}/admission_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Resolve candidate env
# ─────────────────────────────────────────────────────────────────────────────
RESOLVED_ENV=""
if [[ -n "$CANDIDATE_ENV_FILE" ]]; then
  if [[ ! -f "$CANDIDATE_ENV_FILE" ]]; then
    echo "[admission] ERROR: CANDIDATE_ENV_FILE not found: $CANDIDATE_ENV_FILE" >&2
    exit 1
  fi
  # Read env file, skip comments/blanks
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line// /}"
    [[ -z "$line" ]] && continue
    RESOLVED_ENV="${RESOLVED_ENV:+$RESOLVED_ENV }$line"
  done < "$CANDIDATE_ENV_FILE"
elif [[ -n "$CANDIDATE_ENV" ]]; then
  RESOLVED_ENV="$CANDIDATE_ENV"
fi

# Build candidate overrides as --candidate KEY=VALUE args for P37
P37_CANDIDATE_ARGS=()
if [[ -n "$RESOLVED_ENV" ]]; then
  for pair in $RESOLVED_ENV; do
    P37_CANDIDATE_ARGS+=(--candidate "$pair")
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Dependency checks
# ─────────────────────────────────────────────────────────────────────────────
MISSING=()
for probe in "$P136_PROBE" "$P139_PROBE" "$P37_PROBE" "$P25_PROBE"; do
  if [[ ! -f "$probe" ]]; then
    MISSING+=("$probe")
  fi
done
if [[ ! -f "$STRUCTURED_PROMPTS" ]]; then
  MISSING+=("$STRUCTURED_PROMPTS")
fi

if [[ ${#MISSING[@]} -gt 0 && "$DRY_RUN" != "1" ]]; then
  echo "[admission] ERROR: Missing dependencies:" >&2
  for m in "${MISSING[@]}"; do
    echo "  - $m" >&2
  done
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│  Native MoE Resident Admission Gate                              │"
echo "└──────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Candidate:   $CANDIDATE_NAME"
echo "  Env:         ${RESOLVED_ENV:-<default>}"
echo "  Model:       $MODEL"
echo "  DRY_RUN:     $DRY_RUN"
echo "  Output:      $OUT_DIR"
echo "  Stop on P37: $STOP_ON_P37_FAIL"
echo ""
echo "  Admission ladder:"
echo "    1. P136 fixture contract (strict numeric)"
echo "    2. P139/P141 packed contract"
echo "    3. P37 graph-on exact (${P37_PROMPTS}p × ${P37_MAX_NEW}t)"
echo "    4. P25 decode TPS (${P25_MAX_TOKENS})"
echo "    5. Structured ${STRUCTURED_40}/${STRUCTURED_70}"
echo ""

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "  [WARN] Missing (OK for DRY_RUN):"
  for m in "${MISSING[@]}"; do echo "    - $m"; done
  echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# Execution helpers
# ─────────────────────────────────────────────────────────────────────────────
run_or_print() {
  local label="$1"; shift
  echo "━━━ $label ━━━"
  echo "  CMD: $*"
  echo ""
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY_RUN] skipped"
    echo ""
    return 0
  fi
  "$@"
  local rc=$?
  echo ""
  return $rc
}

STAGE_RESULTS=()
record_stage() {
  local stage="$1" status="$2" detail="${3:-}"
  STAGE_RESULTS+=("$stage:$status:$detail")
  echo "  [$status] $stage ${detail:+— $detail}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: P136 fixture contract
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STAGE 1: P136 Fixture Contract (slot repack strict)"
echo "═══════════════════════════════════════════════════════════════════"

P136_OUT="${OUT_DIR}/p136_fixture.json"
if [[ "$DRY_RUN" == "1" ]]; then
  run_or_print "P136" \
    "$PY" "$P136_PROBE" \
      --fixtures "$FIXTURE_DIR" \
      --model "$MODEL" \
      --out "$P136_OUT"
  record_stage "p136_fixture" "DRY_RUN"
else
  mkdir -p "$OUT_DIR"
  if "$PY" "$P136_PROBE" --fixtures "$FIXTURE_DIR" --model "$MODEL" --out "$P136_OUT"; then
    record_stage "p136_fixture" "PASS"
  else
    record_stage "p136_fixture" "FAIL" "fixture contract failed"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: P139/P141 packed contract
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STAGE 2: P139 Packed NVFP4 Contract"
echo "═══════════════════════════════════════════════════════════════════"

P139_OUT="${OUT_DIR}/p139_packed.json"
if [[ "$DRY_RUN" == "1" ]]; then
  run_or_print "P139" \
    "$PY" "$P139_PROBE" \
      --fixtures "$FIXTURE_DIR" \
      --model "$MODEL" \
      --out "$P139_OUT"
  record_stage "p139_packed" "DRY_RUN"
else
  if "$PY" "$P139_PROBE" --fixtures "$FIXTURE_DIR" --model "$MODEL" --out "$P139_OUT"; then
    record_stage "p139_packed" "PASS"
  else
    record_stage "p139_packed" "FAIL" "packed contract failed"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: P37 graph-on exact (FAIL-LOUD gate)
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STAGE 3: P37 Graph-On Exact Match (FAIL-LOUD)"
echo "═══════════════════════════════════════════════════════════════════"

P37_OUT="${OUT_DIR}/p37_exact.json"
P37_EXACT="unknown"

if [[ "$DRY_RUN" == "1" ]]; then
  run_or_print "P37-exact" \
    "$PY" "$P37_PROBE" \
      --model "$MODEL" \
      --out "$P37_OUT" \
      --max-new "$P37_MAX_NEW" \
      "${P37_CANDIDATE_ARGS[@]}"
  record_stage "p37_exact" "DRY_RUN"
else
  if "$PY" "$P37_PROBE" --model "$MODEL" --out "$P37_OUT" --max-new "$P37_MAX_NEW" "${P37_CANDIDATE_ARGS[@]}"; then
    # Check if all_exact is true
    P37_EXACT="$("$PY" -c "
import json, sys
d = json.loads(open('$P37_OUT').read())
exact = d.get('all_exact', d.get('exact', False))
print('true' if exact else 'false')
")"
    if [[ "$P37_EXACT" == "true" ]]; then
      record_stage "p37_exact" "PASS" "all prompts exact"
    else
      record_stage "p37_exact" "CLOSED" "NOT exact — candidate rejected"
      if [[ "$STOP_ON_P37_FAIL" == "1" ]]; then
        echo ""
        echo "  [CLOSED] P37 not exact. STOP_ON_P37_FAIL=1 → skipping P25/structured."
        echo "  Candidate $CANDIDATE_NAME is NOT ready for resident promotion."
        # Write summary and exit
        _write_summary "CLOSED" "P37 not exact"
        exit 2
      fi
    fi
  else
    record_stage "p37_exact" "ERROR" "P37 probe crashed"
    P37_EXACT="error"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: P25 decode TPS
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STAGE 4: P25 Decode TPS (${P25_MAX_TOKENS})"
echo "═══════════════════════════════════════════════════════════════════"

P25_OUT="${OUT_DIR}/p25_decode_tps.json"
if [[ "$DRY_RUN" == "1" ]]; then
  run_or_print "P25-512" \
    "$PY" "$P25_PROBE" \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL" \
      --max-new 512 \
      --warmup 4 \
      --repeat 16 \
      --out "$P25_OUT"
  record_stage "p25_tps" "DRY_RUN"
else
  if "$PY" "$P25_PROBE" --base-url "$BASE_URL" --model "$SERVED_MODEL" --max-new 512 --warmup 4 --repeat 16 --out "$P25_OUT"; then
    record_stage "p25_tps" "PASS"
  else
    record_stage "p25_tps" "FAIL" "P25 probe failed"
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Structured 40/70
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  STAGE 5: Structured Gate (40 + 70 prompts)"
echo "═══════════════════════════════════════════════════════════════════"

STRUCT_40_OUT="${OUT_DIR}/structured_40.json"
STRUCT_70_OUT="${OUT_DIR}/structured_70.json"
if [[ "$DRY_RUN" == "1" ]]; then
  run_or_print "Structured-40" \
    "$PY" "$REPO_ROOT/scripts/openai_mcq_thinking32_eval.py" \
      --task gpqa \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL" \
      --out "$STRUCT_40_OUT" \
      --limit "$STRUCTURED_40" \
      --max-tokens 128 \
      --concurrency 2 \
      --timeout 120
  run_or_print "Structured-70" \
    "$PY" "$REPO_ROOT/scripts/openai_mcq_thinking32_eval.py" \
      --task gpqa \
      --base-url "$BASE_URL" \
      --model "$SERVED_MODEL" \
      --out "$STRUCT_70_OUT" \
      --limit "$STRUCTURED_70" \
      --max-tokens 128 \
      --concurrency 2 \
      --timeout 120
  record_stage "structured_40" "DRY_RUN"
  record_stage "structured_70" "DRY_RUN"
else
  "$PY" "$REPO_ROOT/scripts/openai_mcq_thinking32_eval.py" \
    --task gpqa --base-url "$BASE_URL" --model "$SERVED_MODEL" \
    --out "$STRUCT_40_OUT" --limit "$STRUCTURED_40" --max-tokens 128 --concurrency 2 --timeout 120 \
    && record_stage "structured_40" "PASS" \
    || record_stage "structured_40" "FAIL"
  "$PY" "$REPO_ROOT/scripts/openai_mcq_thinking32_eval.py" \
    --task gpqa --base-url "$BASE_URL" --model "$SERVED_MODEL" \
    --out "$STRUCT_70_OUT" --limit "$STRUCTURED_70" --max-tokens 128 --concurrency 2 --timeout 120 \
    && record_stage "structured_70" "PASS" \
    || record_stage "structured_70" "FAIL"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Write summary JSON
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════════"

if [[ "$DRY_RUN" == "1" ]]; then
  echo ""
  echo "  [DRY_RUN] All stages printed. Set DRY_RUN=0 to execute."
  echo "  Summary JSON would be: $SUMMARY_JSON"
  echo ""
  exit 0
fi

# Determine overall decision
DECISION="UNKNOWN"
REASON=""
# Check stages
has_fail=false
has_closed=false
for entry in "${STAGE_RESULTS[@]}"; do
  IFS=':' read -r stage status detail <<< "$entry"
  if [[ "$status" == "CLOSED" ]]; then has_closed=true; REASON="$stage: $detail"; fi
  if [[ "$status" == "FAIL" || "$status" == "ERROR" ]]; then has_fail=true; REASON="${REASON:-$stage: $detail}"; fi
done

if [[ "$has_closed" == "true" ]]; then
  DECISION="CLOSED"
elif [[ "$has_fail" == "true" ]]; then
  DECISION="BLOCKED"
else
  DECISION="CANDIDATE_READY"
  REASON="all stages passed"
fi

# Extract P25 TPS if available
P25_512_TPS="null"
if [[ -f "$P25_OUT" ]]; then
  P25_512_TPS="$("$PY" -c "
import json
d = json.loads(open('$P25_OUT').read())
tps = d.get('decode_tps_median') or d.get('median_tps') or d.get('tps_512')
print(tps if tps else 'null')
" 2>/dev/null || echo "null")"
fi

# Write JSON
"$PY" - "$SUMMARY_JSON" "$CANDIDATE_NAME" "$DECISION" "$REASON" "$P37_EXACT" "$P25_512_TPS" "$STAMP" <<'PYSUMMARY'
import json, sys
from pathlib import Path

out_path, candidate_name, decision, reason, p37_exact, p25_tps, stamp = sys.argv[1:]

summary = {
    "schema": "lynn-native-moe-resident-admission-v1",
    "created": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    "stamp": stamp,
    "candidate_name": candidate_name,
    "fixture_status": "see p136/p139 outputs",
    "p37_exact": p37_exact == "true",
    "p25_512_decode_tps": float(p25_tps) if p25_tps != "null" else None,
    "structured_40": None,
    "structured_70": None,
    "decision": decision,
    "reason": reason,
}

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PYSUMMARY

echo ""
echo "  Decision:  $DECISION"
echo "  Reason:    $REASON"
echo "  Summary:   $SUMMARY_JSON"
echo ""
