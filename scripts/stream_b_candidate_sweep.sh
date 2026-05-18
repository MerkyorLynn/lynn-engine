#!/usr/bin/env bash
# Stream B candidate sweep helper.
#
# Walks scripts/qwen36_candidate_env_*.env, lists each as a sweep entry
# with the recommended ``r6000_qwen36_candidate_promotion_gate.sh``
# invocation, and (optionally) reads a directory of finished gate JSON
# outputs to apply ``stream_b_promotion_report_card.py`` and aggregate
# DEFAULT / AMBER / closed counts.
#
# This is **planning + accounting** only — it does not actually launch
# any R6000 gate run (that is Codex's lane and uses real GPU). The
# helper exists so promotion discipline is consistent across operators.
#
# Usage:
#
#   # Print the candidate sweep plan
#   scripts/stream_b_candidate_sweep.sh
#
#   # Same, plus aggregate any finished gate JSONs from a directory
#   scripts/stream_b_candidate_sweep.sh --reports-dir reports/promotion-gates
#
#   # Override safe-default reference
#   scripts/stream_b_candidate_sweep.sh --safe-default-tps 107
#
# Promotion bar (2026-05-18 hand-off):
#   DEFAULT: P37 3/3 + structured 40/40 + P25 512 >= 108 TPS
#   AMBER:   P37 drift OK + structured 70/70 + P25 512 >= 118 TPS (opt-in)
#   sprint:  118 TPS
#   122:     A + B stacked only

set -euo pipefail

REPORTS_DIR=""
SAFE_DEFAULT_TPS=""
SHOW_CARDS=0

while (( $# )); do
  case "$1" in
    --reports-dir) REPORTS_DIR="$2"; shift 2 ;;
    --safe-default-tps) SAFE_DEFAULT_TPS="$2"; shift 2 ;;
    --cards) SHOW_CARDS=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "[sweep] unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Stream B candidate sweep plan ==="
echo "Promotion bar (fixed):"
echo "  DEFAULT: P37 3/3 + structured 40/40 + P25 512 >= 108 TPS"
echo "  AMBER:   P37 drift OK + structured 70/70 + P25 512 >= 118 TPS (opt-in)"
echo "  sprint:  118 TPS"
echo "  122:     A + B stacked through full ladder, never microbench-only"
echo

echo "## Candidate env files in scripts/"
shopt -s nullglob
candidates=(scripts/qwen36_candidate_env_*.env)
shopt -u nullglob

if (( ${#candidates[@]} == 0 )); then
  echo "  (none found)"
  echo
else
  for f in "${candidates[@]}"; do
    name="$(basename "$f" .env)"
    name="${name#qwen36_candidate_env_}"
    summary="$(grep '^#' "$f" | head -6 | sed 's/^# *//' )"
    echo "* candidate=\"${name}\""
    echo "    file=${f}"
    if [[ -n "$summary" ]]; then
      echo "    header:"
      while IFS= read -r line; do echo "      ${line}"; done <<< "$summary"
    fi
    echo "    run:"
    echo "      CANDIDATE_NAME=${name} \\"
    echo "      CANDIDATE_ENV_FILE=${f} \\"
    echo "      scripts/r6000_qwen36_candidate_promotion_gate.sh"
    echo
  done
fi

if [[ -n "$REPORTS_DIR" ]]; then
  if [[ ! -d "$REPORTS_DIR" ]]; then
    echo "[sweep] reports-dir not found: $REPORTS_DIR" >&2
    exit 3
  fi
  echo "## Finished gate reports under ${REPORTS_DIR}"
  shopt -s nullglob
  gate_jsons=("$REPORTS_DIR"/*.json)
  shopt -u nullglob
  if (( ${#gate_jsons[@]} == 0 )); then
    echo "  (no JSON gates yet)"
  else
    default_count=0
    amber_count=0
    closed_count=0
    research_count=0
    for j in "${gate_jsons[@]}"; do
      base="$(basename "$j" .json)"
      case "$base" in
        *__p25|*__p37|*__structured|*__p26) continue ;;
      esac
      args=("--gate-json" "$j")
      [[ -n "$SAFE_DEFAULT_TPS" ]] && args+=("--safe-default-tps" "$SAFE_DEFAULT_TPS")
      if (( SHOW_CARDS == 1 )); then
        python3 scripts/stream_b_promotion_report_card.py "${args[@]}" || true
        echo "---"
      else
        decision="$(python3 scripts/stream_b_promotion_report_card.py "${args[@]}" 2>/dev/null \
                    | grep '^## Decision:' | head -1 | sed 's/^## Decision: *//')"
        echo "* ${base}: ${decision}"
        case "$decision" in
          *DEFAULT_promote*) default_count=$((default_count + 1)) ;;
          *AMBER_promote*|*AMBER_only*) amber_count=$((amber_count + 1)) ;;
          *closed*) closed_count=$((closed_count + 1)) ;;
          *research_artifact_only*) research_count=$((research_count + 1)) ;;
        esac
      fi
    done
    echo
    echo "## Aggregate"
    echo "  DEFAULT_promote:        ${default_count}"
    echo "  AMBER_promote/_only:    ${amber_count}"
    echo "  closed:                 ${closed_count}"
    echo "  research_artifact_only: ${research_count}"
  fi
fi

echo
echo "## Discipline reminder"
echo "Per 2026-05-18 hand-off, every candidate result must carry P37 exact +"
echo "structured pass + P25 512 decode TPS together. Microbench-only numbers"
echo "are research artifacts; they never reach DEFAULT or AMBER promote."
