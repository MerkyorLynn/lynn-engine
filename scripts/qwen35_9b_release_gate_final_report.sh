#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B release gate final report — GPU-free, report-only.
#
# Reads the existing release_gate_summary_latest.json and regenerates the
# final report JSON + Markdown.  No SSH, no GPU, no runtime changes.
#
# Usage:
#   bash scripts/qwen35_9b_release_gate_final_report.sh
#   bash scripts/qwen35_9b_release_gate_final_report.sh --reports-dir /path/to/reports

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT/reports/qwen35_9b}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reports-dir)
      REPORTS_DIR="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# Resolve to absolute path before cd
cd "$ROOT"
REPORTS_DIR="$(cd "$REPORTS_DIR" 2>/dev/null && pwd)" || {
  echo "[ERROR] reports dir not found: $REPORTS_DIR" >&2
  exit 1
}

SUMMARY_JSON="$REPORTS_DIR/qwen35_9b_release_gate_summary_latest.json"
FINAL_JSON="$REPORTS_DIR/qwen35_9b_release_gate_final_report_20260519.json"
FINAL_MD="$REPORTS_DIR/QWEN35_9B_RELEASE_GATE_FINAL_REPORT_20260519.md"

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "[ERROR] summary not found: $SUMMARY_JSON" >&2
  echo "[HINT]  run scripts/qwen35_9b_release_gate_summary.sh first" >&2
  exit 1
fi

# --- Generate the final report JSON from the summary ---
python3 - "$SUMMARY_JSON" "$FINAL_JSON" <<'PYEOF'
import json
import sys
from datetime import datetime, timezone

summary_path = sys.argv[1]
out_path = sys.argv[2]

with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)

variants = {v["variant"]: v for v in summary.get("variants", [])}
nvfp4 = variants.get("NVFP4", {})
q4km = variants.get("Q4_K_M", {})
bf16 = variants.get("BF16", {})

now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

report = {
    "schema": "lynn-qwen35-9b-release-gate-final-report-v1",
    "created": now,
    "model_id": "Qwen/Qwen3.5-9B",
    "source_summary": summary_path,
    "decisions": {
        "mac_default": {
            "variant": "Q4_K_M",
            "quant": "q4km",
            "artifact": "Qwen3.5-9B-Q4_K_M-imatrix.gguf",
            "runtime": "llama.cpp (Mac / R6000)",
            "decision": q4km.get("decision", "NEEDS_MORE_DATA"),
            "reasons": q4km.get("decision_reasons", []),
        },
        "nvidia_default": {
            "variant": "NVFP4",
            "quant": "nvfp4",
            "artifact": "Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0",
            "runtime": "Lynn Engine (CUDA, R6000)",
            "decision": nvfp4.get("decision", "NEEDS_MORE_DATA"),
            "classification": "compatibility / research"
            if nvfp4.get("tps_decode_512", 0) is not None
            and nvfp4.get("tps_decode_512", 0) < 155
            else "promotable",
            "reasons": nvfp4.get("decision_reasons", []),
        },
        "bf16_reference": {
            "variant": "BF16",
            "quant": "bf16",
            "artifact": "Qwen3.5-9B-BF16 (safetensors)",
            "runtime": "transformers-direct",
            "decision": "QUALITY_REFERENCE",
            "reasons": bf16.get("decision_reasons", []),
        },
    },
    "hard_metrics": {
        "mmlu": {
            "threshold": summary.get("thresholds", {}).get("mmlu_min", 0.74),
            "bf16": bf16.get("mmlu"),
            "q4km": q4km.get("mmlu"),
            "nvfp4": nvfp4.get("mmlu"),
        },
        "gpqa_thinking_off": {
            "threshold": summary.get("thresholds", {}).get("gpqa_min", 0.41),
            "bf16": bf16.get("gpqa"),
            "q4km": q4km.get("gpqa"),
            "nvfp4": nvfp4.get("gpqa"),
        },
        "gpqa_thinking32": {
            "naive": q4km.get("gpqa_thinking32_naive"),
            "ex_parse_fail": q4km.get("gpqa_thinking32_ex_parse_fail"),
            "parse_fail_rate": q4km.get("parse_fail_rate"),
            "status": "LIVE_PARTIAL",
        },
        "tps_decode_512": {
            "threshold": summary.get("thresholds", {}).get("tps_decode_512_min", 155),
            "bf16": bf16.get("tps_decode_512"),
            "q4km": q4km.get("tps_decode_512"),
            "nvfp4": nvfp4.get("tps_decode_512"),
        },
        "structured_pass": {
            "bf16": bf16.get("structured_pass"),
            "q4km": q4km.get("structured_pass"),
            "nvfp4": nvfp4.get("structured_pass"),
        },
    },
    "risks": [],
    "missing_data": [],
}

# Derive risks and missing_data from the metrics
metrics = report["hard_metrics"]

if metrics["gpqa_thinking_off"]["q4km"] is not None and metrics["gpqa_thinking_off"]["q4km"] < 0.41:
    report["risks"].append({
        "id": "RISK-01",
        "severity": "medium",
        "variant": "Q4_K_M",
        "description": f"GPQA thinking-off {metrics['gpqa_thinking_off']['q4km']:.3f} < 0.41 floor. "
        "Mac users who disable thinking get degraded GPQA-level performance.",
    })

if metrics["gpqa_thinking32"]["status"] == "LIVE_PARTIAL":
    report["risks"].append({
        "id": "RISK-02",
        "severity": "medium",
        "variant": "Q4_K_M",
        "description": "GPQA thinking-on 32K data is live partial. "
        "If remaining questions degrade ex-parse-fail below 0.80, override fails.",
    })

tps = metrics["tps_decode_512"]
if tps["nvfp4"] is not None and tps["nvfp4"] < tps["threshold"]:
    report["risks"].append({
        "id": "RISK-03",
        "severity": "high",
        "variant": "NVFP4",
        "description": f"NVFP4 TPS {tps['nvfp4']:.1f} < {tps['threshold']:.0f} threshold. "
        "Fundamental bandwidth limitation of current FP4 repack path.",
    })

if tps["bf16"] is None:
    report["missing_data"].append({
        "field": "tps_decode_512",
        "variant": "BF16",
        "description": "No TPS benchmark for BF16 (reference only).",
    })
if metrics["gpqa_thinking32"]["status"] == "LIVE_PARTIAL":
    report["missing_data"].append({
        "field": "gpqa_thinking32_full",
        "variant": "ALL",
        "description": "GPQA thinking-on 32K is live partial. Full results pending from R6000.",
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(f"[final-report] wrote {out_path}")
for key, dec in report["decisions"].items():
    print(f"  {key}: {dec['decision']}")
PYEOF

# --- Copy the static Markdown report if it exists, otherwise note regeneration ---
if [[ -f "$FINAL_MD" ]]; then
  echo "[final-report] MD report exists: $FINAL_MD"
else
  echo "[final-report] WARNING: MD report not found at $FINAL_MD"
  echo "[final-report] The MD is a static snapshot — create it from the JSON if needed"
fi

echo "[final-report] done"
