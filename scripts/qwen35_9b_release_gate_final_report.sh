#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-9B release gate final report v2 — dynamic generator.
#
# Reads release_gate_summary_latest.json, overlays the latest p201 thinking32
# data, and writes both the JSON report and the Markdown report.
# No static snapshots — everything is generated at run time.
#
# Usage:
#   bash scripts/qwen35_9b_release_gate_final_report.sh
#   bash scripts/qwen35_9b_release_gate_final_report.sh --reports-dir /path

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPORTS_DIR_DEFAULT="$ROOT/reports/qwen35_9b"
REPORTS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reports-dir) REPORTS_DIR="$2"; shift 2 ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$ROOT"
REPORTS_DIR="${REPORTS_DIR:-$REPORTS_DIR_DEFAULT}"
# resolve to absolute
REPORTS_DIR="$(cd "$REPORTS_DIR" 2>/dev/null && pwd)" || {
  echo "[ERROR] reports dir not found: $REPORTS_DIR" >&2; exit 1
}

SUMMARY="$REPORTS_DIR/qwen35_9b_release_gate_summary_latest.json"
if [[ ! -f "$SUMMARY" ]]; then
  echo "[ERROR] summary not found: $SUMMARY" >&2
  echo "[HINT]  run scripts/qwen35_9b_release_gate_summary.sh first" >&2
  exit 1
fi

OUT_JSON="$REPORTS_DIR/qwen35_9b_release_gate_final_report_20260519.json"
OUT_MD="$REPORTS_DIR/QWEN35_9B_RELEASE_GATE_FINAL_REPORT_20260519.md"

# --- find latest p201 thinking32 JSON by mtime, filename tiebreaker ---
P201_LATEST=""
for f in $(ls -t "$REPORTS_DIR"/p201_gpqa_live_summary*.json 2>/dev/null); do
  [[ -f "$f" ]] || continue
  P201_LATEST="$f"
  break  # ls -t already sorted newest-first
done

python3 - "$SUMMARY" "$P201_LATEST" "$OUT_JSON" "$OUT_MD" <<'PYEOF'
import json
import sys
import os
from datetime import datetime, timezone

summary_path = sys.argv[1]
p201_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
out_json = sys.argv[3]
out_md = sys.argv[4]

with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)

# --- load p201 thinking32 if available ---
t32 = {"naive": None, "ex_parse_fail": None, "parse_fail_rate": None,
       "progress": None, "n": None, "total": None, "status": "MISSING"}
if p201_path and os.path.isfile(p201_path):
    with open(p201_path, encoding="utf-8") as f:
        p201 = json.load(f)
    overall = p201.get("overall", {})
    progress_str = p201.get("progress", "")
    parts = progress_str.split("/") if "/" in progress_str else []
    n = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else overall.get("n")
    total = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 198
    t32 = {
        "naive": overall.get("accuracy"),
        "ex_parse_fail": overall.get("accuracy_excluding_parse_fail") or overall.get("excl_parse_fail"),
        "parse_fail_rate": overall.get("parse_fail_rate"),
        "progress": progress_str,
        "n": n,
        "total": total,
        "source_file": os.path.basename(p201_path),
        "status": "FINAL" if n is not None and total is not None and n >= total else "LIVE_PARTIAL",
    }

# --- override thinking32 in summary variants ---
for v in summary.get("variants", []):
    v["gpqa_thinking32_naive"] = t32["naive"]
    v["gpqa_thinking32_ex_parse_fail"] = t32["ex_parse_fail"]
    v["parse_fail_rate"] = t32["parse_fail_rate"]

# --- build decisions ---
variants = {v["variant"]: v for v in summary.get("variants", [])}
nvfp4 = variants.get("NVFP4", {})
q4km = variants.get("Q4_K_M", {})
bf16 = variants.get("BF16", {})
thresholds = summary.get("thresholds", {})

now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

# Mac decision
mac_decision = q4km.get("decision", "NEEDS_MORE_DATA")
if mac_decision == "PROMOTE_READY" and t32["status"] == "LIVE_PARTIAL":
    mac_decision = "PROMOTE_READY_FOR_MAC_CONDITIONAL"
elif mac_decision == "PROMOTE_READY":
    mac_decision = "PROMOTE_READY_FOR_MAC"

# NVIDIA decision
nvfp4_tps = nvfp4.get("tps_decode_512")
nvfp4_quality_ok = (nvfp4.get("mmlu") or 0) >= thresholds.get("mmlu_min", 0.74) \
    and (nvfp4.get("gpqa") or 0) >= thresholds.get("gpqa_min", 0.41) \
    and nvfp4.get("structured_pass") is True
if nvfp4_quality_ok and nvfp4_tps is not None and nvfp4_tps < thresholds.get("tps_decode_512_min", 155):
    nvidia_decision = "QUALITY_OK_TPS_BLOCKED"
elif nvfp4_quality_ok:
    nvidia_decision = nvfp4.get("decision", "NEEDS_MORE_DATA")
else:
    nvidia_decision = nvfp4.get("decision", "CLOSED")

# --- build risks ---
risks = []
q4km_gpqa = q4km.get("gpqa")
if q4km_gpqa is not None and q4km_gpqa < thresholds.get("gpqa_min", 0.41):
    risks.append({
        "id": "RISK-01", "severity": "medium", "variant": "Q4_K_M",
        "description": f"GPQA thinking-off {q4km_gpqa:.3f} < {thresholds.get('gpqa_min', 0.41)} floor. "
            "Mac users who disable thinking get degraded GPQA-level performance.",
    })
if t32["status"] == "LIVE_PARTIAL":
    risks.append({
        "id": "RISK-02", "severity": "medium", "variant": "Q4_K_M",
        "description": f"GPQA thinking-on 32K is {t32['progress']} — live partial. "
            "If remaining questions degrade ex-parse-fail below the gate, override fails.",
    })
if nvfp4_tps is not None and nvfp4_tps < thresholds.get("tps_decode_512_min", 155):
    risks.append({
        "id": "RISK-03", "severity": "high", "variant": "NVFP4",
        "description": f"NVFP4 TPS {nvfp4_tps:.1f} < {thresholds.get('tps_decode_512_min', 155):.0f} threshold. "
            "Fundamental bandwidth limitation of current FP4 repack path.",
    })

# --- missing_data ---
missing = []
if bf16.get("tps_decode_512") is None:
    missing.append({"field": "tps_decode_512", "variant": "BF16",
                     "description": "No TPS benchmark for BF16 (reference only)."})
if t32["status"] == "LIVE_PARTIAL":
    missing.append({"field": "gpqa_thinking32_full", "variant": "ALL",
                     "description": f"GPQA thinking-on 32K is {t32['progress']} — full results pending from R6000."})
if nvfp4_tps is None:
    missing.append({"field": "tps_decode_512", "variant": "NVFP4",
                     "description": "No TPS benchmark for NVFP4."})

# --- write JSON ---
report = {
    "schema": "lynn-qwen35-9b-release-gate-final-report-v2",
    "created": now,
    "model_id": "Qwen/Qwen3.5-9B",
    "source_summary": os.path.basename(summary_path),
    "thinking32_status": t32,
    "decisions": {
        "mac_default": {
            "variant": "Q4_K_M", "quant": "q4km",
            "artifact": "Qwen3.5-9B-Q4_K_M-imatrix.gguf",
            "runtime": "llama.cpp (Mac / R6000)",
            "decision": mac_decision,
            "reasons": q4km.get("decision_reasons", []),
        },
        "nvidia_default": {
            "variant": "NVFP4", "quant": "nvfp4",
            "artifact": "Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0",
            "runtime": "Lynn Engine (CUDA, R6000)",
            "decision": nvidia_decision,
            "classification": "compatibility / research" if nvfp4_tps is not None
                and nvfp4_tps < thresholds.get("tps_decode_512_min", 155) else "promotable",
            "reasons": nvfp4.get("decision_reasons", []),
        },
        "bf16_reference": {
            "variant": "BF16", "quant": "bf16",
            "artifact": "Qwen3.5-9B-BF16 (safetensors)",
            "runtime": "transformers-direct",
            "decision": "QUALITY_REFERENCE",
            "reasons": bf16.get("decision_reasons", []),
        },
    },
    "hard_metrics": {
        "mmlu": {"threshold": thresholds.get("mmlu_min", 0.74),
                 "bf16": bf16.get("mmlu"), "q4km": q4km.get("mmlu"), "nvfp4": nvfp4.get("mmlu")},
        "gpqa_thinking_off": {"threshold": thresholds.get("gpqa_min", 0.41),
                              "bf16": bf16.get("gpqa"), "q4km": q4km.get("gpqa"), "nvfp4": nvfp4.get("gpqa")},
        "gpqa_thinking32": t32,
        "tps_decode_512": {"threshold": thresholds.get("tps_decode_512_min", 155),
                           "bf16": bf16.get("tps_decode_512"), "q4km": q4km.get("tps_decode_512"),
                           "nvfp4": nvfp4.get("tps_decode_512")},
        "structured_pass": {"bf16": bf16.get("structured_pass"), "q4km": q4km.get("structured_pass"),
                            "nvfp4": nvfp4.get("structured_pass")},
    },
    "risks": risks,
    "missing_data": missing,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")

# --- helper: format number ---
def fmt(val, decimals=4, suffix=""):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"

def fmt_pct(val):
    if val is None:
        return "—"
    return f"{val*100:.2f}%"

# --- write MD ---
t32_label = "**LIVE PARTIAL**" if t32["status"] == "LIVE_PARTIAL" else "**FINAL**"
t32_progress = t32.get("progress") or "N/A"

md = []
md.append("# Qwen3.5-9B Release Gate — Final Report V2")
md.append("")
md.append(f"Generated: {now}")
md.append("")
md.append("## Executive Summary")
md.append("")
md.append("| Platform | Variant | Decision | Blocker |")
md.append("|---|---|---|---|")
md.append(f"| **Mac** | Q4_K_M imatrix | **{mac_decision}** | "
           + ("thinking32 live partial — conditional" if "CONDITIONAL" in mac_decision else "None"))
md.append(f"| **NVIDIA** | NVFP4 W4A16 | **{nvidia_decision}** | "
           + (f"TPS {fmt(nvfp4_tps, 1)} < {thresholds.get('tps_decode_512_min', 155):.0f} tok/s"
              if nvfp4_tps is not None and nvfp4_tps < thresholds.get("tps_decode_512_min", 155) else "—"))
md.append("| **Reference** | BF16 | QUALITY_REFERENCE | N/A |")
md.append("")

md.append("## Hard Metrics")
md.append("")
md.append("### Quality Benchmarks")
md.append("")
md.append("| Metric | Threshold | BF16 | Q4_K_M | NVFP4 | Status |")
md.append("|---|---:|---:|---:|---:|---|")

mmlu = report["hard_metrics"]["mmlu"]
mmlu_vals = [mmlu["bf16"], mmlu["q4km"], mmlu["nvfp4"]]
if mmlu["threshold"] and all(v is not None and v >= mmlu["threshold"] for v in mmlu_vals):
    mmlu_status = "✅ ALL PASS"
else:
    mmlu_status = "⚠️ CHECK"
md.append(f"| MMLU (500, 5-shot) | ≥ {mmlu['threshold']} | {fmt(mmlu['bf16'])} | {fmt(mmlu['q4km'])} | {fmt(mmlu['nvfp4'])} | {mmlu_status} |")

gpqa = report["hard_metrics"]["gpqa_thinking_off"]
gpqa_q4km_status = "⚠️ below floor (thinking32 override)" if gpqa["q4km"] is not None and gpqa["q4km"] < gpqa["threshold"] else "✅"
md.append(f"| GPQA thinking-off | ≥ {gpqa['threshold']} | {fmt(gpqa['bf16'])} | {fmt(gpqa['q4km'])} | {fmt(gpqa['nvfp4'])} | {gpqa_q4km_status} |")
md.append("")

md.append(f"### GPQA Thinking-On 32K ({t32_label} — {t32_progress})")
md.append("")
md.append("| Metric | Gate | Value | Status |")
md.append("|---|---:|---:|---|")
t32_ex = t32.get("ex_parse_fail")
t32_pf = t32.get("parse_fail_rate")
t32_ex_gate = thresholds.get("gpqa_thinking32_ex_parse_min", 0.8)
t32_pf_gate = thresholds.get("gpqa_thinking32_parse_fail_max", 0.25)
md.append(f"| Naive accuracy | — | {fmt(t32.get('naive'))} | informational |")
ex_status = "✅ GATE PASS" if t32_ex is not None and t32_ex >= t32_ex_gate else ("❌ BELOW GATE" if t32_ex is not None else "—")
md.append(f"| Ex-parse-fail accuracy | ≥ {t32_ex_gate} | **{fmt(t32_ex)}** | {ex_status} |")
pf_status = "✅ within tolerance" if t32_pf is not None and t32_pf <= t32_pf_gate else ("❌ ABOVE LIMIT" if t32_pf is not None else "—")
md.append(f"| Parse-fail rate | ≤ {t32_pf_gate} | {fmt(t32_pf)} | {pf_status} |")
md.append(f"| Progress | 198 | {t32_progress} | {'🔄 R6000 still running' if t32['status'] == 'LIVE_PARTIAL' else '✅ complete'} |")
md.append("")

md.append("### Speed Benchmarks")
md.append("")
md.append("| Metric | Threshold | BF16 | Q4_K_M | NVFP4 | Status |")
md.append("|---|---:|---:|---:|---:|---|")
tps = report["hard_metrics"]["tps_decode_512"]
tps_nv_status = "❌ BLOCKED" if tps["nvfp4"] is not None and tps["nvfp4"] < tps["threshold"] else "✅"
tps_q4_status = "✅" if tps["q4km"] is not None and tps["q4km"] >= tps["threshold"] else "❌"
md.append(f"| TPS decode 512 | ≥ {tps['threshold']:.0f} | — | {fmt(tps['q4km'], 1)} | {fmt(tps['nvfp4'], 1)} | Q4KM {tps_q4_status} / NVFP4 {tps_nv_status} |")
md.append("")

md.append("### Structured Content Gate (P196)")
md.append("")
sp = report["hard_metrics"]["structured_pass"]
md.append(f"- BF16: {'PASS' if sp['bf16'] else 'FAIL' if sp['bf16'] is not None else '—'}")
md.append(f"- Q4_K_M: {'PASS' if sp['q4km'] else 'FAIL' if sp['q4km'] is not None else '—'}")
md.append(f"- NVFP4: {'PASS' if sp['nvfp4'] else 'FAIL' if sp['nvfp4'] is not None else '—'}")
md.append(f"- Verdict: W4A8 relative no-regression, absolute AMBER (W4A16 baseline 90%)")
md.append("")

md.append("## Risks")
md.append("")
if risks:
    md.append("| ID | Sev | Variant | Description |")
    md.append("|---|---|---|---|")
    for r in risks:
        md.append(f"| {r['id']} | {r['severity']} | {r['variant']} | {r['description']} |")
else:
    md.append("None identified.")
md.append("")

md.append("## Missing Data")
md.append("")
if missing:
    md.append("| Field | Variant | Description |")
    md.append("|---|---|---|")
    for m in missing:
        md.append(f"| {m['field']} | {m['variant']} | {m['description']} |")
else:
    md.append("None.")
md.append("")

md.append("## Source Reports")
md.append("")
md.append(f"- `{os.path.basename(summary_path)}` — aggregated gate summary")
if t32.get("source_file"):
    md.append(f"- `{t32['source_file']}` — thinking32 GPQA ({t32_label.lower()}, {t32_progress})")
md.append(f"- `P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` — structured content gate")
md.append("")
md.append("---")
md.append("")
md.append("*Generated by `scripts/qwen35_9b_release_gate_final_report.sh`. "
           "Report-only — no GPU, no SSH, no runtime changes.*")
md.append("")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"[final-report-v2] wrote {out_json}")
print(f"[final-report-v2] wrote {out_md}")
print(f"[final-report-v2] thinking32: {t32['status']} ({t32_progress})")
print(f"[final-report-v2] mac: {mac_decision}")
print(f"[final-report-v2] nvidia: {nvidia_decision}")
PYEOF

echo "[final-report-v2] done"
