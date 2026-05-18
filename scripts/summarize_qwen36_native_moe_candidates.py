#!/usr/bin/env python3
"""Unified Native MoE candidate summary.

Reads all available candidate reports (graceful on missing) and produces a
single JSON + Markdown table with per-candidate verdict and next-step
recommendation.

Auto-discovers reports by naming convention in --report-dir.

Usage:
  python scripts/summarize_qwen36_native_moe_candidates.py \\
    --report-dir reports/qwen36_35b \\
    --out reports/qwen36_35b/native_moe_candidate_summary.json \\
    --md-out reports/qwen36_35b/native_moe_candidate_summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Thresholds (mirror p140)
# ─────────────────────────────────────────────────────────────
DEFAULT_SLOT_MAX_ABS = 1e-3
DEFAULT_COSINE_MIN = 0.999999
DEFAULT_LATENCY_MS = 0.059

AMBER_SLOT_MAX_ABS = 0.003
AMBER_UNIQUE_MAX_ABS = 0.002
AMBER_LATENCY_MS = 0.055


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _find_latest(rd: Path, pattern: str) -> Path | None:
    matches = sorted(rd.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _agg_cosine_min(data: dict[str, Any]) -> float | None:
    """Extract min cosine from summary or per-result aggregation."""
    for key in ("cosine_min", "min_cosine"):
        if key in data and data[key] is not None:
            return float(data[key])
    if "results" in data:
        for key in ("slot_ref_cosine", "slot_cosine", "cosine",
                     "stored_ref_cosine", "unique_ref_cosine"):
            vals = [r[key] for r in data["results"] if key in r]
            if vals:
                return min(vals)
    return None


def _agg_slot_max_abs(data: dict[str, Any]) -> float | None:
    for key in ("slot_max_abs", "max_max_abs"):
        if key in data and data[key] is not None:
            return float(data[key])
    if "results" in data:
        for key in ("slot_ref_max_abs", "slot_max_abs", "max_abs"):
            vals = [r[key] for r in data["results"] if key in r]
            if vals:
                return max(vals)
    return None


def _agg_unique_max_abs(data: dict[str, Any]) -> float | None:
    if "unique_max_abs" in data and data["unique_max_abs"] is not None:
        return float(data["unique_max_abs"])
    if "results" in data:
        for key in ("unique_ref_max_abs", "unique_max_abs"):
            vals = [r[key] for r in data["results"] if key in r]
            if vals:
                return max(vals)
    return None


def _classify(slot_abs: float | None, cos_min: float | None,
              lat: float | None, uniq_abs: float | None,
              p140_recommend_p37: bool) -> tuple[str, str]:
    """Return (verdict, recommend_next_step)."""
    if slot_abs is None or lat is None:
        return "MISSING", "no data"

    # DEFAULT check
    if (slot_abs <= DEFAULT_SLOT_MAX_ABS
            and (cos_min is not None and cos_min >= DEFAULT_COSINE_MIN)
            and lat <= DEFAULT_LATENCY_MS):
        return "DEFAULT", "default promote"

    # AMBER_FAST check
    if (lat <= AMBER_LATENCY_MS
            and slot_abs <= AMBER_SLOT_MAX_ABS
            and (uniq_abs is None or uniq_abs <= AMBER_UNIQUE_MAX_ABS)):
        if p140_recommend_p37:
            return "AMBER_FAST", "P37 exploratory"
        return "AMBER_FAST", "await P140 gate clearance"

    # Slow but exact
    if slot_abs == 0.0:
        return "EXACT_SLOW", "research artifact — too slow for serving"

    return "CLOSED", "no further action"


# ─────────────────────────────────────────────────────────────
# Candidate registry
# ─────────────────────────────────────────────────────────────

CANDIDATES = [
    {
        "id": "native_slot_output_owned_bf16_fast",
        "label": "native_slot_output_owned_bf16 (fast, default ref)",
        "patterns": [
            "native_slot_output_owned_bf16_report*.json",
        ],
    },
    {
        "id": "native_slot_output_owned_bf16_slotorder",
        "label": "native_slot_output_owned_bf16 (slot-order + route-bf16)",
        "patterns": [
            "native_slot_output_owned_bf16_slotorder*report*.json",
        ],
    },
    {
        "id": "native_slot_output_owned_bf16_dualref",
        "label": "native_slot_output_owned_bf16 (dual-ref)",
        "patterns": [
            "native_slot_output_owned_bf16_dualref*report*.json",
        ],
    },
    {
        "id": "native_slot_strict_bf16",
        "label": "native_slot_strict_bf16 (cuBLAS oracle)",
        "patterns": [
            "native_slot_strict_bf16*slotorder*report*.json",
            "native_slot_strict_bf16*report*.json",
        ],
    },
    {
        "id": "native_slot_tc_bf16",
        "label": "native_slot_tc_bf16 (TensorCore probe)",
        "patterns": [
            "native_slot_tc_bf16*report*.json",
        ],
    },
    {
        "id": "native_slot_fused_bf16",
        "label": "native_slot_fused_bf16 (fused probe)",
        "patterns": [
            "native_slot_fused*report*.json",
        ],
    },
]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def gather(report_dir: Path) -> dict[str, Any]:
    """Gather all candidate data into a unified report."""
    # ── p136 ──
    p136_file = _find_latest(report_dir,
                              "p136_slot_repack_contract_slotorder_report*.json")
    p136_data = _load_json(p136_file) if p136_file else None

    # ── p140 gate ──
    p140_file = report_dir / "p140_native_moe_risk_gate.json"
    p140_data = _load_json(p140_file)
    p140_recommend_p37 = bool(
        p140_data and p140_data.get("recommend_p37_exploratory"))

    # ── p137 (slot-order preferred) ──
    p137_file = _find_latest(report_dir,
                              "p137_moe_slot_stage_diagnostics*slotorder*.json")
    if not p137_file:
        p137_file = _find_latest(
            report_dir, "p137_moe_slot_stage_diagnostics*.json")
    p137_data = _load_json(p137_file) if p137_file else None

    # ── Candidates ──
    candidates: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        report_path = None
        for pat in spec["patterns"]:
            report_path = _find_latest(report_dir, pat)
            if report_path:
                break

        if report_path is None:
            candidates.append({
                "id": spec["id"],
                "label": spec["label"],
                "report": None,
                "status": "MISSING",
                "slot_max_abs": None,
                "unique_max_abs": None,
                "cosine_min": None,
                "avg_latency_ms": None,
                "verdict": "MISSING",
                "recommend_next_step": "no report found",
            })
            continue

        data = _load_json(report_path)
        if data is None:
            candidates.append({
                "id": spec["id"],
                "label": spec["label"],
                "report": str(report_path),
                "status": "UNREADABLE",
                "verdict": "CLOSED",
                "recommend_next_step": "report unreadable",
            })
            continue

        slot_abs = _agg_slot_max_abs(data)
        uniq_abs = _agg_unique_max_abs(data)
        cos_min = _agg_cosine_min(data)
        lat = data.get("avg_latency_ms")

        verdict, next_step = _classify(
            slot_abs, cos_min, lat, uniq_abs, p140_recommend_p37)

        candidates.append({
            "id": spec["id"],
            "label": spec["label"],
            "report": str(report_path),
            "slot_max_abs": slot_abs,
            "unique_max_abs": uniq_abs,
            "cosine_min": cos_min,
            "avg_latency_ms": lat,
            "verdict": verdict,
            "recommend_next_step": next_step,
        })

    # ── Assemble ──
    any_default = any(c["verdict"] == "DEFAULT" for c in candidates)
    any_amber = any(c["verdict"] == "AMBER_FAST" for c in candidates)

    return {
        "schema": "lynn-native-moe-candidate-summary-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_dir": str(report_dir),
        "p136": {
            "report": str(p136_file) if p136_file else None,
            "verdict": p136_data.get("verdict") if p136_data else None,
            "passed": p136_data.get("passed") if p136_data else None,
            "total": p136_data.get("total") if p136_data else None,
        },
        "p140_gate": {
            "report": str(p140_file) if p140_file.exists() else None,
            "verdict": p140_data.get("verdict") if p140_data else None,
            "recommend_p37_exploratory": p140_recommend_p37,
        },
        "p137_diagnostics": {
            "report": str(p137_file) if p137_file else None,
            "native_full_vs_torch_slot_max_abs": (
                p137_data["summary"]["native_full_vs_torch_slot_max_abs"]
                if p137_data and "summary" in p137_data else None
            ),
            "native_full_ms_mean": (
                p137_data["summary"]["native_full_ms_mean"]
                if p137_data and "summary" in p137_data else None
            ),
        },
        "candidates": candidates,
        "summary": {
            "has_default_candidate": any_default,
            "has_amber_candidate": any_amber,
            "best_verdict": (
                "DEFAULT" if any_default
                else "AMBER_FAST" if any_amber
                else "CLOSED"
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render unified Markdown table."""
    lines = [
        "# Native MoE Candidate Summary",
        "",
        f"**Generated:** {report['created']}",
        "",
    ]

    # p136
    p136 = report["p136"]
    lines.append("## Prerequisites")
    lines.append("")
    lines.append(f"| Source | Status |")
    lines.append(f"|--------|--------|")
    p136_status = (
        f"{p136['verdict']} ({p136['passed']}/{p136['total']})"
        if p136["verdict"] else "MISSING"
    )
    lines.append(f"| p136 slot-order contract | {p136_status} |")
    p140 = report["p140_gate"]
    p140_status = p140["verdict"] or "MISSING"
    lines.append(
        f"| P140 risk gate | {p140_status} "
        f"(recommend_p37={p140['recommend_p37_exploratory']}) |")
    p137 = report["p137_diagnostics"]
    p137_status = "present" if p137["report"] else "MISSING"
    lines.append(f"| P137 diagnostics | {p137_status} |")
    lines.append("")

    # Candidates table
    lines.append("## Candidates")
    lines.append("")
    lines.append(
        "| Candidate | slot_max_abs | unique_max_abs | cosine_min "
        "| latency (ms) | Verdict | Next Step |")
    lines.append(
        "|-----------|-------------|----------------|-----------"
        "|-------------|---------|-----------|")

    for c in report["candidates"]:
        sa = f"{c['slot_max_abs']:.6e}" if c["slot_max_abs"] is not None else "—"
        ua = f"{c['unique_max_abs']:.6e}" if c.get("unique_max_abs") is not None else "—"
        cm = f"{c['cosine_min']:.10f}" if c.get("cosine_min") is not None else "—"
        lat = f"{c['avg_latency_ms']:.4f}" if c.get("avg_latency_ms") is not None else "—"
        badge = {
            "DEFAULT": "🟢",
            "AMBER_FAST": "🟡",
            "EXACT_SLOW": "🔵",
            "CLOSED": "🔴",
            "MISSING": "⚪",
        }.get(c["verdict"], "❓")
        lines.append(
            f"| {c['label']} | {sa} | {ua} | {cm} "
            f"| {lat} | {badge} {c['verdict']} | {c['recommend_next_step']} |")

    lines.append("")

    # Overall
    summ = report["summary"]
    badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡", "CLOSED": "🔴"}.get(
        summ["best_verdict"], "⚪")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"**Best verdict: {badge} {summ['best_verdict']}**")
    lines.append("")
    if summ["has_default_candidate"]:
        lines.append("> ✅ A DEFAULT candidate exists — eligible for serving path promotion.")
    elif summ["has_amber_candidate"]:
        lines.append(
            "> 🟡 AMBER candidates exist — no default promote. "
            "P37 exploratory permitted if P140 gate clears.")
    else:
        lines.append(
            "> 🔴 No viable candidates. Further kernel work needed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified Native MoE candidate summary")
    ap.add_argument("--report-dir", default="reports/qwen36_35b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[summary] ERROR: {report_dir} not found", file=sys.stderr)
        return 1

    report = gather(report_dir)

    out_json = Path(args.out or report_dir / "native_moe_candidate_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    out_md = Path(args.md_out or report_dir / "native_moe_candidate_summary.md")
    out_md.write_text(render_markdown(report))

    # Console
    print(f"\n{'='*80}")
    print(" NATIVE MOE CANDIDATE SUMMARY")
    print(f"{'='*80}")
    for c in report["candidates"]:
        badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡", "EXACT_SLOW": "🔵",
                 "CLOSED": "🔴", "MISSING": "⚪"}.get(c["verdict"], "❓")
        lat = f"{c['avg_latency_ms']:.4f}ms" if c.get("avg_latency_ms") else "—"
        sa = f"{c['slot_max_abs']:.2e}" if c.get("slot_max_abs") is not None else "—"
        print(f"  {badge} {c['verdict']:<12} lat={lat:<12} abs={sa:<12} {c['recommend_next_step']}")
    print(f"{'='*80}")
    summ = report["summary"]
    badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡", "CLOSED": "🔴"}.get(
        summ["best_verdict"], "⚪")
    print(f"  BEST: {badge} {summ['best_verdict']}")
    print(f"{'='*80}")
    print(f"  JSON: {out_json}")
    print(f"   MD:  {out_md}")
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
