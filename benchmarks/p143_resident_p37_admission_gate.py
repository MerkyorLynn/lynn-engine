#!/usr/bin/env python3
"""P143 · Resident P37 admission gate.

Read-only gate that prevents AMBER_GRAPHSAFE fixture candidates from being
promoted to resident/default.  Consumes a stage report (p142 graphsafe) and
an optional P37 end-to-end report.

Verdicts (never outputs DEFAULT_PROMOTE):

  CLOSED_STAGE_BLOCK     : stage verdict not AMBER_GRAPHSAFE/DEFAULT_STAGE
                           OR stage latency/max_abs beyond thresholds
  WAITING_FOR_P37_REPORT : P37 report not found
  CLOSED_GRAPH_COLLAPSE  : P37 report shows collapse (token0_collapse,
                           repetition, or collapse flag)
  P25_ALLOWED            : P37 exact=true OR passed=3,total=3
  CLOSED_P37_DRIFT       : exact=false and no collapse

Usage
─────
  python benchmarks/p143_resident_p37_admission_gate.py \\
    --report-dir reports/qwen36_35b \\
    --out reports/qwen36_35b/p143_resident_p37_admission.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────

STAGE_MAX_ABS = 0.003
STAGE_MAX_LATENCY_MS = 0.06

ALLOWED_STAGE_VERDICTS = {"AMBER_GRAPHSAFE", "DEFAULT_STAGE"}


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


def _find_latest(report_dir: Path, pattern: str) -> Path | None:
    matches = sorted(report_dir.glob(pattern),
                     key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


# ─────────────────────────────────────────────────────────────
# P37 field extraction (schema-tolerant)
# ─────────────────────────────────────────────────────────────

def _get_bool(data: dict, *keys: str) -> bool | None:
    """Return the first truthy boolean found in keys, or None."""
    for k in keys:
        if k in data:
            v = data[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes")
            if isinstance(v, (int, float)):
                return bool(v)
    return None


def _get_passed_total(data: dict) -> tuple[int | None, int | None]:
    """Extract passed/total from various schema formats."""
    passed = data.get("passed")
    total = data.get("total")
    if passed is not None and total is not None:
        return int(passed), int(total)
    # Also check nested "summary"
    summary = data.get("summary", {})
    if isinstance(summary, dict):
        passed = summary.get("passed")
        total = summary.get("total")
        if passed is not None and total is not None:
            return int(passed), int(total)
    return None, None


# ─────────────────────────────────────────────────────────────
# Gate logic
# ─────────────────────────────────────────────────────────────

def _admit(stage_verdict: str | None,
           stage_max_abs: float | None,
           stage_latency: float | None,
           p37_data: dict[str, Any] | None,
           p37_found: bool,
           ) -> tuple[str, str]:
    """Return (verdict, reason)."""

    # ── Stage check ──
    if stage_verdict not in ALLOWED_STAGE_VERDICTS:
        return "CLOSED_STAGE_BLOCK", (
            f"stage verdict '{stage_verdict}' not in "
            f"{sorted(ALLOWED_STAGE_VERDICTS)}")
    if stage_max_abs is not None and stage_max_abs > STAGE_MAX_ABS:
        return "CLOSED_STAGE_BLOCK", (
            f"stage max_abs {stage_max_abs:.6f} > {STAGE_MAX_ABS}")
    if stage_latency is not None and stage_latency > STAGE_MAX_LATENCY_MS:
        return "CLOSED_STAGE_BLOCK", (
            f"stage latency {stage_latency:.4f}ms > {STAGE_MAX_LATENCY_MS}ms")

    # ── P37 presence ──
    if not p37_found or p37_data is None:
        return "WAITING_FOR_P37_REPORT", "P37 report not found"

    # ── Collapse check (first, because it overrides) ──
    collapse = _get_bool(p37_data, "collapse", "token0_collapse",
                         "repetition")
    if collapse:
        return "CLOSED_GRAPH_COLLAPSE", (
            "P37 report indicates graph collapse (collapse/token0_collapse/"
            "repetition)")

    # ── Exact / passed ──
    exact = _get_bool(p37_data, "exact", "exact_match")
    passed, total = _get_passed_total(p37_data)

    if exact is True:
        return "P25_ALLOWED", "P37 exact match"
    if passed is not None and total is not None and total > 0 and passed == total:
        return "P25_ALLOWED", f"P37 all tests passed ({passed}/{total})"

    # ── Drift ──
    if exact is False:
        return "CLOSED_P37_DRIFT", "P37 exact=false, no collapse detected"

    # Fallback: if exact is None and no passed/total, treat as drift
    return "CLOSED_P37_DRIFT", (
        "P37 report present but no exact/pass signal found")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def gather(report_dir: Path,
           stage_path: str | None = None,
           p37_path: str | None = None,
           ) -> dict[str, Any]:
    """Run the admission gate and return the report dict."""

    # ── Stage report ──
    if stage_path:
        stage_file = Path(stage_path)
    else:
        stage_file = _find_latest(report_dir,
                                  "p142_graphsafe_v31_fixture_report*.json")
    stage_data = _load_json(stage_file) if stage_file else None

    stage_verdict = stage_data.get("verdict") if stage_data else None
    stage_max_abs = stage_data.get("max_max_abs") if stage_data else None
    stage_latency = stage_data.get("avg_latency_ms") if stage_data else None
    stage_cosine = stage_data.get("min_cosine") if stage_data else None
    stage_candidate = stage_data.get("candidate") if stage_data else None

    # ── P37 report ──
    p37_found = False
    if p37_path:
        p37_file = Path(p37_path)
        p37_found = p37_file.exists()
    else:
        # Deliberately do not auto-discover arbitrary historical P37 files.
        # This gate is candidate-specific; callers must pass P37_REPORT once a
        # fresh resident graph-safe P37 probe exists. Picking up stale reports
        # can incorrectly close or admit a new candidate.
        p37_file = None
        p37_found = False
    p37_data = _load_json(p37_file) if p37_found else None

    # ── Gate ──
    verdict, reason = _admit(
        stage_verdict, stage_max_abs, stage_latency,
        p37_data, p37_found)

    return {
        "schema": "lynn-p143-resident-p37-admission-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_dir": str(report_dir),
        "inputs": {
            "stage_report": {
                "path": str(stage_file) if stage_file else None,
                "candidate": stage_candidate,
                "verdict": stage_verdict,
                "max_max_abs": stage_max_abs,
                "avg_latency_ms": stage_latency,
                "min_cosine": stage_cosine,
            },
            "p37_report": {
                "path": str(p37_file) if p37_found else None,
                "found": p37_found,
                "exact": _get_bool(p37_data, "exact", "exact_match") if p37_data else None,
                "collapse": (
                    _get_bool(p37_data, "collapse", "token0_collapse",
                              "repetition") if p37_data else None),
                "passed": _get_passed_total(p37_data)[0] if p37_data else None,
                "total": _get_passed_total(p37_data)[1] if p37_data else None,
            },
        },
        "admission": {
            "verdict": verdict,
            "reason": reason,
            "default_promote_allowed": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render admission gate report as Markdown."""
    adm = report["admission"]
    badge = {
        "P25_ALLOWED": "🟢",
        "WAITING_FOR_P37_REPORT": "⏳",
        "CLOSED_STAGE_BLOCK": "🔴",
        "CLOSED_GRAPH_COLLAPSE": "🔴",
        "CLOSED_P37_DRIFT": "🔴",
    }.get(adm["verdict"], "⚪")

    lines = [
        "# P143 · Resident P37 Admission Gate",
        "",
        f"**Generated:** {report['created']}",
        "",
        "## Admission Decision",
        "",
        f"**Verdict: {badge} {adm['verdict']}**",
        "",
        f"> {adm['reason']}",
        "",
        f"| Flag | Value |",
        f"|------|-------|",
        f"| default_promote_allowed | {'✅' if adm['default_promote_allowed'] else '❌ (never allowed)'} |",
        "",
        "## Inputs",
        "",
    ]

    # Stage
    inp = report["inputs"]
    stg = inp["stage_report"]
    stg_badge = {"AMBER_GRAPHSAFE": "🟡", "DEFAULT_STAGE": "🟢"}.get(
        stg["verdict"] or "", "⚪")
    lines.append("### Stage Report (p142 graphsafe)")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| path | `{Path(stg['path']).name if stg['path'] else '—'}` |")
    lines.append(f"| candidate | {stg.get('candidate') or '—'} |")
    lines.append(f"| verdict | {stg_badge} {stg['verdict'] or '—'} |")
    lines.append(f"| max_max_abs | {stg['max_max_abs'] or '—'} |")
    lat = stg.get("avg_latency_ms")
    lines.append(f"| avg_latency_ms | {f'{lat:.4f}' if lat is not None else '—'} |")
    cos = stg.get("min_cosine")
    lines.append(f"| min_cosine | {f'{cos:.10f}' if cos is not None else '—'} |")
    lines.append("")

    # P37
    p37 = inp["p37_report"]
    lines.append("### P37 Report")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    if p37["found"]:
        lines.append(f"| path | `{Path(p37['path']).name}` |")
        lines.append(f"| exact | {p37.get('exact')} |")
        lines.append(f"| collapse | {p37.get('collapse')} |")
        lines.append(f"| passed | {p37.get('passed') or '—'} |")
        lines.append(f"| total | {p37.get('total') or '—'} |")
    else:
        lines.append(f"| path | ⚪ NOT FOUND |")
    lines.append("")

    # Decision matrix
    lines.append("## Decision Matrix")
    lines.append("")
    lines.append("| Stage verdict | P37 state | → Admission |")
    lines.append("|---------------|-----------|-------------|")
    lines.append("| not AMBER_GRAPHSAFE/DEFAULT_STAGE | — | 🔴 CLOSED_STAGE_BLOCK |")
    lines.append("| ok, but max_abs/latency exceeded | — | 🔴 CLOSED_STAGE_BLOCK |")
    lines.append("| ok | missing | ⏳ WAITING_FOR_P37_REPORT |")
    lines.append("| ok | collapse=true | 🔴 CLOSED_GRAPH_COLLAPSE |")
    lines.append("| ok | exact=true OR passed=total | 🟢 P25_ALLOWED |")
    lines.append("| ok | exact=false, no collapse | 🔴 CLOSED_P37_DRIFT |")
    lines.append("")
    lines.append("> **Note:** DEFAULT_PROMOTE is never output. "
                 "Maximum admission is P25_ALLOWED.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P143 · Resident P37 admission gate")
    ap.add_argument("--report-dir", default="reports/qwen36_35b")
    ap.add_argument("--stage-report", default=None,
                    help="Explicit stage report path (env: STAGE_REPORT)")
    ap.add_argument("--p37-report", default=None,
                    help="Explicit P37 report path (env: P37_REPORT)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[p143] ERROR: {report_dir} not found", file=sys.stderr)
        return 1

    stage_path = args.stage_report or os.environ.get("STAGE_REPORT")
    p37_path = args.p37_report or os.environ.get("P37_REPORT")

    report = gather(report_dir, stage_path=stage_path, p37_path=p37_path)

    out_json = Path(args.out
                    or report_dir / "p143_resident_p37_admission.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    out_md = Path(args.md_out
                  or report_dir / "p143_resident_p37_admission.md")
    out_md.write_text(render_markdown(report))

    # Console
    adm = report["admission"]
    badge = {
        "P25_ALLOWED": "🟢",
        "WAITING_FOR_P37_REPORT": "⏳",
        "CLOSED_STAGE_BLOCK": "🔴",
        "CLOSED_GRAPH_COLLAPSE": "🔴",
        "CLOSED_P37_DRIFT": "🔴",
    }.get(adm["verdict"], "⚪")
    print(f"\n{'='*70}")
    print(" P143 · RESIDENT P37 ADMISSION GATE")
    print(f"{'='*70}")
    print(f"  {badge} {adm['verdict']}")
    print(f"  {adm['reason']}")
    print(f"{'─'*70}")

    inp = report["inputs"]
    stg = inp["stage_report"]
    p37 = inp["p37_report"]
    sv = stg['verdict'] or '—'
    sa = f"{stg['max_max_abs']:.6f}" if stg.get('max_max_abs') is not None else '—'
    sl = f"{stg['avg_latency_ms']:.4f}ms" if stg.get('avg_latency_ms') is not None else '—'
    print(f"  stage:  {sv:<20} abs={sa:<12} lat={sl}")
    if p37["found"]:
        print(f"  P37:    exact={p37.get('exact')}  collapse={p37.get('collapse')}  "
              f"passed={p37.get('passed')}/{p37.get('total')}")
    else:
        print(f"  P37:    ⚪ NOT FOUND")
    print(f"{'─'*70}")
    print(f"  default_promote_allowed = {adm['default_promote_allowed']}")
    print(f"{'='*70}")
    print(f"  JSON: {out_json}")
    print(f"   MD:  {out_md}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
