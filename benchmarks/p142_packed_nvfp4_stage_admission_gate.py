#!/usr/bin/env python3
"""P142 · Packed NVFP4 stage admission gate.

Read-only harness that consumes p141 stage diagnostics (AMBER_STAGE check),
p140 packed NVFP4 probe report, and the unified candidate summary, then
outputs a three-tier admission verdict:

  DEFAULT_BLOCKED : p141 verdict is AMBER_STAGE OR packed probe is CLOSED
                    → no default/resident promote permitted
  P37_ALLOWED     : p141 cleared AND packed probe accuracy within P37 bounds
                    → P37 exploratory work may proceed
  CLOSED          : packed probe accuracy or latency beyond hard limits
                    → block all further work on this path

This gate exists to prevent fixture-stage AMBER from leaking into the
resident/default promotion path.  It does NOT touch kernel implementation.

Usage
─────
  python benchmarks/p142_packed_nvfp4_stage_admission_gate.py \\
    --report-dir reports/qwen36_35b \\
    --out reports/qwen36_35b/p142_packed_nvfp4_stage_admission.json

Override individual paths with --p141-report, --p140-packed-report,
--candidate-summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────

# P37 exploratory bounds for packed NVFP4
P37_MAX_ABS = 0.005
P37_MAX_LATENCY_MS = 0.10

# Hard CLOSED limits — any metric beyond these is auto-CLOSED
HARD_MAX_ABS = 0.01
HARD_MAX_LATENCY_MS = 0.15


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
# Gate logic
# ─────────────────────────────────────────────────────────────

def _within_p37_bounds(max_abs: float | None, latency_ms: float | None) -> bool:
    return (
        max_abs is not None
        and max_abs <= P37_MAX_ABS
        and latency_ms is not None
        and latency_ms <= P37_MAX_LATENCY_MS
    )


def _admit(p141_verdict: str | None,
           p141_max_abs: float | None,
           p141_latency: float | None,
           packed_verdict: str | None,
           packed_max_abs: float | None,
           packed_latency: float | None,
           ) -> tuple[str, str, bool]:
    """Return (admission_verdict, reason).

    Decision hierarchy (first match wins):
      1. HARD CLOSED — accuracy/latency beyond absolute limits
      2. DEFAULT_BLOCKED — p141 AMBER_STAGE or packed CLOSED
      3. P37_ALLOWED — p141 cleared, packed within P37 bounds
      4. CLOSED — fallback

    ``DEFAULT_BLOCKED`` and ``p37_exploratory_allowed`` are intentionally
    separate flags.  A fixture-stage AMBER can never promote to default, but if
    it is fast and within loose P37 bounds it may feed a graph-safe resident ABI
    experiment.  The older p140 packed probe must not veto the newer p141-v2
    exploratory path.
    """
    # ── Hard limits ──
    if packed_max_abs is not None and packed_max_abs > HARD_MAX_ABS:
        return "CLOSED", (
            f"packed max_abs {packed_max_abs:.6f} > hard limit {HARD_MAX_ABS}"), False
    if packed_latency is not None and packed_latency > HARD_MAX_LATENCY_MS:
        return "CLOSED", (
            f"packed latency {packed_latency:.4f}ms > hard limit "
            f"{HARD_MAX_LATENCY_MS}ms"), False

    # ── AMBER_STAGE blocks default promotion ──
    if p141_verdict == "AMBER_STAGE":
        allowed = _within_p37_bounds(p141_max_abs, p141_latency)
        return "DEFAULT_BLOCKED", (
            "p141 verdict AMBER_STAGE — fixture-stage AMBER prevents "
            "default/resident promote"), allowed

    # ── Packed probe CLOSED blocks default too ──
    if packed_verdict == "CLOSED":
        return "DEFAULT_BLOCKED", (
            "packed NVFP4 probe verdict CLOSED — no default/resident promote"), False

    # ── P37 exploratory ──
    if _within_p37_bounds(packed_max_abs, packed_latency):
        return "P37_ALLOWED", (
            "packed probe within P37 bounds — exploratory work permitted"), True

    return "CLOSED", "conditions not met for any admission tier", False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def gather(report_dir: Path,
           p141_path: str | None = None,
           p140_packed_path: str | None = None,
           summary_path: str | None = None,
           ) -> dict[str, Any]:
    """Run the admission gate and return the report dict."""

    # ── p141 stage diagnostics ──
    if p141_path:
        p141_file = Path(p141_path)
    else:
        p141_file = _find_latest(report_dir, "p141_v2_report*.json")
    p141_data = _load_json(p141_file) if p141_file else None

    # ── p140 packed NVFP4 probe ──
    if p140_packed_path:
        p140_file = Path(p140_packed_path)
    else:
        p140_file = _find_latest(report_dir, "p140_packed_nvfp4_probe_report*.json")
    p140_data = _load_json(p140_file) if p140_file else None

    # ── Candidate summary ──
    if summary_path:
        summary_file = Path(summary_path)
    else:
        summary_file = _find_latest(report_dir, "native_moe_candidate_summary*.json")
    summary_data = _load_json(summary_file) if summary_file else None

    # ── Extract metrics ──
    p141_verdict = p141_data.get("verdict") if p141_data else None
    p141_max_abs = p141_data.get("max_max_abs") if p141_data else None
    p141_latency = p141_data.get("avg_latency_ms") if p141_data else None
    p141_cosine = p141_data.get("min_cosine") if p141_data else None

    packed_verdict = p140_data.get("verdict") if p140_data else None
    packed_max_abs = p140_data.get("max_max_abs") if p140_data else None
    packed_latency = p140_data.get("avg_latency_ms") if p140_data else None
    packed_cosine = p140_data.get("min_cosine") if p140_data else None

    summary_best = None
    summary_has_default = None
    summary_has_amber = None
    if summary_data and "summary" in summary_data:
        summary_best = summary_data["summary"].get("best_verdict")
        summary_has_default = summary_data["summary"].get("has_default_candidate")
        summary_has_amber = summary_data["summary"].get("has_amber_candidate")

    # ── Gate ──
    admission_verdict, reason, p37_exploratory_allowed = _admit(
        p141_verdict, p141_max_abs, p141_latency,
        packed_verdict, packed_max_abs, packed_latency)

    # ── Derive flags ──
    default_promote_blocked = admission_verdict in ("DEFAULT_BLOCKED", "CLOSED")

    return {
        "schema": "lynn-p142-packed-nvfp4-stage-admission-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_dir": str(report_dir),
        "inputs": {
            "p141_v2_report": {
                "path": str(p141_file) if p141_file else None,
                "verdict": p141_verdict,
                "max_max_abs": p141_max_abs,
                "avg_latency_ms": p141_latency,
                "min_cosine": p141_cosine,
            },
            "p140_packed_nvfp4_probe": {
                "path": str(p140_file) if p140_file else None,
                "verdict": packed_verdict,
                "max_max_abs": packed_max_abs,
                "avg_latency_ms": packed_latency,
                "min_cosine": packed_cosine,
            },
            "candidate_summary": {
                "path": str(summary_file) if summary_file else None,
                "best_verdict": summary_best,
                "has_default_candidate": summary_has_default,
                "has_amber_candidate": summary_has_amber,
            },
        },
        "admission": {
            "verdict": admission_verdict,
            "reason": reason,
            "default_promote_blocked": default_promote_blocked,
            "p37_exploratory_allowed": p37_exploratory_allowed,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render admission gate report as Markdown."""
    adm = report["admission"]
    badge = {
        "DEFAULT_BLOCKED": "🟡",
        "P37_ALLOWED": "🔵",
        "CLOSED": "🔴",
    }.get(adm["verdict"], "⚪")

    lines = [
        "# P142 · Packed NVFP4 Stage Admission Gate",
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
        f"| default_promote_blocked | {'✅' if adm['default_promote_blocked'] else '❌'} |",
        f"| p37_exploratory_allowed | {'✅' if adm['p37_exploratory_allowed'] else '❌'} |",
        "",
        "## Inputs",
        "",
    ]

    # p141
    inp = report["inputs"]
    p141 = inp["p141_v2_report"]
    p141_badge = {"AMBER_STAGE": "🟡", "GREEN": "🟢"}.get(
        p141["verdict"] or "", "⚪")
    lines.append("### p141 Stage Diagnostics (V2)")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| path | `{Path(p141['path']).name if p141['path'] else '—'}` |")
    lines.append(f"| verdict | {p141_badge} {p141['verdict'] or '—'} |")
    lines.append(f"| max_max_abs | {p141['max_max_abs'] or '—'} |")
    lat = p141.get("avg_latency_ms")
    lines.append(f"| avg_latency_ms | {f'{lat:.4f}' if lat is not None else '—'} |")
    cos = p141.get("min_cosine")
    lines.append(f"| min_cosine | {f'{cos:.10f}' if cos is not None else '—'} |")
    lines.append("")

    # p140 packed
    pp = inp["p140_packed_nvfp4_probe"]
    pp_badge = {"CLOSED": "🔴", "GREEN": "🟢", "DEFAULT": "🟢"}.get(
        pp["verdict"] or "", "⚪")
    lines.append("### p140 Packed NVFP4 Probe")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| path | `{Path(pp['path']).name if pp['path'] else '—'}` |")
    lines.append(f"| verdict | {pp_badge} {pp['verdict'] or '—'} |")
    lines.append(f"| max_max_abs | {pp['max_max_abs'] or '—'} |")
    lat = pp.get("avg_latency_ms")
    lines.append(f"| avg_latency_ms | {f'{lat:.4f}' if lat is not None else '—'} |")
    cos = pp.get("min_cosine")
    lines.append(f"| min_cosine | {f'{cos:.10f}' if cos is not None else '—'} |")
    lines.append("")

    # candidate summary
    cs = inp["candidate_summary"]
    lines.append("### Candidate Summary")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| path | `{Path(cs['path']).name if cs['path'] else '—'}` |")
    lines.append(f"| best_verdict | {cs['best_verdict'] or '—'} |")
    lines.append(f"| has_default_candidate | {'✅' if cs.get('has_default_candidate') else '❌'} |")
    lines.append(f"| has_amber_candidate | {'✅' if cs.get('has_amber_candidate') else '❌'} |")
    lines.append("")

    # Decision matrix
    lines.append("## Decision Matrix")
    lines.append("")
    lines.append("| p141 verdict | packed verdict | → Admission |")
    lines.append("|-------------|---------------|-------------|")
    lines.append("| AMBER_STAGE within P37 bounds | any | 🟡 DEFAULT_BLOCKED + P37_ALLOWED |")
    lines.append("| AMBER_STAGE outside P37 bounds | any | 🟡 DEFAULT_BLOCKED |")
    lines.append("| GREEN | CLOSED | 🟡 DEFAULT_BLOCKED |")
    lines.append("| GREEN | GREEN/AMBER (within P37 bounds) | 🔵 P37_ALLOWED |")
    lines.append("| any | hard limit exceeded | 🔴 CLOSED |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P142 · Packed NVFP4 stage admission gate")
    ap.add_argument("--report-dir", default="reports/qwen36_35b")
    ap.add_argument("--p141-report", default=None,
                    help="Explicit p141 V2 report path (env: P141_REPORT)")
    ap.add_argument("--p140-packed-report", default=None,
                    help="Explicit p140 packed NVFP4 probe path "
                         "(env: P140_PACKED_REPORT)")
    ap.add_argument("--candidate-summary", default=None,
                    help="Explicit candidate summary path "
                         "(env: CANDIDATE_SUMMARY)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[p142] ERROR: {report_dir} not found", file=sys.stderr)
        return 1

    import os
    p141_path = args.p141_report or os.environ.get("P141_REPORT")
    p140_packed_path = (args.p140_packed_report
                        or os.environ.get("P140_PACKED_REPORT"))
    summary_path = args.candidate_summary or os.environ.get("CANDIDATE_SUMMARY")

    report = gather(report_dir,
                    p141_path=p141_path,
                    p140_packed_path=p140_packed_path,
                    summary_path=summary_path)

    out_json = Path(args.out
                    or report_dir / "p142_packed_nvfp4_stage_admission.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    out_md = Path(args.md_out
                  or report_dir / "p142_packed_nvfp4_stage_admission.md")
    out_md.write_text(render_markdown(report))

    # Console
    adm = report["admission"]
    badge = {"DEFAULT_BLOCKED": "🟡", "P37_ALLOWED": "🔵",
             "CLOSED": "🔴"}.get(adm["verdict"], "⚪")
    print(f"\n{'='*70}")
    print(" P142 · PACKED NVFP4 STAGE ADMISSION GATE")
    print(f"{'='*70}")
    print(f"  {badge} {adm['verdict']}")
    print(f"  {adm['reason']}")
    print(f"{'─'*70}")

    inp = report["inputs"]
    p141 = inp["p141_v2_report"]
    pp = inp["p140_packed_nvfp4_probe"]
    p141_v = p141['verdict'] or '—'
    pp_v = pp['verdict'] or '—'
    p141_a = f"{p141['max_max_abs']:.6f}" if p141.get('max_max_abs') is not None else '—'
    pp_a = f"{pp['max_max_abs']:.6f}" if pp.get('max_max_abs') is not None else '—'
    p141_l = f"{p141['avg_latency_ms']:.4f}ms" if p141.get('avg_latency_ms') is not None else '—'
    pp_l = f"{pp['avg_latency_ms']:.4f}ms" if pp.get('avg_latency_ms') is not None else '—'
    print(f"  p141:    {p141_v:<15} abs={p141_a:<12} lat={p141_l}")
    print(f"  packed:  {pp_v:<15} abs={pp_a:<12} lat={pp_l}")
    print(f"{'─'*70}")
    print(f"  default_promote_blocked = {adm['default_promote_blocked']}")
    print(f"  p37_exploratory_allowed = {adm['p37_exploratory_allowed']}")
    print(f"{'='*70}")
    print(f"  JSON: {out_json}")
    print(f"   MD:  {out_md}")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
