#!/usr/bin/env python3
"""Unified Native MoE candidate summary.

Reads all available candidate reports (graceful on missing) and produces a
single JSON + Markdown table with per-candidate verdict and next-step
recommendation.

Auto-discovers reports by naming convention in --report-dir.
Optionally merges reports from extra directories (--extra-report-dir) to
pick up candidates from other worktrees (e.g. TensorCore probes).

Usage:
  python scripts/summarize_qwen36_native_moe_candidates.py \\
    --report-dir reports/qwen36_35b \\
    --extra-report-dir /path/to/other/reports/qwen36_35b \\
    --out reports/qwen36_35b/native_moe_candidate_summary.json \\
    --md-out reports/qwen36_35b/native_moe_candidate_summary.md
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


def _find_latest_in_dirs(dirs: list[Path], pattern: str) -> Path | None:
    """Search multiple dirs in order; return the first match found."""
    for d in dirs:
        matches = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime)
        if matches:
            return matches[-1]
    return None


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


def _gather_packed_slot(report_dir: Path,
                        search_dirs: list[Path],
                        p138_manifest_path: str | None = None,
                        p139_report_path: str | None = None,
                        ) -> dict[str, Any]:
    """Gather p138 packed-slot manifest + p139 contract data."""
    # ── p138 ──
    if p138_manifest_path:
        p138_file = Path(p138_manifest_path)
    else:
        p138_file = _find_latest_in_dirs(
            search_dirs, "p138_packed_slot_fixtures_manifest*.json")
    p138_data = _load_json(p138_file) if p138_file else None

    # ── p139 ──
    if p139_report_path:
        p139_file = Path(p139_report_path)
    else:
        p139_file = _find_latest_in_dirs(
            search_dirs, "p139_slot_packed_contract*.json")
    p139_data = _load_json(p139_file) if p139_file else None

    # ── Compute sizes from p138 ──
    packed_bytes_total = 0
    bf16_bytes_total = 0
    if p138_data and "fixtures" in p138_data:
        for fx in p138_data["fixtures"]:
            packed_bytes_total += fx.get("packed_bytes", 0)
            bf16_bytes_total += fx.get("bf16_equiv_bytes", 0)

    packed_fixture_mb = packed_bytes_total / (1024 * 1024) if packed_bytes_total else None
    bf16_equiv_mb = bf16_bytes_total / (1024 * 1024) if bf16_bytes_total else None
    size_reduction_pct = (
        (1.0 - packed_bytes_total / bf16_bytes_total) * 100.0
        if bf16_bytes_total > 0 else None
    )

    # ── p139 verdict ──
    p139_verdict = p139_data.get("verdict") if p139_data else None
    p139_max_abs_max = p139_data.get("max_abs_max") if p139_data else None
    p139_passed = p139_data.get("passed") if p139_data else None
    p139_total = p139_data.get("total") if p139_data else None
    num_fixtures = p138_data.get("num_fixtures") if p138_data else None

    packed_ready = bool(
        p139_verdict == "GREEN"
        and packed_fixture_mb is not None
        and bf16_equiv_mb is not None
    )

    # ── recommend ──
    if (p139_verdict == "GREEN"
            and size_reduction_pct is not None and size_reduction_pct > 60.0):
        recommend_next_step = "build native packed NVFP4 kernel probe"
    elif p139_verdict == "GREEN":
        recommend_next_step = "packed fixtures ready — evaluate kernel feasibility"
    elif p139_verdict:
        recommend_next_step = "fix p139 contract failures before kernel work"
    else:
        recommend_next_step = "no p138/p139 data"

    return {
        "p138_manifest": str(p138_file) if p138_file else None,
        "p139_report": str(p139_file) if p139_file else None,
        "num_fixtures": num_fixtures,
        "packed_fixture_mb": round(packed_fixture_mb, 2) if packed_fixture_mb is not None else None,
        "bf16_equiv_mb": round(bf16_equiv_mb, 2) if bf16_equiv_mb is not None else None,
        "size_reduction_pct": round(size_reduction_pct, 1) if size_reduction_pct is not None else None,
        "p139_verdict": p139_verdict,
        "p139_max_abs_max": p139_max_abs_max,
        "p139_passed": p139_passed,
        "p139_total": p139_total,
        "packed_ready_for_kernel": packed_ready,
        "recommend_next_step": recommend_next_step,
    }


def _classify(slot_abs: float | None, cos_min: float | None,
              lat: float | None, uniq_abs: float | None,
              p140_recommend_p37: bool,
              pretransposed: bool = False) -> tuple[str, str]:
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
        amber_label = "AMBER_FAST_PRETRANSPOSED" if pretransposed else "AMBER_FAST"
        if p140_recommend_p37:
            return amber_label, "P37 exploratory"
        return amber_label, "await P140 gate clearance"

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
            "native_slot_tensorcore_probe*report*.json",
            "native_slot_tensorcore_probe*.json",
        ],
    },
    {
        "id": "native_slot_fused_bf16",
        "label": "native_slot_fused_bf16 (fused probe)",
        "patterns": [
            "native_slot_fused*report*.json",
            "native_slot_tensorcore_fused_probe*report*.json",
            "native_slot_tensorcore_fused_probe*.json",
        ],
    },
    {
        "id": "native_slot_tensorcore_pretransposed_probe",
        "label": "native_slot_tensorcore_pretransposed_probe (p139b)",
        "patterns": [
            "native_slot_tensorcore_pretransposed_probe*report*.json",
            "native_slot_tensorcore_pretransposed_probe*.json",
        ],
        "pretransposed": True,
    },
]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def gather(report_dir: Path,
           extra_dirs: list[Path] | None = None,
           p138_manifest_path: str | None = None,
           p139_report_path: str | None = None,
           ) -> dict[str, Any]:
    """Gather all candidate data into a unified report.

    ``report_dir`` is the primary search directory.  ``extra_dirs`` are
    consulted in order when a report is not found locally.
    ``p138_manifest_path`` and ``p139_report_path`` allow explicit override
    of packed-slot report locations (e.g. from env vars).
    """
    search_dirs = [report_dir] + (extra_dirs or [])

    # ── p136 ──
    p136_file = _find_latest_in_dirs(
        search_dirs, "p136_slot_repack_contract_slotorder_report*.json")
    p136_data = _load_json(p136_file) if p136_file else None

    # ── p140 gate (local only — it is branch-specific) ──
    p140_file = report_dir / "p140_native_moe_risk_gate.json"
    p140_data = _load_json(p140_file)
    p140_recommend_p37 = bool(
        p140_data and p140_data.get("recommend_p37_exploratory"))

    # ── p137 (slot-order preferred) ──
    p137_file = _find_latest_in_dirs(
        search_dirs, "p137_moe_slot_stage_diagnostics*slotorder*.json")
    if not p137_file:
        p137_file = _find_latest_in_dirs(
            search_dirs, "p137_moe_slot_stage_diagnostics*.json")
    p137_data = _load_json(p137_file) if p137_file else None

    # ── Packed-slot (p138/p139) ──
    packed_slot = _gather_packed_slot(
        report_dir, search_dirs,
        p138_manifest_path=p138_manifest_path,
        p139_report_path=p139_report_path,
    )

    # ── Candidates ──
    candidates: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        report_path = None
        for pat in spec["patterns"]:
            report_path = _find_latest_in_dirs(search_dirs, pat)
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
        pretransposed = bool(spec.get("pretransposed"))

        verdict, next_step = _classify(
            slot_abs, cos_min, lat, uniq_abs, p140_recommend_p37,
            pretransposed=pretransposed)

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
    any_amber = any(c["verdict"].startswith("AMBER_FAST") for c in candidates)

    return {
        "schema": "lynn-native-moe-candidate-summary-v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_dir": str(report_dir),
        "extra_report_dirs": [str(d) for d in (extra_dirs or [])],
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
        "packed_slot": packed_slot,
        "candidates": candidates,
        "summary": {
            "has_default_candidate": any_default,
            "has_amber_candidate": any_amber,
            "best_verdict": (
                "DEFAULT" if any_default
                else "AMBER_FAST_PRETRANSPOSED" if any(
                    c["verdict"] == "AMBER_FAST_PRETRANSPOSED"
                    for c in candidates)
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

    # Packed-slot section
    ps = report.get("packed_slot", {})
    p139_v = ps.get("p139_verdict")
    ps_badge = {"GREEN": "🟢", "RED": "🔴"}.get(p139_v or "", "⚪")
    lines.append("## Packed-Slot Readiness (p138/p139)")
    lines.append("")
    if ps.get("p138_manifest"):
        lines.append(f"- **p138 manifest:** `{Path(ps['p138_manifest']).name}`")
    else:
        lines.append("- **p138 manifest:** ⚪ MISSING")
    if ps.get("p139_report"):
        lines.append(f"- **p139 contract:** `{Path(ps['p139_report']).name}`")
    else:
        lines.append("- **p139 contract:** ⚪ MISSING")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| num_fixtures | {ps.get('num_fixtures', '—')} |")
    pf_mb = ps.get("packed_fixture_mb")
    lines.append(f"| packed_fixture_mb | {f'{pf_mb:.2f}' if pf_mb is not None else '—'} |")
    be_mb = ps.get("bf16_equiv_mb")
    lines.append(f"| bf16_equiv_mb | {f'{be_mb:.2f}' if be_mb is not None else '—'} |")
    rp = ps.get("size_reduction_pct")
    lines.append(f"| size_reduction_pct | {f'{rp:.1f}%' if rp is not None else '—'} |")
    lines.append(f"| p139_verdict | {ps_badge} {p139_v or '—'} |")
    ma = ps.get("p139_max_abs_max")
    lines.append(f"| p139_max_abs_max | {f'{ma}' if ma is not None else '—'} |")
    prk = ps.get("packed_ready_for_kernel")
    lines.append(f"| packed_ready_for_kernel | {'✅' if prk else '❌'} |")
    lines.append(f"| recommend_next_step | {ps.get('recommend_next_step', '—')} |")
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
            "AMBER_FAST_PRETRANSPOSED": "🟡",
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
    badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡",
             "AMBER_FAST_PRETRANSPOSED": "🟡", "CLOSED": "🔴"}.get(
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
    ap.add_argument("--extra-report-dir", action="append", default=[],
                    metavar="DIR",
                    help="Additional report directories to search "
                         "(can be repeated)")
    ap.add_argument("--p138-manifest", default=None,
                    help="Explicit p138 packed-slot manifest path "
                         "(env: P138_MANIFEST)")
    ap.add_argument("--p139-report", default=None,
                    help="Explicit p139 packed-slot contract path "
                         "(env: P139_REPORT)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[summary] ERROR: {report_dir} not found", file=sys.stderr)
        return 1

    extra_dirs = [Path(d) for d in args.extra_report_dir if Path(d).is_dir()]

    p138_manifest = (args.p138_manifest
                     or os.environ.get("P138_MANIFEST"))
    p139_report = (args.p139_report
                   or os.environ.get("P139_REPORT"))

    report = gather(report_dir, extra_dirs=extra_dirs,
                    p138_manifest_path=p138_manifest,
                    p139_report_path=p139_report)

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
        badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡",
                 "AMBER_FAST_PRETRANSPOSED": "🟡", "EXACT_SLOW": "🔵",
                 "CLOSED": "🔴", "MISSING": "⚪"}.get(c["verdict"], "❓")
        lat = f"{c['avg_latency_ms']:.4f}ms" if c.get("avg_latency_ms") else "—"
        sa = f"{c['slot_max_abs']:.2e}" if c.get("slot_max_abs") is not None else "—"
        print(f"  {badge} {c['verdict']:<12} lat={lat:<12} abs={sa:<12} {c['recommend_next_step']}")
    print(f"{'─'*80}")
    ps = report.get("packed_slot", {})
    p139_v = ps.get("p139_verdict")
    ps_badge = {"GREEN": "🟢", "RED": "🔴"}.get(p139_v or "", "⚪")
    rp = ps.get("size_reduction_pct")
    rp_s = f"{rp:.1f}%" if rp is not None else "—"
    prk = "✅" if ps.get("packed_ready_for_kernel") else "❌"
    print(f"  PACKED-SLOT: {ps_badge} p139={p139_v or '—':<6} "
          f"reduction={rp_s:<8} ready={prk}  {ps.get('recommend_next_step', '—')}")
    print(f"{'='*80}")
    summ = report["summary"]
    badge = {"DEFAULT": "🟢", "AMBER_FAST": "🟡",
             "AMBER_FAST_PRETRANSPOSED": "🟡", "CLOSED": "🔴"}.get(
        summ["best_verdict"], "⚪")
    print(f"  BEST: {badge} {summ['best_verdict']}")
    print(f"{'='*80}")
    print(f"  JSON: {out_json}")
    print(f"   MD:  {out_md}")
    print(f"{'='*80}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
