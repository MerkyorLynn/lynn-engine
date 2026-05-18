#!/usr/bin/env python3
"""P140 · Native MoE candidate risk gate.

Read-only harness that consumes existing p136 slot-order report, native fast
candidate report, p137 diagnostics, and strict oracle report, then outputs a
three-tier verdict: DEFAULT / AMBER / CLOSED.

No kernel execution, no resident, no model loading — pure report aggregation.

Verdict rules
─────────────
  DEFAULT : p136 18/18 GREEN  AND  candidate slot_max_abs <= 1e-3
            AND  cosine_min >= 0.999999  AND  latency <= 0.059 ms
  AMBER   : p136 18/18 GREEN  AND  candidate slot_max_abs <= 0.003
            AND  unique_max_abs <= 0.002  AND  latency <= 0.055 ms
            → annotated "no default promote; P37 exploratory only"
  CLOSED  : thresholds exceeded OR required reports missing

Usage
─────
  python benchmarks/p140_native_moe_candidate_risk_gate.py \\
    --report-dir reports/qwen36_35b \\
    --out reports/qwen36_35b/p140_native_moe_risk_gate.json

The script auto-discovers reports by naming convention.  Override paths
with --p136-report, --candidate-report, --p137-report, --strict-report.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────

# DEFAULT tier
DEFAULT_SLOT_MAX_ABS = 1e-3
DEFAULT_COSINE_MIN = 0.999999
DEFAULT_LATENCY_MS = 0.059

# AMBER tier
AMBER_SLOT_MAX_ABS = 0.003
AMBER_UNIQUE_MAX_ABS = 0.002
AMBER_LATENCY_MS = 0.055

# CLOSED tier — any metric beyond these is auto-CLOSED
CLOSED_SLOT_MAX_ABS = 0.00293  # current fast native value
CLOSED_UNIQUE_MAX_ABS = 0.00195

P136_EXPECTED_TOTAL = 18


# ─────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────

@dataclass
class GateInputs:
    """All metrics consumed by the gate."""
    # p136 contract
    p136_report_path: str | None = None
    p136_verdict: str | None = None
    p136_passed: int | None = None
    p136_total: int | None = None
    p136_max_abs_max: float | None = None

    # native fast candidate (output-owned or slot-tc)
    candidate_report_path: str | None = None
    candidate_name: str | None = None
    slot_max_abs: float | None = None
    unique_max_abs: float | None = None
    cosine_min: float | None = None
    avg_latency_ms: float | None = None

    # p137 diagnostics (optional, for context)
    p137_report_path: str | None = None
    p137_native_full_vs_torch_slot_max_abs: float | None = None
    p137_native_full_vs_torch_slot_cosine_min: float | None = None
    p137_native_full_ms_mean: float | None = None

    # strict oracle (optional, for context)
    strict_report_path: str | None = None
    strict_all_exact: bool | None = None
    strict_avg_latency_ms: float | None = None


@dataclass
class GateVerdict:
    tier: str  # DEFAULT | AMBER | CLOSED
    recommend_p37_exploratory: bool
    reasons: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Report discovery
# ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _find_latest(report_dir: Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching glob pattern."""
    matches = sorted(report_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def discover_reports(report_dir: Path) -> dict[str, Path | None]:
    """Auto-discover report files by naming convention."""
    return {
        "p136": _find_latest(report_dir, "p136_slot_repack_contract_slotorder_report*.json"),
        "candidate": _find_latest(report_dir, "native_slot_output_owned_bf16*slotorder*report*.json"),
        "candidate_fallback": _find_latest(report_dir, "native_slot_output_owned_bf16*report*.json"),
        "p137": _find_latest(report_dir, "p137_moe_slot_stage_diagnostics*slotorder*.json"),
        "p137_fallback": _find_latest(report_dir, "p137_moe_slot_stage_diagnostics*.json"),
        "strict": _find_latest(report_dir, "native_slot_strict_bf16*slotorder*report*.json"),
        "strict_fallback": _find_latest(report_dir, "native_slot_strict_bf16*report*.json"),
    }


# ─────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────

def load_gate_inputs(
    report_dir: Path,
    p136_path: str | None = None,
    candidate_path: str | None = None,
    p137_path: str | None = None,
    strict_path: str | None = None,
) -> GateInputs:
    """Load all inputs for the risk gate."""
    discovered = discover_reports(report_dir)
    inp = GateInputs()

    # ── p136 ──
    p136_file = Path(p136_path) if p136_path else discovered["p136"]
    if p136_file:
        p136 = _load_json(p136_file)
        if p136:
            inp.p136_report_path = str(p136_file)
            inp.p136_verdict = p136.get("verdict")
            inp.p136_passed = p136.get("passed")
            inp.p136_total = p136.get("total")
            inp.p136_max_abs_max = p136.get("max_abs_max")

    # ── candidate ──
    cand_file = Path(candidate_path) if candidate_path else (
        discovered["candidate"] or discovered["candidate_fallback"]
    )
    if cand_file:
        cand = _load_json(cand_file)
        if cand:
            inp.candidate_report_path = str(cand_file)
            inp.candidate_name = cand.get("candidate")
            # Support both flat summary and per-result aggregation
            inp.slot_max_abs = cand.get("slot_max_abs") or cand.get("max_max_abs")
            inp.unique_max_abs = cand.get("unique_max_abs")
            inp.cosine_min = cand.get("cosine_min") or cand.get("min_cosine")
            inp.avg_latency_ms = cand.get("avg_latency_ms")

            # Aggregate cosine_min from per-result data if not in summary
            if inp.cosine_min is None and "results" in cand:
                cosine_keys = [
                    "slot_ref_cosine", "slot_cosine", "cosine",
                    "stored_ref_cosine", "unique_ref_cosine",
                ]
                for key in cosine_keys:
                    vals = [r[key] for r in cand["results"] if key in r]
                    if vals:
                        inp.cosine_min = min(vals)
                        break

    # ── p137 ──
    p137_file = Path(p137_path) if p137_path else (
        discovered["p137"] or discovered["p137_fallback"]
    )
    if p137_file:
        p137 = _load_json(p137_file)
        if p137:
            inp.p137_report_path = str(p137_file)
            summary = p137.get("summary", {})
            inp.p137_native_full_vs_torch_slot_max_abs = summary.get(
                "native_full_vs_torch_slot_max_abs"
            )
            inp.p137_native_full_vs_torch_slot_cosine_min = summary.get(
                "native_full_vs_torch_slot_cosine_min"
            )
            inp.p137_native_full_ms_mean = summary.get("native_full_ms_mean")

    # ── strict oracle ──
    strict_file = Path(strict_path) if strict_path else (
        discovered["strict"] or discovered["strict_fallback"]
    )
    if strict_file:
        strict = _load_json(strict_file)
        if strict:
            inp.strict_report_path = str(strict_file)
            inp.strict_all_exact = strict.get("all_exact")
            inp.strict_avg_latency_ms = strict.get("avg_latency_ms")

    return inp


# ─────────────────────────────────────────────────────────────
# Gate logic
# ─────────────────────────────────────────────────────────────

def evaluate_gate(inp: GateInputs) -> GateVerdict:
    """Apply three-tier verdict rules."""
    reasons: list[str] = []
    annotations: list[str] = []

    # ── Pre-checks: required reports ──
    if inp.p136_report_path is None:
        return GateVerdict("CLOSED", False,
                           ["p136 slot-order report not found"])

    if inp.p136_verdict != "GREEN" or inp.p136_passed != P136_EXPECTED_TOTAL:
        return GateVerdict("CLOSED", False, [
            f"p136 not 18/18 GREEN: verdict={inp.p136_verdict}, "
            f"passed={inp.p136_passed}/{inp.p136_total}"
        ])

    if inp.candidate_report_path is None:
        return GateVerdict("CLOSED", False,
                           ["native candidate report not found"])

    # Extract metrics with safe defaults
    slot_abs = inp.slot_max_abs
    uniq_abs = inp.unique_max_abs
    cos_min = inp.cosine_min
    lat = inp.avg_latency_ms

    if slot_abs is None or cos_min is None or lat is None:
        return GateVerdict("CLOSED", False, [
            "candidate report missing required fields "
            f"(slot_max_abs={slot_abs}, cosine_min={cos_min}, latency={lat})"
        ])

    # ── DEFAULT check ──
    default_ok = True
    if slot_abs > DEFAULT_SLOT_MAX_ABS:
        reasons.append(f"slot_max_abs {slot_abs:.6e} > {DEFAULT_SLOT_MAX_ABS}")
        default_ok = False
    if cos_min < DEFAULT_COSINE_MIN:
        reasons.append(f"cosine_min {cos_min:.10f} < {DEFAULT_COSINE_MIN}")
        default_ok = False
    if lat > DEFAULT_LATENCY_MS:
        reasons.append(f"latency {lat:.4f}ms > {DEFAULT_LATENCY_MS}ms")
        default_ok = False

    if default_ok:
        annotations.append("eligible for default promote")
        if inp.strict_all_exact:
            annotations.append("strict oracle exact-match confirmed")
        return GateVerdict("DEFAULT", True, reasons, annotations)

    # ── AMBER check ──
    amber_ok = True
    amber_reasons: list[str] = []

    if slot_abs > AMBER_SLOT_MAX_ABS:
        amber_reasons.append(
            f"slot_max_abs {slot_abs:.6e} > {AMBER_SLOT_MAX_ABS}")
        amber_ok = False
    if uniq_abs is not None and uniq_abs > AMBER_UNIQUE_MAX_ABS:
        amber_reasons.append(
            f"unique_max_abs {uniq_abs:.6e} > {AMBER_UNIQUE_MAX_ABS}")
        amber_ok = False
    if lat > AMBER_LATENCY_MS:
        amber_reasons.append(
            f"latency {lat:.4f}ms > {AMBER_LATENCY_MS}ms")
        amber_ok = False

    if amber_ok:
        annotations.append("NO default promote — P37 exploratory only")
        annotations.append(
            f"slot_max_abs={slot_abs:.6e} exceeds DEFAULT threshold "
            f"{DEFAULT_SLOT_MAX_ABS}")
        if uniq_abs is not None:
            annotations.append(
                f"unique_max_abs={uniq_abs:.6e} (ref, not blocking)")
        if inp.strict_avg_latency_ms is not None:
            annotations.append(
                f"strict cuBLAS oracle latency={inp.strict_avg_latency_ms:.4f}ms "
                "(too slow for serving)")
        return GateVerdict("AMBER", True, reasons + amber_reasons, annotations)

    # ── CLOSED ──
    all_reasons = reasons + amber_reasons
    annotations.append("exceeds AMBER thresholds — not recommended for P37")
    return GateVerdict("CLOSED", False, all_reasons, annotations)


# ─────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────

def build_report(inp: GateInputs, verdict: GateVerdict) -> dict[str, Any]:
    """Build the full JSON report."""
    return {
        "schema": "lynn-native-moe-risk-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gate": "p140_native_moe_candidate_risk_gate",
        "inputs": {
            "p136": {
                "report": inp.p136_report_path,
                "verdict": inp.p136_verdict,
                "passed": inp.p136_passed,
                "total": inp.p136_total,
                "max_abs_max": inp.p136_max_abs_max,
            },
            "candidate": {
                "report": inp.candidate_report_path,
                "name": inp.candidate_name,
                "slot_max_abs": inp.slot_max_abs,
                "unique_max_abs": inp.unique_max_abs,
                "cosine_min": inp.cosine_min,
                "avg_latency_ms": inp.avg_latency_ms,
            },
            "p137": {
                "report": inp.p137_report_path,
                "native_full_vs_torch_slot_max_abs":
                    inp.p137_native_full_vs_torch_slot_max_abs,
                "native_full_vs_torch_slot_cosine_min":
                    inp.p137_native_full_vs_torch_slot_cosine_min,
                "native_full_ms_mean": inp.p137_native_full_ms_mean,
            },
            "strict_oracle": {
                "report": inp.strict_report_path,
                "all_exact": inp.strict_all_exact,
                "avg_latency_ms": inp.strict_avg_latency_ms,
            },
        },
        "thresholds": {
            "default": {
                "slot_max_abs": DEFAULT_SLOT_MAX_ABS,
                "cosine_min": DEFAULT_COSINE_MIN,
                "latency_ms": DEFAULT_LATENCY_MS,
            },
            "amber": {
                "slot_max_abs": AMBER_SLOT_MAX_ABS,
                "unique_max_abs": AMBER_UNIQUE_MAX_ABS,
                "latency_ms": AMBER_LATENCY_MS,
            },
        },
        "verdict": verdict.tier,
        "recommend_p37_exploratory": verdict.recommend_p37_exploratory,
        "reasons": verdict.reasons,
        "annotations": verdict.annotations,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary."""
    tier = report["verdict"]
    badge = {"DEFAULT": "🟢", "AMBER": "🟡", "CLOSED": "🔴"}.get(tier, "❓")
    inp = report["inputs"]

    lines = [
        f"# P140 Native MoE Candidate Risk Gate — {badge} {tier}",
        "",
        f"**Generated:** {report['created']}",
        "",
        "## Verdict",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Tier | **{tier}** |",
        f"| Recommend P37 exploratory | **{report['recommend_p37_exploratory']}** |",
        "",
    ]

    # Reasons
    if report["reasons"]:
        lines.append("## Reasons")
        lines.append("")
        for r in report["reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    # Annotations
    if report["annotations"]:
        lines.append("## Annotations")
        lines.append("")
        for a in report["annotations"]:
            lines.append(f"- {a}")
        lines.append("")

    # Inputs table
    lines.append("## Input Metrics")
    lines.append("")
    lines.append("| Source | Metric | Value |")
    lines.append("|--------|--------|-------|")

    p136 = inp["p136"]
    lines.append(
        f"| p136 contract | verdict | {p136['verdict']} ({p136['passed']}/{p136['total']}) |")
    lines.append(
        f"| p136 contract | max_abs_max | {p136['max_abs_max']} |")

    cand = inp["candidate"]
    if cand["slot_max_abs"] is not None:
        lines.append(
            f"| candidate ({cand['name']}) | slot_max_abs | {cand['slot_max_abs']:.6e} |")
    if cand["unique_max_abs"] is not None:
        lines.append(
            f"| candidate | unique_max_abs | {cand['unique_max_abs']:.6e} |")
    if cand["cosine_min"] is not None:
        lines.append(
            f"| candidate | cosine_min | {cand['cosine_min']:.10f} |")
    if cand["avg_latency_ms"] is not None:
        lines.append(
            f"| candidate | avg_latency_ms | {cand['avg_latency_ms']:.4f} ms |")

    strict = inp["strict_oracle"]
    if strict["avg_latency_ms"] is not None:
        lines.append(
            f"| strict oracle | avg_latency_ms | {strict['avg_latency_ms']:.4f} ms |")
    if strict["all_exact"] is not None:
        lines.append(
            f"| strict oracle | all_exact | {strict['all_exact']} |")

    p137 = inp["p137"]
    if p137["native_full_vs_torch_slot_max_abs"] is not None:
        lines.append(
            f"| p137 diagnostics | native_full_vs_torch_slot_max_abs | "
            f"{p137['native_full_vs_torch_slot_max_abs']:.6e} |")
    if p137["native_full_ms_mean"] is not None:
        lines.append(
            f"| p137 diagnostics | native_full_ms_mean | "
            f"{p137['native_full_ms_mean']:.4f} ms |")

    lines.append("")

    # Thresholds
    lines.append("## Thresholds")
    lines.append("")
    lines.append("| Tier | slot_max_abs | cosine_min | unique_max_abs | latency_ms |")
    lines.append("|------|-------------|------------|----------------|------------|")
    th = report["thresholds"]
    lines.append(
        f"| DEFAULT | ≤{th['default']['slot_max_abs']} | "
        f"≥{th['default']['cosine_min']} | — | ≤{th['default']['latency_ms']} |")
    lines.append(
        f"| AMBER | ≤{th['amber']['slot_max_abs']} | — | "
        f"≤{th['amber']['unique_max_abs']} | ≤{th['amber']['latency_ms']} |")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="P140 Native MoE candidate risk gate")
    ap.add_argument(
        "--report-dir", default="reports/qwen36_35b",
        help="Directory containing p136/candidate/p137/strict reports")
    ap.add_argument("--p136-report", default=None, help="Explicit p136 report path")
    ap.add_argument("--candidate-report", default=None, help="Explicit candidate report path")
    ap.add_argument("--p137-report", default=None, help="Explicit p137 report path")
    ap.add_argument("--strict-report", default=None, help="Explicit strict oracle report path")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <report-dir>/p140_native_moe_risk_gate.json)")
    ap.add_argument("--md-out", default=None, help="Output Markdown path")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.is_dir():
        print(f"[p140] ERROR: report directory not found: {report_dir}", file=sys.stderr)
        return 1

    inp = load_gate_inputs(
        report_dir,
        p136_path=args.p136_report,
        candidate_path=args.candidate_report,
        p137_path=args.p137_report,
        strict_path=args.strict_report,
    )

    verdict = evaluate_gate(inp)
    report = build_report(inp, verdict)

    # Write JSON
    out_json = Path(args.out or report_dir / "p140_native_moe_risk_gate.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    # Write Markdown
    out_md = Path(args.md_out or report_dir / "p140_native_moe_risk_gate.md")
    out_md.write_text(render_markdown(report))

    # Console summary
    tier = verdict.tier
    badge = {"DEFAULT": "🟢", "AMBER": "🟡", "CLOSED": "🔴"}[tier]
    print(f"\n{'='*70}")
    print(f" P140 RISK GATE: {badge} {tier}")
    print(f"{'='*70}")
    print(f" recommend_p37_exploratory: {verdict.recommend_p37_exploratory}")
    if verdict.reasons:
        print(f" reasons:")
        for r in verdict.reasons:
            print(f"   - {r}")
    if verdict.annotations:
        print(f" annotations:")
        for a in verdict.annotations:
            print(f"   - {a}")
    print(f"{'='*70}")
    print(f" JSON: {out_json}")
    print(f"  MD:  {out_md}")
    print(f"{'='*70}\n")

    return {"DEFAULT": 0, "AMBER": 0, "CLOSED": 1}[tier]


if __name__ == "__main__":
    raise SystemExit(main())
