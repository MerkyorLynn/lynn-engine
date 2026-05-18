#!/usr/bin/env python3
"""Summarize a Qwen3.6 MoE fixture gate report.

This is the small traffic light that sits between P134 and expensive service
gates.  Native MoE candidates should first pass the fixture contract; only
numerically clean and meaningfully faster candidates deserve P37/P25 time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_numeric(mapping: dict[str, Any], paths: list[str]) -> float | None:
    for path in paths:
        cur: Any = mapping
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is not None:
            try:
                return float(cur)
            except (TypeError, ValueError):
                pass
    return None


def classify(
    report: dict[str, Any],
    *,
    min_speedup: float,
    strict_exact: bool,
    reference_report: dict[str, Any] | None = None,
    candidate_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = int(report.get("total") or 0)
    passed = int(report.get("passed") or 0)
    failed = int(report.get("failed") or max(total - passed, 0))
    verdict = report.get("verdict", "UNKNOWN")
    summary = report.get("summary") or {}

    exact_count = int(summary.get("exact_count") or 0)
    max_abs = _as_float(summary.get("max_abs_max"))
    rel_l2 = _as_float(summary.get("rel_l2_max"))
    cosine_min = _as_float(summary.get("cosine_min"), 0.0)
    ref_ms = _as_float(summary.get("ref_ms_mean"))
    cand_ms = _as_float(summary.get("candidate_ms_mean"))

    if reference_report and ref_ms <= 0.0:
        ref_summary = reference_report.get("summary") or {}
        ref_override = _first_numeric(
            ref_summary,
            ["ref_ms_mean", "candidate_ms_mean", "mean_ms", "latency_ms"],
        )
        if ref_override is not None:
            ref_ms = ref_override

    if candidate_metrics and cand_ms <= 0.0:
        cand_override = _first_numeric(
            candidate_metrics,
            [
                "candidate_ms_mean",
                "mean_ms",
                "latency_ms",
                "ms_mean",
                "summary.candidate_ms_mean",
                "summary.mean_ms",
                "summary.latency_ms",
                "candidate.candidate_ms_mean",
                "candidate.mean_ms",
            ],
        )
        if cand_override is not None:
            cand_ms = cand_override

    has_candidate_timing = cand_ms > 0.0
    speedup = (ref_ms / cand_ms) if has_candidate_timing else None

    candidate_backend = report.get("candidate_backend")
    candidate_output_dir = report.get("candidate_output_dir")
    routed_only = bool(report.get("routed_only"))

    reasons: list[str] = []
    if verdict != "GREEN" or failed > 0 or passed != total:
        decision = "CLOSED_NUMERIC"
        reasons.append(f"contract not green ({passed}/{total} passed)")
    elif strict_exact and exact_count != total:
        decision = "CLOSED_NUMERIC"
        reasons.append(f"strict exact required but exact_count={exact_count}/{total}")
    elif not candidate_backend and not candidate_output_dir:
        decision = "BASELINE_REFERENCE"
        reasons.append("self-check report; use as baseline reference")
    elif not has_candidate_timing:
        decision = "PASS_NUMERIC_ONLY"
        reasons.append("candidate output is numerically clean but has no latency timing")
    elif speedup is not None and speedup >= min_speedup:
        decision = "FAST_CANDIDATE"
        reasons.append(f"candidate speedup {speedup:.3f}x >= {min_speedup:.3f}x")
    else:
        decision = "PASS_SLOW"
        if speedup is None:
            reasons.append("candidate passed but no speedup is available")
        else:
            reasons.append(f"candidate speedup {speedup:.3f}x < {min_speedup:.3f}x")

    return {
        "schema": "lynn-moe-fixture-gate-summary-v1",
        "source_report": report.get("_source_report"),
        "decision": decision,
        "reasons": reasons,
        "mode": "routed_only" if routed_only else "full_moe",
        "candidate_backend": candidate_backend,
        "candidate_output_dir": candidate_output_dir,
        "total": total,
        "passed": passed,
        "failed": failed,
        "exact_count": exact_count,
        "max_abs_max": max_abs,
        "rel_l2_max": rel_l2,
        "cosine_min": cosine_min,
        "ref_ms_mean": ref_ms,
        "candidate_ms_mean": cand_ms,
        "reference_report": reference_report.get("_source_report") if reference_report else None,
        "candidate_metrics": candidate_metrics.get("_source_report") if candidate_metrics else None,
        "speedup": speedup,
        "min_speedup": min_speedup,
        "strict_exact": strict_exact,
    }


def print_human(summary: dict[str, Any]) -> None:
    print("═══════════════════════════════════════════════════════════════════════")
    print(" Qwen3.6 MoE Fixture Gate Summary")
    print("═══════════════════════════════════════════════════════════════════════")
    print(f" Decision:    {summary['decision']}")
    print(f" Mode:        {summary['mode']}")
    print(f" Passed:      {summary['passed']}/{summary['total']}")
    print(f" Exact:       {summary['exact_count']}/{summary['total']}")
    print(f" max_abs:     {summary['max_abs_max']:.6e}")
    print(f" rel_l2:      {summary['rel_l2_max']:.6e}")
    print(f" cosine_min:  {summary['cosine_min']:.9f}")
    print(f" ref_ms:      {summary['ref_ms_mean']:.6f}")
    print(f" cand_ms:     {summary['candidate_ms_mean']:.6f}")
    if summary["speedup"] is None:
        print(" speedup:     n/a")
    else:
        print(f" speedup:     {summary['speedup']:.3f}x")
    print(" Reasons:")
    for reason in summary["reasons"]:
        print(f"  - {reason}")
    print("═══════════════════════════════════════════════════════════════════════")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", help="P134 JSON report path")
    ap.add_argument("--out", default=None, help="Optional summary JSON output path")
    ap.add_argument(
        "--min-speedup",
        type=float,
        default=1.05,
        help="Minimum fixture-level speedup required for FAST_CANDIDATE.",
    )
    ap.add_argument(
        "--allow-nonexact",
        action="store_true",
        help="Do not close candidates solely because exact_count < total.",
    )
    ap.add_argument(
        "--reference-report",
        default=None,
        help="Optional p134 self-check report to provide ref_ms_mean when the "
             "candidate report only contains precomputed outputs.",
    )
    ap.add_argument(
        "--candidate-metrics",
        default=None,
        help="Optional JSON with candidate latency, e.g. candidate_ms_mean or mean_ms.",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress human output")
    args = ap.parse_args()

    path = Path(args.report)
    report = json.loads(path.read_text())
    report["_source_report"] = str(path)
    reference_report = None
    if args.reference_report:
        reference_path = Path(args.reference_report)
        reference_report = json.loads(reference_path.read_text())
        reference_report["_source_report"] = str(reference_path)
    candidate_metrics = None
    if args.candidate_metrics:
        metrics_path = Path(args.candidate_metrics)
        candidate_metrics = json.loads(metrics_path.read_text())
        candidate_metrics["_source_report"] = str(metrics_path)
    summary = classify(
        report,
        min_speedup=args.min_speedup,
        strict_exact=not args.allow_nonexact,
        reference_report=reference_report,
        candidate_metrics=candidate_metrics,
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if not args.quiet:
        print_human(summary)

    return 0 if not summary["decision"].startswith("CLOSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
