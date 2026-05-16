#!/usr/bin/env python3
"""P69: acceptance gate for future grouped per-16 active-MoE kernels.

This gate is intentionally small and boring. P48/P56 taught us that exciting
microbench numbers are not enough. A future fused grouped per-16 kernel must
show a stronger active-MoE boundary win than the two-stage P68 reference before
we spend full-runtime gate time on it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


DEFAULT_MIN_SPEEDUP = 1.25
DEFAULT_MIN_COSINE = 0.999999
DEFAULT_MAX_REL_L2 = 0.01


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_speedups(report: dict) -> list[float]:
    out = []
    for case in report.get("cases", []):
        if "native_tile_vs_triton_speedup" in case:
            out.append(float(case["native_tile_vs_triton_speedup"]))
        elif "candidate_vs_triton_speedup" in case:
            out.append(float(case["candidate_vs_triton_speedup"]))
    return out


def _summary_speedup(report: dict) -> float | None:
    summary = report.get("summary", {})
    for key in (
        "mean_native_tile_vs_triton_speedup",
        "mean_candidate_vs_triton_speedup",
        "mean_tile_vs_triton_speedup",
    ):
        if key in summary and summary[key] is not None:
            return float(summary[key])
    vals = _case_speedups(report)
    return mean(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Candidate active-MoE JSON report")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP)
    ap.add_argument("--min-cosine", type=float, default=DEFAULT_MIN_COSINE)
    ap.add_argument("--max-rel-l2", type=float, default=DEFAULT_MAX_REL_L2)
    args = ap.parse_args()

    report_path = Path(args.report)
    report = _load(report_path)
    summary = report.get("summary", {})
    speedup = _summary_speedup(report)
    min_cosine = float(summary.get("min_cosine_vs_triton", 0.0))
    max_rel_l2 = float(summary.get("max_rel_l2_vs_triton", float("inf")))
    subkernel_pass = bool(report.get("subkernel_contract_pass", False))

    checks = {
        "subkernel_contract_pass": subkernel_pass,
        "speedup_ge_threshold": bool(speedup is not None and speedup >= args.min_speedup),
        "cosine_ge_threshold": min_cosine >= args.min_cosine,
        "rel_l2_le_threshold": max_rel_l2 <= args.max_rel_l2,
    }
    pass_gate = all(checks.values())
    result = {
        "schema_version": "lynn-engine-p69-grouped-kernel-acceptance-gate-v1",
        "candidate_report": str(report_path),
        "thresholds": {
            "min_speedup": args.min_speedup,
            "min_cosine": args.min_cosine,
            "max_rel_l2": args.max_rel_l2,
        },
        "observed": {
            "mean_speedup": speedup,
            "min_cosine_vs_triton": min_cosine,
            "max_rel_l2_vs_triton": max_rel_l2,
            "runtime_promote_in_candidate": bool(report.get("runtime_promote", False)),
        },
        "checks": checks,
        "pass": pass_gate,
        "decision": (
            "Candidate is strong enough for full-generate/server gates."
            if pass_gate
            else "Candidate is not strong enough for promotion work; keep it as kernel signal only."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
