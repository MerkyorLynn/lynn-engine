#!/usr/bin/env python3
"""P5-C gate: build an opt-in native FP4 projection policy.

Inputs are P5-B projection reports. Output is a policy JSON that may be used by
future runtime wiring to select `native_scaled_mm` only for explicitly accepted
linear-attention projections.

This script intentionally does not enable engine defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_fp4_policy import build_policy_from_p5_reports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-cosine", type=float, default=0.98)
    ap.add_argument("--max-rel-l2", type=float, default=0.25)
    ap.add_argument(
        "--require-speedup",
        action="store_true",
        help="Require native path to be faster than scalar bridge. Off by default because P5-C is correctness opt-in.",
    )
    args = ap.parse_args()

    policy = build_policy_from_p5_reports(
        args.reports,
        min_cosine=args.min_cosine,
        max_rel_l2=args.max_rel_l2,
        require_speedup=args.require_speedup,
    )
    policy["source_reports"] = str(args.reports)
    policy["verdict"] = "PASS" if policy["allowlist"] else "FAIL"
    if not args.require_speedup:
        slow = [
            rec for rec in policy["accepted"]
            if rec.get("speed_ratio") is not None and rec["speed_ratio"] <= 1.0
        ]
        if slow:
            policy["performance_warning"] = (
                "Accepted projections pass correctness gates but are not faster "
                "than scalar_bridge yet. Keep native FP4 disabled by default."
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0 if policy["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
