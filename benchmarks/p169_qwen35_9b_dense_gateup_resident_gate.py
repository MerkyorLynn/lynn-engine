#!/usr/bin/env python3
"""P169 · Qwen3.5-9B resident dense gate/up fusion gate.

P168 proved that concatenating dense FFN gate/up weights is exact and slightly
faster on isolated fixtures.  P169 runs the real resident decode path with the
existing 9B fast profile, comparing:

* baseline: current linear-graph profile;
* candidate: same profile plus ``LYNN_DENSE_FFN_GATE_UP_FUSED=1``.

The gate is exact-first: any greedy token drift closes the candidate before
service-level promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p148_qwen35_9b_nvfp4_fast_profile import (  # noqa: E402
    BASELINE_ENV,
    PROMPTS,
    _compare_modes,
    _run_mode,
    _summarize_mode,
)


def _candidate_decision(comparison: dict[str, Any]) -> str:
    if not comparison.get("all_exact"):
        return "DENSE_GATEUP_RESIDENT_CLOSED_NUMERIC"
    speedup = float(comparison.get("speedup_mean") or 0.0)
    if speedup < 1.005:
        return "DENSE_GATEUP_RESIDENT_CLOSED_FLAT"
    return "DENSE_GATEUP_RESIDENT_CANDIDATE"


def _linear_graph_env() -> dict[str, str]:
    env = dict(BASELINE_ENV)
    env.update({
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    })
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate resident dense FFN fused gate/up.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = args.prompt or PROMPTS
    baseline_env = _linear_graph_env()
    baseline_env["LYNN_DENSE_FFN_GATE_UP_FUSED"] = "0"
    candidate_env = _linear_graph_env()
    candidate_env["LYNN_DENSE_FFN_GATE_UP_FUSED"] = "1"

    baseline = _run_mode(
        model=args.model,
        label="linear_graph_default",
        env=baseline_env,
        max_new_values=args.max_new,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    candidate = _run_mode(
        model=args.model,
        label="dense_gateup_fused",
        env=candidate_env,
        max_new_values=args.max_new,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    comparison = _compare_modes(baseline, candidate)
    decision = _candidate_decision(comparison)
    report = {
        "schema": "lynn-qwen35-9b-dense-gateup-resident-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "max_new_values": args.max_new,
        "max_seq_len": args.max_seq_len,
        "prompts": prompts,
        "modes": [baseline, candidate],
        "summaries": [_summarize_mode(baseline), _summarize_mode(candidate)],
        "comparison": comparison,
        "decision": decision,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "summaries": report["summaries"],
        "comparison": {
            "exact_count": comparison.get("exact_count"),
            "total": comparison.get("total"),
            "speedup_mean": comparison.get("speedup_mean"),
            "speedup_min": comparison.get("speedup_min"),
        },
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0 if comparison.get("all_exact") else 2


if __name__ == "__main__":
    raise SystemExit(main())
