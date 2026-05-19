#!/usr/bin/env python3
"""P190 · Qwen3.5-9B true-FP8 dense resident gate.

P187 proved the fixture-level tensorwise fused-gate/up FP8 path is faster than
BF16 dense FFN.  P190 checks whether the same opt-in resident path preserves
real greedy generation against the current safe convstrict profile before any
P150/service promotion.

P197 integration: if a drift probe report is provided, its verdict overrides
the simple exact-match decision.  AMBER from P197 cannot silently become the
default — it stays AMBER_NUMERIC and requires explicit review.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from p148_qwen35_9b_nvfp4_fast_profile import _compare_modes, _run_mode, _summarize_mode
from p184_qwen35_9b_nvfp4_convstrict_exact_gate import CONVSTRICT_ENV, _load_prompts


def _merge(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(updates)
    return out


TRUE_FP8_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar",
    },
)


def _load_p197_verdict(path: str) -> dict[str, Any] | None:
    """Load a P197 drift probe report and return its verdict + reasons."""
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "verdict": data.get("verdict"),
        "reasons": data.get("reasons", []),
        "first_drift_prompt": data.get("comparison", {}).get("first_drift_prompt"),
        "first_drift_step": data.get("comparison", {}).get("first_drift_step"),
        "drift_ratio": data.get("comparison", {}).get("drift_ratio"),
        "topk_combined_min": data.get("comparison", {}).get("topk_combined_min"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B true-FP8 dense resident exact gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--p197-report", default=None,
                    help="Path to P197 drift probe report JSON (optional)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = _load_prompts(Path(args.prompts_json), args.limit)
    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-true-fp8-resident-gate-v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "prompts_json": args.prompts_json,
        "limit": args.limit,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "reference": "convstrict_w4a16",
        "candidate": "convstrict_true_fp4xfp8_dense_ffn",
    }

    # Load P197 drift signal if available
    p197 = _load_p197_verdict(args.p197_report) if args.p197_report else None
    if p197:
        report["p197_drift"] = p197

    reference = _run_mode(
        model=args.model,
        label="convstrict_w4a16_reference",
        env=CONVSTRICT_ENV,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    candidate = _run_mode(
        model=args.model,
        label="convstrict_true_fp4xfp8_dense_ffn",
        env=TRUE_FP8_ENV,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    comparison = _compare_modes(reference, candidate)
    report["reference_summary"] = _summarize_mode(reference)
    report["candidate_summary"] = _summarize_mode(candidate)
    report["comparison"] = comparison

    # Determine verdict
    # P197 drift signal takes precedence over simple exact match
    if p197 and p197["verdict"] == "CLOSED_NUMERIC":
        report["verdict"] = "CLOSED_NUMERIC"
        report["verdict_reasons"] = [
            "P197 drift probe: CLOSED_NUMERIC",
            *p197.get("reasons", []),
        ]
    elif p197 and p197["verdict"] == "AMBER_NUMERIC":
        report["verdict"] = "AMBER_NUMERIC"
        report["verdict_reasons"] = [
            "P197 drift probe: AMBER_NUMERIC — requires explicit review, not default",
            *p197.get("reasons", []),
        ]
    elif comparison.get("all_exact"):
        report["verdict"] = "STRICT_CANDIDATE"
        report["verdict_reasons"] = [
            "all prompts: identical greedy tokens",
            *(["P197: STRICT_CANDIDATE"] if p197 else []),
        ]
    elif (comparison.get("exact_count") or 0) == 0:
        report["verdict"] = "CLOSED_NUMERIC"
        report["verdict_reasons"] = [
            f"exact_count=0/{comparison.get('total')}",
            *(["P197: " + p197["verdict"]] if p197 else []),
        ]
    else:
        # Partial exact — check P197 if available
        if p197 and p197["verdict"] == "STRICT_CANDIDATE":
            report["verdict"] = "STRICT_CANDIDATE"
            report["verdict_reasons"] = [
                f"exact_count={comparison.get('exact_count')}/{comparison.get('total')} "
                "but P197 token-level: STRICT_CANDIDATE",
            ]
        else:
            report["verdict"] = "AMBER_NUMERIC"
            report["verdict_reasons"] = [
                f"exact_count={comparison.get('exact_count')}/{comparison.get('total')} "
                "— requires explicit review, not default",
            ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "verdict_reasons": report.get("verdict_reasons", []),
        "exact": f"{comparison.get('exact_count')}/{comparison.get('total')}",
        "p197": p197["verdict"] if p197 else "not_provided",
        "reference_decode_tps": report["reference_summary"].get("decode_tps_mean"),
        "candidate_decode_tps": report["candidate_summary"].get("decode_tps_mean"),
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
