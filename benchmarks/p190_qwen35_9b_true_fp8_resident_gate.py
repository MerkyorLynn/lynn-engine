#!/usr/bin/env python3
"""P190 · Qwen3.5-9B true-FP8 dense resident gate.

P187 proved the fixture-level tensorwise fused-gate/up FP8 path is faster than
BF16 dense FFN.  P190 checks whether the same opt-in resident path preserves
real greedy generation against the current safe convstrict profile before any
P150/service promotion.
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B true-FP8 dense resident exact gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = _load_prompts(Path(args.prompts_json), args.limit)
    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-true-fp8-resident-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "prompts_json": args.prompts_json,
        "limit": args.limit,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "reference": "convstrict_w4a16",
        "candidate": "convstrict_true_fp4xfp8_dense_ffn",
    }

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
    if comparison.get("all_exact"):
        report["verdict"] = "TRUE_FP8_RESIDENT_EXACT"
    elif (comparison.get("exact_count") or 0) == 0:
        report["verdict"] = "TRUE_FP8_RESIDENT_RED"
    else:
        report["verdict"] = "TRUE_FP8_RESIDENT_AMBER_DRIFT"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "exact": f"{comparison.get('exact_count')}/{comparison.get('total')}",
        "reference_decode_tps": report["reference_summary"].get("decode_tps_mean"),
        "candidate_decode_tps": report["candidate_summary"].get("decode_tps_mean"),
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
