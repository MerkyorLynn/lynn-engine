#!/usr/bin/env python3
"""P184 · Qwen3.5-9B NVFP4 convstrict exact gate.

Validate the P183 `graph_plus_conv_triton` candidate on a larger structured
prompt set before it can become the 9B safe runtime profile.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from p148_qwen35_9b_nvfp4_fast_profile import (
    BASELINE_ENV,
    _compare_modes,
    _run_mode,
    _summarize_mode,
)


def _merge(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(updates)
    return out


GRAPH_EXACT_ENV = _merge(
    BASELINE_ENV,
    {
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    },
)

CONVSTRICT_ENV = _merge(
    GRAPH_EXACT_ENV,
    {
        "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
    },
)


def _load_prompts(path: Path, limit: int) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for row in rows:
        system = (row.get("system") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        if system:
            prompts.append(f"System: {system}\nUser: {prompt}")
        else:
            prompts.append(prompt)
    if limit:
        prompts = prompts[:limit]
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B convstrict exact gate.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--limit", type=int, default=70)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = _load_prompts(Path(args.prompts_json), args.limit)
    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-nvfp4-convstrict-exact-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "prompts_json": args.prompts_json,
        "limit": args.limit,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "candidate": "graph_plus_conv_triton",
    }

    reference = _run_mode(
        model=args.model,
        label="linear_graph_only_reference",
        env=GRAPH_EXACT_ENV,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    candidate = _run_mode(
        model=args.model,
        label="graph_plus_conv_triton",
        env=CONVSTRICT_ENV,
        max_new_values=[args.max_new],
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    comparison = _compare_modes(reference, candidate)
    report["reference_summary"] = _summarize_mode(reference)
    report["candidate_summary"] = _summarize_mode(candidate)
    report["comparison"] = comparison
    report["verdict"] = "CONVSTRICT_EXACT" if comparison.get("all_exact") else "CONVSTRICT_DRIFT"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "verdict": report["verdict"],
        "exact": f"{comparison.get('exact_count')}/{comparison.get('total')}",
        "candidate_decode_tps": report["candidate_summary"].get("decode_tps_mean"),
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
