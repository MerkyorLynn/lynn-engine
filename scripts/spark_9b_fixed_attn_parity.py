#!/usr/bin/env python3
"""M3 step 1: token parity of LYNN_FULL_ATTN_FIXED_SHAPE on 9B dense FP8.

The fixed-shape full-attn path (index_copy_ KV write + full-cache masked SDPA)
must match the variable-slice path token-for-token before it can be wired into a
capture-once/replay-many decode graph. Eager fixed-shape is EXPECTED to be a bit
slower (it attends over the full max_T window each token); the speed payoff comes
only after graph capture. This probe checks correctness, not speed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/merkyor/models/Qwen3.5-9B-lynn-native-w4a8-fp8")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default="/home/merkyor/reports/qwen35_9b/m3_fixed_attn_parity.json")
    args = ap.parse_args()

    import torch

    os.environ.setdefault("LYNN_W4A8_FP8_PATH", "1")
    os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "0"
    from engine.resident_runner import LynnIncrementalRunner

    prompts = [
        "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=True)

    def gen(fixed: bool) -> list[dict]:
        os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "1" if fixed else "0"
        rows = []
        for p in prompts:
            t = time.time()
            r = runner.generate(p, max_new=args.max_new)
            wall = time.time() - t
            rows.append({
                "ids": [int(x) for x in r["new_ids"]],
                "tps": (r.get("timings", {}) or {}).get("decode_tps"),
                "wall": round(wall, 3),
                "head": r["completion_text"][:90],
            })
        return rows

    base = gen(False)
    fixed = gen(True)

    rows = []
    for i, p in enumerate(prompts):
        a, b = base[i]["ids"], fixed[i]["ids"]
        n = min(len(a), len(b))
        pm = n
        for j in range(n):
            if a[j] != b[j]:
                pm = j
                break
        rows.append({
            "prompt": p[:48],
            "exact": a == b,
            "prefix_match": pm,
            "n": len(a),
            "base_tps": base[i]["tps"],
            "fixed_tps": fixed[i]["tps"],
        })
        print(
            f"[m3-parity] exact={a == b} prefix={pm}/{len(a)} "
            f"base_tps={base[i]['tps']} fixed_tps={fixed[i]['tps']} "
            f":: {fixed[i]['head']!r}",
            flush=True,
        )

    all_exact = all(r["exact"] for r in rows)
    min_prefix = min(r["prefix_match"] for r in rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"all_exact": all_exact, "min_prefix": min_prefix, "rows": rows,
                    "base": base, "fixed": fixed}, ensure_ascii=False, indent=2)
    )
    print(f"[m3-parity] wrote {args.out}", flush=True)
    print(f"[m3-parity] ALL_EXACT={all_exact} MIN_PREFIX={min_prefix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
