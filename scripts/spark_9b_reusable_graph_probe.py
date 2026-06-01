#!/usr/bin/env python3
"""M3 step 2: reusable decode-graph parity + speedup on 9B dense FP8.

Three configs on one loaded runner:
  baseline     — eager, variable-slice attn (the 15 TPS floor)
  eager_fixed  — eager, fixed-shape attn (parity reference for the graph)
  graphed      — LYNN_REUSABLE_DECODE_GRAPH=1 (capture once, replay per token)

Gate: graphed tokens must match eager_fixed tokens (the graph just replays the
fixed-shape path). Headline: graphed_tps vs baseline_tps — the dispatch-kill.
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
    ap.add_argument("--out", default="/home/merkyor/reports/qwen35_9b/m3_reusable_graph.json")
    args = ap.parse_args()

    import torch

    os.environ.setdefault("LYNN_W4A8_FP8_PATH", "1")
    os.environ["LYNN_LINEAR_STATE_UPDATE"] = "inplace"          # required for graph replay
    os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = "1"
    os.environ["LYNN_NATIVE_FP4_LM_HEAD"] = "0"
    os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "0"
    os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "0"
    from engine.resident_runner import LynnIncrementalRunner

    prompts = [
        "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=True)

    def gen(label: str, reusable: bool, fixed: bool) -> list[dict]:
        os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "1" if reusable else "0"
        os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "1" if fixed else "0"
        rows = []
        for p in prompts:
            t = time.time()
            try:
                r = runner.generate(p, max_new=args.max_new)
                wall = time.time() - t
                rows.append({
                    "ids": [int(x) for x in r["new_ids"]],
                    "tps": (r.get("timings", {}) or {}).get("decode_tps"),
                    "wall": round(wall, 3),
                    "head": r["completion_text"][:80],
                })
                print(f"[m3-rg] {label} tps={rows[-1]['tps']} wall={rows[-1]['wall']} "
                      f":: {r['completion_text'][:46]!r}", flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                rows.append({"error": f"{type(exc).__name__}: {exc}",
                             "tb": traceback.format_exc()[-1600:]})
                print(f"[m3-rg] {label} EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
                break
        return rows

    baseline = gen("baseline", reusable=False, fixed=False)
    eager_fixed = gen("eager_fixed", reusable=False, fixed=True)
    graphed = gen("graphed", reusable=True, fixed=False)

    parity = []
    for i in range(len(prompts)):
        g = graphed[i] if i < len(graphed) else {"error": "missing"}
        e = eager_fixed[i] if i < len(eager_fixed) else {"error": "missing"}
        if "error" in g or "error" in e:
            parity.append({"prompt": prompts[i][:40], "status": "error"})
            continue
        a, b = e["ids"], g["ids"]
        n = min(len(a), len(b))
        pm = n
        for j in range(n):
            if a[j] != b[j]:
                pm = j
                break
        parity.append({
            "prompt": prompts[i][:40],
            "graph_eq_eagerfixed": a == b,
            "prefix": pm,
            "n": len(b),
            "baseline_tps": baseline[i].get("tps") if i < len(baseline) else None,
            "graphed_tps": g["tps"],
        })
        print(f"[m3-rg] PARITY graph==eagerfixed={a == b} prefix={pm}/{len(b)} "
              f"baseline_tps={parity[-1]['baseline_tps']} graphed_tps={g['tps']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"parity": parity, "baseline": baseline, "eager_fixed": eager_fixed, "graphed": graphed},
        ensure_ascii=False, indent=2))
    print(f"[m3-rg] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
