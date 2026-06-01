#!/usr/bin/env python3
"""P1+graph on 35B-A3B MoE FP8: does graph-safe MoE dispatch let the reusable
decode graph capture the MoE decode, and does it beat baseline?

35B MoE decode is ~480 launches/token (vs 9B dense's few) — this is where the
reusable CUDA graph should actually pay off. Three configs on one loaded runner:
  baseline           — eager, original dynamic MoE dispatch (torch.unique loop)
  graph_safe_eager   — eager, P1 fixed-K dispatch (LYNN_FP8_MOE_GRAPH_SAFE=1)
  graphed            — P1 + reusable CUDA graph (LYNN_REUSABLE_DECODE_GRAPH=1)

Gate: graphed tokens match graph_safe_eager (graph replays the same path).
Headline: graphed_tps vs baseline_tps.
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
    ap.add_argument("--model", default="/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a8-fp8")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default="/home/merkyor/reports/qwen36_35b/m_graph_probe.json")
    args = ap.parse_args()

    import torch

    os.environ.setdefault("LYNN_W4A8_FP8_PATH", "1")
    os.environ["LYNN_LINEAR_STATE_UPDATE"] = "inplace"
    os.environ["LYNN_LINEAR_ATTN_RECURRENT_INPLACE"] = "1"
    os.environ["LYNN_NATIVE_FP4_LM_HEAD"] = "0"
    os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "0"
    os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "0"
    os.environ["LYNN_FP8_MOE_GRAPH_SAFE"] = "0"
    from engine.resident_runner import LynnIncrementalRunner

    prompts = [
        "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
    ]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=True)

    def gen(label: str, moe_gs: bool, reusable: bool) -> list[dict]:
        os.environ["LYNN_FP8_MOE_GRAPH_SAFE"] = "1" if moe_gs else "0"
        os.environ["LYNN_REUSABLE_DECODE_GRAPH"] = "1" if reusable else "0"
        os.environ["LYNN_FULL_ATTN_FIXED_SHAPE"] = "0"  # capture forces it; eager stays off
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
                print(f"[m-35b] {label} tps={rows[-1]['tps']} wall={rows[-1]['wall']} "
                      f":: {r['completion_text'][:46]!r}", flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback
                rows.append({"error": f"{type(exc).__name__}: {exc}",
                             "tb": traceback.format_exc()[-1800:]})
                print(f"[m-35b] {label} EXCEPTION: {type(exc).__name__}: {exc}", flush=True)
                break
        return rows

    baseline = gen("baseline", moe_gs=False, reusable=False)
    gs_eager = gen("graph_safe_eager", moe_gs=True, reusable=False)
    graphed = gen("graphed", moe_gs=True, reusable=True)

    parity = []
    for i in range(len(prompts)):
        g = graphed[i] if i < len(graphed) else {"error": "missing"}
        e = gs_eager[i] if i < len(gs_eager) else {"error": "missing"}
        if "error" in g or "error" in e:
            parity.append({"prompt": prompts[i][:40], "status": "error"})
            continue
        a, b = e["ids"], g["ids"]
        n = min(len(a), len(b))
        pm = next((j for j in range(n) if a[j] != b[j]), n)
        parity.append({
            "prompt": prompts[i][:40],
            "graph_eq_gseager": a == b,
            "prefix": pm,
            "n": len(b),
            "baseline_tps": baseline[i].get("tps") if i < len(baseline) and "tps" in baseline[i] else None,
            "gseager_tps": e.get("tps"),
            "graphed_tps": g["tps"],
        })
        print(f"[m-35b] PARITY graph==gseager={a == b} prefix={pm}/{len(b)} "
              f"baseline_tps={parity[-1]['baseline_tps']} gseager_tps={e.get('tps')} "
              f"graphed_tps={g['tps']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"parity": parity, "baseline": baseline, "graph_safe_eager": gs_eager, "graphed": graphed},
        ensure_ascii=False, indent=2))
    print(f"[m-35b] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
