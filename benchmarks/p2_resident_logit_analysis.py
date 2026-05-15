#!/usr/bin/env python3
"""P2-H gate: resident BF16 vs NVFP4 top-k/logit divergence analysis.

Exact token parity is intentionally strict and can fail for benign reasons when
the BF16 top logits are close. This report keeps the same resident runner but
adds per-step top-k overlap and top-1 margin diagnostics so a WARN can be
classified as quantization noise instead of a loader/kernel failure.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner


DEFAULT_PROMPTS = [
    {"id": "zh_moe_short", "prompt": "用一句话解释MoE:"},
    {"id": "en_distance", "prompt": "If a train travels 60 mph for 2.5 hours, the distance is"},
    {"id": "python_factorial", "prompt": "Python递归阶乘函数:"},
    {"id": "rope_alibi", "prompt": "比较RoPE和ALiBi:"},
]


def _run_model(model: str, prompts: list[dict[str, str]], args, dtype: torch.dtype) -> dict[str, Any]:
    runner = LynnIncrementalRunner(model, device=args.device, dtype=dtype, verbose=True)
    outputs = {}
    for item in prompts:
        outputs[item["id"]] = runner.generate(
            item["prompt"],
            max_new=args.max_new,
            top_k=args.top_k,
        )
    load_seconds = runner.load_seconds
    del runner
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {"model": model, "load_seconds": load_seconds, "outputs": outputs}


def _compare_topk(b_trace: list[dict[str, Any]], n_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for i, (b, n) in enumerate(zip(b_trace, n_trace)):
        b_ids = b["ids"]
        n_ids = n["ids"]
        overlap = len(set(b_ids) & set(n_ids))
        steps.append({
            "step": i,
            "bf16_top1": b_ids[0],
            "nvfp4_top1": n_ids[0],
            "top1_match": b_ids[0] == n_ids[0],
            "topk_overlap": overlap,
            "topk_overlap_rate": overlap / max(1, len(b_ids)),
            "bf16_top1_margin": b.get("top1_margin"),
            "nvfp4_top1_margin": n.get("top1_margin"),
            "bf16_top1_rank_in_nvfp4": (n_ids.index(b_ids[0]) + 1) if b_ids[0] in n_ids else None,
            "nvfp4_top1_rank_in_bf16": (b_ids.index(n_ids[0]) + 1) if n_ids[0] in b_ids else None,
        })
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompts = DEFAULT_PROMPTS
    print("=== BF16 top-k suite ===", flush=True)
    bf16 = _run_model(args.bf16, prompts, args, dtype)
    print("=== NVFP4 top-k suite ===", flush=True)
    nvfp4 = _run_model(args.v8, prompts, args, dtype)

    items = []
    total_overlap = 0.0
    total_steps = 0
    exact_prompts = 0
    for item in prompts:
        pid = item["id"]
        b = bf16["outputs"][pid]
        n = nvfp4["outputs"][pid]
        step_cmp = _compare_topk(b["topk_trace"], n["topk_trace"])
        total_overlap += sum(s["topk_overlap_rate"] for s in step_cmp)
        total_steps += len(step_cmp)
        exact = b["new_ids"] == n["new_ids"]
        exact_prompts += int(exact)
        items.append({
            "id": pid,
            "prompt": item["prompt"],
            "bf16_text": b["text"],
            "nvfp4_text": n["text"],
            "bf16_new_ids": b["new_ids"],
            "nvfp4_new_ids": n["new_ids"],
            "exact_new_ids_match": exact,
            "steps": step_cmp,
        })

    avg_overlap = total_overlap / max(1, total_steps)
    result = {
        "schema_version": "lynn-engine-p2-resident-logit-analysis-v1",
        "max_new": args.max_new,
        "top_k": args.top_k,
        "bf16_load_seconds": bf16["load_seconds"],
        "nvfp4_load_seconds": nvfp4["load_seconds"],
        "summary": {
            "num_prompts": len(items),
            "exact_prompt_matches": exact_prompts,
            "exact_prompt_match_rate": exact_prompts / max(1, len(items)),
            "avg_topk_overlap_rate": avg_overlap,
        },
        "items": items,
    }
    result["verdict"] = "PASS" if exact_prompts == len(items) else "ANALYSIS"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
