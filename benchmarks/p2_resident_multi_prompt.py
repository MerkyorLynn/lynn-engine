#!/usr/bin/env python3
"""P2-G gate: resident multi-prompt BF16 vs NVFP4 incremental parity.

This gate loads BF16 once, runs a prompt suite, releases it, then loads NVFP4
once and runs the same suite. It measures the steady-state incremental decode
path without paying the 40-layer load cost for every prompt.
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


def _read_prompts(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_PROMPTS
    prompts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get("prompt") or item.get("input") or item.get("question")
            if not prompt:
                raise ValueError(f"Prompt record {i} missing prompt/input/question")
            prompts.append({
                "id": str(item.get("id", f"prompt_{i:03d}")),
                "prompt": str(prompt),
            })
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _run_suite(
    model: str,
    prompts: list[dict[str, str]],
    *,
    max_new: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    runner = LynnIncrementalRunner(model, device=device, dtype=dtype, verbose=True)
    outputs = {}
    for item in prompts:
        outputs[item["id"]] = runner.generate(item["prompt"], max_new=max_new)
    load_seconds = runner.load_seconds
    del runner
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "model": model,
        "load_seconds": load_seconds,
        "outputs": outputs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-jsonl")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warn-threshold", type=float, default=0.75)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompts = _read_prompts(args.prompts_jsonl)

    print("=== BF16 resident suite ===", flush=True)
    bf16 = _run_suite(args.bf16, prompts, max_new=args.max_new, device=args.device, dtype=dtype)
    print("=== NVFP4 resident suite ===", flush=True)
    nvfp4 = _run_suite(args.v8, prompts, max_new=args.max_new, device=args.device, dtype=dtype)

    items = []
    exact_matches = 0
    total_token_matches = 0
    total_tokens = 0
    for item in prompts:
        pid = item["id"]
        b = bf16["outputs"][pid]
        n = nvfp4["outputs"][pid]
        token_match_count = sum(a == b for a, b in zip(b["new_ids"], n["new_ids"]))
        exact = b["new_ids"] == n["new_ids"]
        exact_matches += int(exact)
        total_token_matches += token_match_count
        total_tokens += max(1, args.max_new)
        items.append({
            "id": pid,
            "prompt": item["prompt"],
            "bf16": b,
            "nvfp4": n,
            "comparison": {
                "exact_new_ids_match": exact,
                "token_match_count": token_match_count,
                "token_match_rate": token_match_count / max(1, args.max_new),
            },
        })

    exact_rate = exact_matches / max(1, len(items))
    token_match_rate = total_token_matches / max(1, total_tokens)
    result = {
        "schema_version": "lynn-engine-p2-resident-multi-prompt-v1",
        "max_new": args.max_new,
        "device": args.device,
        "dtype": args.dtype,
        "bf16_load_seconds": bf16["load_seconds"],
        "nvfp4_load_seconds": nvfp4["load_seconds"],
        "summary": {
            "num_prompts": len(items),
            "exact_prompt_matches": exact_matches,
            "exact_prompt_match_rate": exact_rate,
            "token_match_rate": token_match_rate,
        },
        "items": items,
    }
    if exact_matches == len(items):
        result["verdict"] = "PASS"
    elif token_match_rate >= args.warn_threshold:
        result["verdict"] = "WARN"
    else:
        result["verdict"] = "FAIL"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
