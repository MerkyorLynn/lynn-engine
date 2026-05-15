#!/usr/bin/env python3
"""P2-F gate: multi-prompt incremental parity between BF16 and NVFP4.

This is intentionally a correctness gate, not a performance benchmark. It runs
Lynn engine's own incremental decode path on a small prompt suite and compares
BF16 vs NVFP4 v8-RTN generated token IDs. Each prompt reloads weights so failures
are isolated and the JSON artifact is easy to inspect after an overnight run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import generate_incremental


DEFAULT_PROMPTS = [
    {
        "id": "zh_moe_short",
        "prompt": "用一句话解释MoE:",
    },
    {
        "id": "en_distance",
        "prompt": "If a train travels 60 mph for 2.5 hours, the distance is",
    },
    {
        "id": "python_factorial",
        "prompt": "Python递归阶乘函数:",
    },
    {
        "id": "rope_alibi",
        "prompt": "比较RoPE和ALiBi:",
    },
]


def _read_prompts(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_PROMPTS
    prompts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
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


def _run(model: str, prompt: str, max_new: int, device: str, dtype: torch.dtype) -> dict[str, Any]:
    text, ids = generate_incremental(
        model,
        prompt,
        max_new=max_new,
        device=device,
        dtype=dtype,
        verbose=True,
    )
    return {
        "model": model,
        "text": text,
        "new_ids": [int(x) for x in ids],
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
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompts = _read_prompts(args.prompts_jsonl)
    items = []
    exact_matches = 0
    total_token_matches = 0
    total_tokens = 0

    for prompt_item in prompts:
        print(f"\n=== {prompt_item['id']} ===", flush=True)
        bf16 = _run(args.bf16, prompt_item["prompt"], args.max_new, args.device, dtype)
        nvfp4 = _run(args.v8, prompt_item["prompt"], args.max_new, args.device, dtype)
        token_match_count = sum(a == b for a, b in zip(bf16["new_ids"], nvfp4["new_ids"]))
        exact = bf16["new_ids"] == nvfp4["new_ids"]
        exact_matches += int(exact)
        total_token_matches += token_match_count
        total_tokens += max(1, args.max_new)
        items.append({
            "id": prompt_item["id"],
            "prompt": prompt_item["prompt"],
            "bf16": bf16,
            "nvfp4": nvfp4,
            "comparison": {
                "exact_new_ids_match": exact,
                "token_match_count": token_match_count,
                "token_match_rate": token_match_count / max(1, args.max_new),
            },
        })

    exact_rate = exact_matches / max(1, len(items))
    token_match_rate = total_token_matches / max(1, total_tokens)
    result = {
        "schema_version": "lynn-engine-p2-incremental-multi-prompt-v1",
        "max_new": args.max_new,
        "device": args.device,
        "dtype": args.dtype,
        "summary": {
            "num_prompts": len(items),
            "exact_prompt_matches": exact_matches,
            "exact_prompt_match_rate": exact_rate,
            "token_match_rate": token_match_rate,
        },
        "items": items,
    }
    result["verdict"] = "PASS" if exact_matches == len(items) else "WARN"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
