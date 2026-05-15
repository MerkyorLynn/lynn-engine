#!/usr/bin/env python3
"""Minimal CLI for Lynn engine's resident incremental runner.

This is the bridge between correctness benchmarks and a future HTTP server:
load one model once, run one or more prompts through Lynn engine's own
incremental decode path, and emit a machine-readable JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner


def _read_prompts(args) -> list[dict[str, str]]:
    if args.prompts_jsonl:
        prompts = []
        with open(args.prompts_jsonl, encoding="utf-8") as f:
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
        if prompts:
            return prompts
    return [{"id": "prompt_000", "prompt": args.prompt}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用一句话解释MoE:")
    ap.add_argument("--prompts-jsonl")
    ap.add_argument("--out")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument(
        "--chat-template",
        action="store_true",
        help="encode prompts as a no-think chat turn instead of raw text",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prompts = _read_prompts(args)
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=dtype,
        verbose=not args.quiet,
    )
    items = []
    for item in prompts:
        output = runner.generate(
            item["prompt"],
            max_new=args.max_new,
            top_k=args.top_k,
            use_chat_template=args.chat_template,
        )
        items.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "output": output,
        })

    result = {
        "schema_version": "lynn-engine-resident-cli-v1",
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "max_new": args.max_new,
        "chat_template": args.chat_template,
        "load_seconds": runner.load_seconds,
        "cuda_memory_after_load": runner.cuda_memory_after_load,
        "items": items,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
