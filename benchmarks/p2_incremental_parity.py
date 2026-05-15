#!/usr/bin/env python3
"""P2-E gate: incremental decode parity between BF16 and NVFP4.

This gate exercises Lynn engine's cache/recurrent-state decode path. It compares
the generated token IDs from BF16 and NVFP4 v8-RTN for the same prompt. Unlike
the brute-force gate, this path uses prefill once and then one-token decode
steps with cached full-attention KV and linear-attention recurrent state.
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
    ap.add_argument("--prompt", default="用一句话解释MoE:")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    bf16 = _run(args.bf16, args.prompt, args.max_new, args.device, dtype)
    nvfp4 = _run(args.v8, args.prompt, args.max_new, args.device, dtype)
    token_match_count = sum(a == b for a, b in zip(bf16["new_ids"], nvfp4["new_ids"]))
    result = {
        "schema_version": "lynn-engine-p2-incremental-parity-v1",
        "prompt": args.prompt,
        "max_new": args.max_new,
        "device": args.device,
        "dtype": args.dtype,
        "bf16": bf16,
        "nvfp4": nvfp4,
        "comparison": {
            "exact_new_ids_match": bf16["new_ids"] == nvfp4["new_ids"],
            "token_match_count": token_match_count,
            "token_match_rate": token_match_count / max(1, args.max_new),
        },
    }
    result["verdict"] = (
        "PASS" if result["comparison"]["exact_new_ids_match"] else "WARN"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
