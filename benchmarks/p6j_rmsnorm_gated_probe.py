#!/usr/bin/env python3
"""P6-J: RMSNormGated Triton correctness and microbench probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.qwen36_linear_attn_block import rms_norm_gated  # noqa: E402
from triton_kernels.rmsnorm_gated import HAS_TRITON, rms_norm_gated_triton  # noqa: E402


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.rows, args.dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(args.rows, args.dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.dim, device="cuda", dtype=torch.bfloat16) * 0.05

    ref = rms_norm_gated(x, weight, gate)
    tri = rms_norm_gated_triton(x, weight, gate)
    result = {
        "schema_version": "lynn-engine-p6j-rmsnorm-gated-probe-v1",
        "device": torch.cuda.get_device_name("cuda"),
        "has_triton": HAS_TRITON,
        "shape": [args.rows, args.dim],
        "max_abs": float((tri.float() - ref.float()).abs().max().item()),
        "cosine": float(F.cosine_similarity(tri.float().flatten(), ref.float().flatten(), dim=0).item()),
        "latency_ms": {
            "torch_ref": _bench(lambda: rms_norm_gated(x, weight, gate), args.warmup, args.iters),
            "triton": _bench(lambda: rms_norm_gated_triton(x, weight, gate), args.warmup, args.iters),
        },
    }
    result["speedup"] = result["latency_ms"]["torch_ref"] / result["latency_ms"]["triton"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
