#!/usr/bin/env python3
"""P4-E hot-path profile for the opt-in native FP4 runtime backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import (  # noqa: E402
    PackedNVFP4Linear,
    _compact_scale_to_swizzled_fp8,
    _quantize_activation_to_fp4,
)


def _bench(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]], *, warmup: int, iters: int) -> float:
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
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weight", default="linear_attn.in_proj_qkv.weight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    v8_dir = Path(args.v8)
    base = f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        linear = PackedNVFP4Linear.from_safetensors(st, base, name=args.weight, device=args.device)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    x = torch.randn((1, linear.in_features), device=args.device, dtype=torch.bfloat16, generator=gen)

    # Warm persistent weight-scale cache before profiling activation-side work.
    scale_b = linear._native_scale_b()
    act_packed, act_scale = _quantize_activation_to_fp4(x)
    scale_a = _compact_scale_to_swizzled_fp8(act_scale, outer_dim=1, k=linear.in_features)

    def quantize():
        return _quantize_activation_to_fp4(x)

    def swizzle_a():
        return _compact_scale_to_swizzled_fp8(act_scale, outer_dim=1, k=linear.in_features)

    def scaled_mm_only():
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            linear.weight_packed.view(torch.float4_e2m1fn_x2).t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )

    def end_to_end_native():
        return linear(x[0], backend="native_scaled_mm")

    def scalar_bridge():
        return linear(x[0], backend="scalar_bridge")

    result = {
        "schema_version": "lynn-engine-p4-native-fp4-hotpath-profile-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "weight": args.weight,
        "shape": {
            "m": 1,
            "n": linear.out_features,
            "k": linear.in_features,
        },
        "latency_ms": {
            "activation_quantize": _bench(quantize, warmup=args.warmup, iters=args.iters),
            "scale_a_swizzle": _bench(swizzle_a, warmup=args.warmup, iters=args.iters),
            "scaled_mm_only": _bench(scaled_mm_only, warmup=args.warmup, iters=args.iters),
            "native_scaled_mm_end_to_end": _bench(end_to_end_native, warmup=args.warmup, iters=args.iters),
            "scalar_bridge": _bench(scalar_bridge, warmup=args.warmup, iters=args.iters),
        },
        "notes": [
            "weight scale_b is cached before profiling.",
            "scale_a swizzle uses cached index tensors after first call.",
            "P5 should target the largest remaining hot-path component.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
