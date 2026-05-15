#!/usr/bin/env python3
"""P10-A: native FP4 fused linear-attention input projection probe.

Earlier P5 probes replaced individual projections with native FP4
`torch._scaled_mm`; that lost to the BF16 resident path because each projection
paid activation quantization and dispatch overhead separately.

This probe tests the production-shaped variant: concatenate the packed NVFP4
`qkv/z/b/a` input projections and run one activation quantization plus one
native FP4 matmul. If this does not beat the current BF16 fused in-projection,
native FP4 must move to deeper fused kernels rather than another wrapper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402
from engine.nvfp4_runtime import (  # noqa: E402
    _compact_scale_to_swizzled_fp8,
    load_packed_nvfp4_linear,
)
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


PROJ_NAMES = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
]


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
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


def _native_fused_projection(x_2d: torch.Tensor, packed: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    act_packed, scale_a = quantize_fp4_m1_native(x_2d)
    return torch._scaled_mm(
        act_packed.view(torch.float4_e2m1fn_x2),
        packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    layer_w, inferred = load_qwen36_layer(
        args.model,
        args.layer,
        device=device,
        dequant_dtype=dtype,
    )
    bf16_weight = torch.cat(
        [layer_w[f"{name}.weight"] for name in PROJ_NAMES],
        dim=0,
    ).contiguous()

    packed_parts = []
    effective_scale_parts = []
    for name in PROJ_NAMES:
        base = f"model.language_model.layers.{args.layer}.{name}"
        lin = load_packed_nvfp4_linear(args.model, base, name=base, device=device)
        packed_parts.append(lin.weight_packed)
        effective_scale_parts.append(
            lin.weight_scale.float() / lin.weight_global_scale.to(device).float()
        )

    packed = torch.cat(packed_parts, dim=0).contiguous()
    effective_scale = torch.cat(effective_scale_parts, dim=0).contiguous()
    out_features = int(packed.shape[0])
    in_features = int(packed.shape[1] * 2)
    scale_b = _compact_scale_to_swizzled_fp8(effective_scale, outer_dim=out_features, k=in_features)

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    x = torch.randn(1, in_features, device=device, dtype=dtype, generator=gen)

    ref = F.linear(x, bf16_weight)
    native = _native_fused_projection(x, packed, scale_b).to(dtype)
    diff = (native.float() - ref.float()).abs()
    cosine = F.cosine_similarity(native.float().flatten(), ref.float().flatten(), dim=0).item()

    # Split quantization from matmul so we can see whether the target bottleneck
    # is activation quantization, scaled_mm, or wrapper composition.
    act_packed, scale_a = quantize_fp4_m1_native(x)

    def bf16_fused() -> torch.Tensor:
        return F.linear(x, bf16_weight)

    def native_quant_plus_mm() -> torch.Tensor:
        return _native_fused_projection(x, packed, scale_b)

    def native_mm_only() -> torch.Tensor:
        return torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            packed.view(torch.float4_e2m1fn_x2).t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )

    def quant_only() -> torch.Tensor:
        return quantize_fp4_m1_native(x)[0]

    timing = {
        "bf16_fused_linear_ms": _bench(bf16_fused, args.warmup, args.iters),
        "native_fp4_quant_plus_scaled_mm_ms": _bench(native_quant_plus_mm, args.warmup, args.iters),
        "native_fp4_scaled_mm_only_ms": _bench(native_mm_only, args.warmup, args.iters),
        "activation_quant_only_ms": _bench(quant_only, args.warmup, args.iters),
    }
    timing["native_vs_bf16_ratio"] = timing["bf16_fused_linear_ms"] / timing["native_fp4_quant_plus_scaled_mm_ms"]
    timing["native_mm_only_vs_bf16_ratio"] = timing["bf16_fused_linear_ms"] / timing["native_fp4_scaled_mm_only_ms"]

    result = {
        "schema_version": "lynn-engine-p10a-native-fp4-fused-inproj-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "inferred": inferred,
        "shape": {
            "in_features": in_features,
            "out_features": out_features,
            "bf16_weight": list(bf16_weight.shape),
            "packed": list(packed.shape),
            "scale_b": list(scale_b.shape),
        },
        "diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(native.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(cosine),
        },
        "timing_ms": timing,
        "pass": bool(cosine >= 0.98),
        "notes": [
            "This is the first native fused FP4 projection shape, not full decode integration.",
            "A speed ratio > 1 means native quant+scaled_mm beats current BF16 fused in-proj.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
