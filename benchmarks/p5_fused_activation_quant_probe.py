#!/usr/bin/env python3
"""P5-A probe: Triton fused activation quantization for native FP4 decode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import (  # noqa: E402
    PackedNVFP4Linear,
    _compact_scale_to_swizzled_fp8,
    _quantize_activation_to_fp4,
)
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(cosine),
    }


def _bench(fn, *, warmup: int, iters: int) -> float:
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

    ref_packed, ref_compact_scale = _quantize_activation_to_fp4(x)
    ref_scale_a = _compact_scale_to_swizzled_fp8(ref_compact_scale, outer_dim=1, k=linear.in_features)
    tri_packed, tri_scale_a = quantize_fp4_m1_native(x)

    scale_b = linear._native_scale_b()
    ref_out = torch._scaled_mm(
        ref_packed.view(torch.float4_e2m1fn_x2),
        linear.weight_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=ref_scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    ).float()
    tri_out = torch._scaled_mm(
        tri_packed.view(torch.float4_e2m1fn_x2),
        linear.weight_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=tri_scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    ).float()

    packed_mismatch_bytes = int((ref_packed != tri_packed).sum().item())
    packed_match = packed_mismatch_bytes == 0
    scale_cmp = _compare(tri_scale_a.float(), ref_scale_a.float())
    out_cmp = _compare(tri_out, ref_out)

    result = {
        "schema_version": "lynn-engine-p5-fused-activation-quant-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "weight": args.weight,
        "shape": {
            "m": 1,
            "k": linear.in_features,
            "packed_shape": list(tri_packed.shape),
            "scale_a_len": int(tri_scale_a.numel()),
        },
        "packed_match": packed_match,
        "packed_mismatch_bytes": packed_mismatch_bytes,
        "packed_mismatch_rate": packed_mismatch_bytes / tri_packed.numel(),
        "scale_comparison": scale_cmp,
        "output_comparison": out_cmp,
        "latency_ms": {
            "torch_reference_quantize_plus_swizzle": _bench(
                lambda: _compact_scale_to_swizzled_fp8(
                    _quantize_activation_to_fp4(x)[1],
                    outer_dim=1,
                    k=linear.in_features,
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "triton_fused_quantize": _bench(
                lambda: quantize_fp4_m1_native(x),
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "verdict": "PASS" if out_cmp["cosine"] >= 0.9999 and out_cmp["rel_l2"] <= 0.01 else "FAIL",
        "notes": [
            "Packed bytes may differ at rare midpoint ties; output equivalence is the correctness gate.",
            "Scale_a is expected to match exactly.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
