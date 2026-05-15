#!/usr/bin/env python3
"""P4-A feasibility probe: PyTorch native FP4 scaled_mm on Blackwell.

This does not yet reproduce NVFP4 checkpoint numerics. Its job is narrower:

1. prove R6000's torch build exposes `float4_e2m1fn_x2`;
2. prove existing `weight_packed` bytes can be viewed as torch float4 storage;
3. prove `torch._scaled_mm` runs FP4 blockwise 1x16 on sm_120;
4. capture the scale-layout contract needed for the real NVFP4 repack step.

P3 remains the correctness oracle. P4-A is the first native FP4 GEMM feasibility
gate before writing a production kernel/repacker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import PackedNVFP4Linear
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed


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


def _scale_len(m: int, n: int, k: int) -> tuple[int, int]:
    """Empirical PyTorch 2.10 FP4 blockwise-1x16 scale lengths.

    `_scaled_mm` reports these lengths for float4_e2m1fn_x2 with logical K.
    cuBLASLt rounds the outer M/N dimensions up to at least 128 rows/cols for
    this scale layout. This is not the NVFP4 checkpoint's compact
    `[out, K/16]` scale layout; it is the expanded torch runtime contract.
    """
    return max(m, 128) * (k // 16), max(n, 128) * (k // 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weight", default="linear_attn.in_proj_qkv.weight")
    ap.add_argument("--out-features", type=int, default=16, help="N slice to probe")
    ap.add_argument("--tokens", type=int, default=1, help="M dimension")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("torch.float4_e2m1fn_x2 is unavailable")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable")

    v8_dir = Path(args.v8)
    base = f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        packed = PackedNVFP4Linear.from_safetensors(st, base, name=args.weight, device=args.device)

    n = min(args.out_features, packed.out_features)
    k = packed.in_features
    m = args.tokens
    weight_u8 = packed.weight_packed[:n].contiguous()
    weight_fp4 = weight_u8.view(torch.float4_e2m1fn_x2)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    # A synthetic FP4 activation matrix proves the native FP4 GEMM path. Real
    # activation quantization/repack is the next P4 step.
    act_u8 = torch.randint(0, 256, (m, k // 2), device=args.device, dtype=torch.uint8, generator=gen)
    act_fp4 = act_u8.view(torch.float4_e2m1fn_x2)

    scale_a_len, scale_b_len = _scale_len(m, n, k)
    scale_a = torch.ones(scale_a_len, device=args.device, dtype=torch.float32).to(torch.float8_e4m3fn)
    scale_b = torch.ones(scale_b_len, device=args.device, dtype=torch.float32).to(torch.float8_e4m3fn)

    def native_scaled_mm():
        return torch._scaled_mm(
            act_fp4,
            weight_fp4.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )

    native_out = native_scaled_mm()
    torch.cuda.synchronize()

    # Also time the existing scalar bridge on one token / same N slice for context.
    scalar_x = torch.randn(k, device=args.device, dtype=torch.bfloat16, generator=gen)
    scalar_weight_u8 = packed.weight_packed[:n].contiguous()
    scalar_scale = packed.weight_scale[:n].contiguous()

    def scalar_bridge():
        return nvfp4_matvec_packed(
            scalar_x,
            scalar_weight_u8,
            scalar_scale,
            packed.weight_global_scale,
        )

    scalar_out = scalar_bridge()
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p4-native-fp4-scaled-mm-feasibility-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "weight": args.weight,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "has_float4_e2m1fn_x2": hasattr(torch, "float4_e2m1fn_x2"),
            "has_scaled_mm": hasattr(torch, "_scaled_mm"),
        },
        "native_fp4_scaled_mm": {
            "m": m,
            "n": n,
            "k": k,
            "activation_storage_shape": list(act_u8.shape),
            "weight_storage_shape": list(weight_u8.shape),
            "activation_dtype": str(act_fp4.dtype),
            "weight_dtype": str(weight_fp4.dtype),
            "scale_a_len": scale_a_len,
            "scale_b_len": scale_b_len,
            "scale_dtype": str(scale_a.dtype),
            "output_shape": list(native_out.shape),
            "output_dtype": str(native_out.dtype),
            "latency_ms": _bench(native_scaled_mm, warmup=args.warmup, iters=args.iters),
        },
        "scalar_bridge_context": {
            "n": n,
            "k": k,
            "output_shape": list(scalar_out.shape),
            "output_dtype": str(scalar_out.dtype),
            "latency_ms": _bench(scalar_bridge, warmup=args.warmup, iters=args.iters),
        },
        "next_steps": [
            "Quantize real BF16 activations into torch.float4_e2m1fn_x2 layout.",
            "Repack NVFP4 checkpoint scales from compact [out, K/16] into torch._scaled_mm's expanded scale_b vector.",
            "Compare native scaled_mm output against P3 scalar bridge for one real Linear.",
        ],
        "verdict": "PASS",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
