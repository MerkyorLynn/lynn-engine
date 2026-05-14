#!/usr/bin/env python3
"""P4-D probe: PackedNVFP4Linear native scaled-mm backend.

This exercises the new opt-in runtime backend:

    PackedNVFP4Linear.forward(..., backend="native_scaled_mm")

The oracle is the existing P3 scalar bridge backend for the same packed weight.
Because native_scaled_mm quantizes activations to FP4 while the scalar bridge
uses BF16 activations, this probe measures integration and expected activation
quantization drift, not exact equality.
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
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
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
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cosine-threshold", type=float, default=0.985)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.20)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)
    base = f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        linear = PackedNVFP4Linear.from_safetensors(st, base, name=args.weight, device=args.device)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    x = torch.randn((linear.in_features,), device=args.device, dtype=dtype, generator=gen)

    def scalar():
        return linear(x, backend="scalar_bridge", output_dtype=dtype)

    def native():
        return linear(x, backend="native_scaled_mm", output_dtype=dtype)

    scalar_out = scalar()
    native_out = native()
    torch.cuda.synchronize()
    comparison = _compare(native_out, scalar_out)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p4-native-fp4-runtime-wrapper-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "weight": args.weight,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "shape": {
            "in_features": linear.in_features,
            "out_features": linear.out_features,
        },
        "comparison": comparison,
        "latency_ms": {
            "scalar_bridge": _bench(scalar, warmup=args.warmup, iters=args.iters),
            "native_scaled_mm": _bench(native, warmup=args.warmup, iters=args.iters),
        },
        "thresholds": {
            "cosine": args.cosine_threshold,
            "rel_l2": args.rel_l2_threshold,
        },
        "notes": [
            "native_scaled_mm is opt-in; scalar_bridge remains the default backend.",
            "native_scaled_mm quantizes activations to FP4, so it is expected to differ from the BF16-activation scalar bridge.",
            "Current native_scaled_mm backend still quantizes activations and swizzles scale_a in Python per call; this is an integration gate, not the final performance path.",
            "This probe validates runtime integration before replacing decode layer internals.",
        ],
    }
    result["verdict"] = (
        "PASS"
        if comparison["cosine"] >= args.cosine_threshold
        and comparison["rel_l2"] <= args.rel_l2_threshold
        else "FAIL"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
