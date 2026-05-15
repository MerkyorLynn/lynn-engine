#!/usr/bin/env python3
"""P3-B probe: PackedNVFP4Linear runtime wrapper parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dequant import dequantize_nvfp4_v8_rtn_weight
from engine.nvfp4_runtime import PackedNVFP4Linear


DEFAULT_TENSOR = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"


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
    ap.add_argument("--tensor", default=DEFAULT_TENSOR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    device = torch.device(args.device)
    base = args.tensor.removesuffix(".weight")
    with safe_open(Path(args.v8) / "model.safetensors", framework="pt", device="cpu") as st:
        linear = PackedNVFP4Linear.from_safetensors(st, base, name=args.tensor, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    x_1d = torch.randn(linear.in_features, device=device, generator=generator, dtype=torch.bfloat16)
    x_2d = x_1d.reshape(1, -1)
    x_3d = x_1d.reshape(1, 1, -1)

    ref_weight = dequantize_nvfp4_v8_rtn_weight(
        linear.weight_packed,
        linear.weight_scale,
        linear.weight_global_scale,
        output_dtype=torch.float32,
    )
    ref_1d = F.linear(x_1d.float().unsqueeze(0), ref_weight).squeeze(0).to(torch.bfloat16)
    out_1d = linear(x_1d)
    out_2d = linear(x_2d)
    out_3d = linear(x_3d)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-runtime-wrapper-probe-v1",
        "tensor": args.tensor,
        "v8_model": str(Path(args.v8)),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "linear": {
            "in_features": linear.in_features,
            "out_features": linear.out_features,
            "weight_packed_shape": list(linear.weight_packed.shape),
            "weight_scale_shape": list(linear.weight_scale.shape),
        },
        "comparisons": {
            "shape_1d": list(out_1d.shape),
            "shape_2d": list(out_2d.shape),
            "shape_3d": list(out_3d.shape),
            "wrapper_1d_vs_reference": _compare(out_1d, ref_1d),
            "wrapper_2d_vs_1d": _compare(out_2d.reshape(-1), out_1d),
            "wrapper_3d_vs_1d": _compare(out_3d.reshape(-1), out_1d),
        },
        "timing_ms": {
            "wrapper_1d": _bench(lambda: linear(x_1d), warmup=args.warmup, iters=args.iters),
            "wrapper_3d": _bench(lambda: linear(x_3d), warmup=args.warmup, iters=args.iters),
            "resident_dequant_reference": _bench(
                lambda: F.linear(x_1d.float().unsqueeze(0), ref_weight).squeeze(0),
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "notes": [
            "PackedNVFP4Linear is the runtime contract for decode-path native packed linear",
            "it intentionally supports one token only; batched prefill remains resident BF16/dequant until a GEMM kernel lands",
        ],
    }
    c = result["comparisons"]["wrapper_1d_vs_reference"]
    result["verdict"] = "PASS" if c["cosine"] > 0.999 and c["rel_l2"] < 0.01 else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

