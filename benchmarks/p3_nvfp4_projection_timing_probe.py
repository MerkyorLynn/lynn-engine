#!/usr/bin/env python3
"""P3-E probe: timing breakdown for packed linear-attention projections."""
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


PROJECTIONS = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
]


def _load_packed(v8_dir: Path, layer: int, short_name: str, device: str) -> PackedNVFP4Linear:
    base = f"model.language_model.layers.{layer}.{short_name.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(st, base, name=short_name, device=device)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    device = args.device
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    h = torch.randn(1, 1, 2048, device=device, dtype=torch.bfloat16, generator=gen)
    core = torch.randn(1, 1, 4096, device=device, dtype=torch.bfloat16, generator=gen)

    rows: list[dict[str, Any]] = []
    for name in PROJECTIONS:
        packed = _load_packed(Path(args.v8), args.layer, name, device)
        x = core if name == "linear_attn.out_proj.weight" else h
        ref_weight = dequantize_nvfp4_v8_rtn_weight(
            packed.weight_packed,
            packed.weight_scale,
            packed.weight_global_scale,
            output_dtype=torch.float32,
        )
        ref = F.linear(x.float(), ref_weight).to(torch.bfloat16)
        out = packed(x)
        comparison = _compare(out, ref)
        packed_ms = _bench(lambda: packed(x), warmup=args.warmup, iters=args.iters)
        resident_ms = _bench(
            lambda: F.linear(x.float(), ref_weight).to(torch.bfloat16),
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(
            {
                "projection": name,
                "in_features": packed.in_features,
                "out_features": packed.out_features,
                "comparison": comparison,
                "timing_ms": {
                    "packed": packed_ms,
                    "resident_dequantized": resident_ms,
                    "packed_minus_resident": packed_ms - resident_ms,
                },
            }
        )

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-projection-timing-probe-v1",
        "v8_model": str(Path(args.v8)),
        "layer": args.layer,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "projections": rows,
        "totals_ms": {
            "packed": sum(x["timing_ms"]["packed"] for x in rows),
            "resident_dequantized": sum(x["timing_ms"]["resident_dequantized"] for x in rows),
        },
        "notes": [
            "This isolates projection cost only; it does not include conv, recurrent update, RMSNormGated, or activations.",
            "Use this to decide where fusion/tensor-core work buys the most.",
        ],
    }
    result["verdict"] = "PASS" if all(x["comparison"]["cosine"] > 0.999 for x in rows) else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

