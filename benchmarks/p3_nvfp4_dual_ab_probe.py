#!/usr/bin/env python3
"""P3-E fusion probe: compute in_proj_a and in_proj_b in one packed kernel."""
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
from triton_kernels.nvfp4_linear import nvfp4_dual_matvec_packed


def _load(v8_dir: Path, layer: int, short: str, device: str) -> PackedNVFP4Linear:
    base = f"model.language_model.layers.{layer}.{short.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(st, base, name=short, device=device)


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
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    a = _load(Path(args.v8), args.layer, "linear_attn.in_proj_a.weight", args.device)
    b = _load(Path(args.v8), args.layer, "linear_attn.in_proj_b.weight", args.device)
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    x = torch.randn(a.in_features, device=args.device, dtype=torch.bfloat16, generator=gen)

    separate_a = a(x)
    separate_b = b(x)
    fused_a, fused_b = nvfp4_dual_matvec_packed(
        x,
        a.weight_packed,
        a.weight_scale,
        a.weight_global_scale,
        b.weight_packed,
        b.weight_scale,
        b.weight_global_scale,
    )
    fused_a = fused_a.to(x.dtype)
    fused_b = fused_b.to(x.dtype)
    torch.cuda.synchronize()

    sep_ms = _bench(lambda: (a(x), b(x)), warmup=args.warmup, iters=args.iters)
    fused_ms = _bench(
        lambda: nvfp4_dual_matvec_packed(
            x,
            a.weight_packed,
            a.weight_scale,
            a.weight_global_scale,
            b.weight_packed,
            b.weight_scale,
            b.weight_global_scale,
        ),
        warmup=args.warmup,
        iters=args.iters,
    )

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-dual-ab-probe-v1",
        "v8_model": str(Path(args.v8)),
        "layer": args.layer,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "comparisons": {
            "a_fused_vs_separate": _compare(fused_a, separate_a),
            "b_fused_vs_separate": _compare(fused_b, separate_b),
        },
        "timing_ms": {
            "separate_a_then_b": sep_ms,
            "fused_dual_ab": fused_ms,
            "speedup": sep_ms / fused_ms if fused_ms > 0 else None,
        },
        "notes": [
            "A/B are 32-output projections; separate kernels are launch-overhead dominated.",
            "This probe tests whether pairing same-shaped packed matvecs is worth generalizing.",
        ],
    }
    ok = all(v["cosine"] > 0.999 for v in result["comparisons"].values())
    result["verdict"] = "PASS" if ok else "FAIL"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

