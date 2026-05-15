#!/usr/bin/env python3
"""P4-C probe: BF16 activation quantization for native FP4 `_scaled_mm`.

P4-B proved the checkpoint weight scale repack contract for PyTorch's native
FP4 `_scaled_mm` path. P4-C adds the other half of the GEMM contract:

  - quantize BF16 activations into torch.float4_e2m1fn_x2 storage;
  - repack activation scales into the same swizzled scale vector layout;
  - compare native `_scaled_mm` with an explicit FP4-dequant reference.

This still uses a deterministic synthetic BF16 activation matrix. The goal is
to prove the quantize/repack/native-GEMM math before wiring real layer
activations through the runtime.
"""
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

from engine.dequant import E2M1_TO_FLOAT
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


def _scale_shape(dim: int, k: int) -> tuple[int, int]:
    return max(dim, 128), max(k // 16, 4)


def _torch_scaled_mm_scale_index(row: int, group: int, groups: int) -> int:
    tile = row // 128
    row_in_tile = row % 128
    return (
        tile * (128 * groups)
        + (group // 4) * 512
        + (row_in_tile % 32) * 16
        + (row_in_tile // 32) * 4
        + (group % 4)
    )


def _compact_scale_to_swizzled_fp8(scale: torch.Tensor, *, outer_dim: int, k: int) -> torch.Tensor:
    """Convert compact `[outer, K/16]` scales to torch._scaled_mm layout."""
    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got {tuple(scale.shape)}")
    rows, groups = _scale_shape(outer_dim, k)
    actual_groups = scale.shape[1]
    expanded = torch.ones(rows * groups, device=scale.device, dtype=torch.float32)
    for row in range(scale.shape[0]):
        for group in range(actual_groups):
            expanded[_torch_scaled_mm_scale_index(row, group, groups)] = scale[row, group]
    return expanded.to(torch.float8_e4m3fn)


def _quantize_activation_to_fp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize `[M, K]` BF16/FP32 activations to FP4 packed bytes.

    Returns:
      packed uint8 `[M, K/2]`,
      compact FP32 scales `[M, K/16]`,
      dequantized FP32 activation using float8-rounded scales.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    if x.shape[1] % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {x.shape[1]}")
    table = E2M1_TO_FLOAT.to(device=x.device)
    x32 = x.float()
    m, k = x32.shape
    groups = k // 16
    xg = x32.reshape(m, groups, 16)
    scale = (xg.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
    normalized = xg.abs() / scale.unsqueeze(-1)
    # Pick the nearest E2M1 magnitude code. This is intentionally explicit and
    # deterministic; production quantization can replace this with a fused op.
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(m, k)
    low = codes[:, 0::2]
    high = codes[:, 1::2] << 4
    packed = (low | high).contiguous()

    # Match native `_scaled_mm`: runtime scales are float8_e4m3fn, not FP32.
    rounded_scale = scale.to(torch.float8_e4m3fn).float()
    signs = torch.where((codes & 0x08).bool(), -1.0, 1.0)
    values = table[(codes & 0x07).long()] * signs
    dequant = values.reshape(m, groups, 16) * rounded_scale.unsqueeze(-1)
    return packed, scale, dequant.reshape(m, k).float()


def _dequant_weight_with_fp8_scales(packed: PackedNVFP4Linear, n: int) -> torch.Tensor:
    """Explicit weight dequant using the same float8-rounded scales as native."""
    table = E2M1_TO_FLOAT.to(device=packed.weight_packed.device)
    weight_u8 = packed.weight_packed[:n].contiguous()
    flat = weight_u8.flatten()
    low = flat & 0x0F
    high = (flat & 0xF0) >> 4
    codes = torch.stack((low, high), dim=1).flatten().reshape(n, packed.in_features)
    signs = torch.where((codes & 0x08).bool(), -1.0, 1.0)
    values = table[(codes & 0x07).long()] * signs

    effective_scale = (
        packed.weight_scale[:n].float()
        / packed.weight_global_scale.to(packed.weight_scale.device).float()
    )
    rounded_scale = effective_scale.to(torch.float8_e4m3fn).float()
    scale_full = rounded_scale.repeat_interleave(16, dim=1)
    return (values * scale_full).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weight", default="linear_attn.in_proj_qkv.weight")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out-features", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.05)
    args = ap.parse_args()

    v8_dir = Path(args.v8)
    base = f"model.language_model.layers.{args.layer}.{args.weight.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        packed = PackedNVFP4Linear.from_safetensors(st, base, name=args.weight, device=args.device)

    m = args.tokens
    n = min(args.out_features, packed.out_features)
    k = packed.in_features

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    activation_bf16 = torch.randn((m, k), device=args.device, dtype=torch.bfloat16, generator=gen)
    act_packed, act_scale, act_ref = _quantize_activation_to_fp4(activation_bf16)
    act_fp4 = act_packed.view(torch.float4_e2m1fn_x2)

    weight_u8 = packed.weight_packed[:n].contiguous()
    weight_fp4 = weight_u8.view(torch.float4_e2m1fn_x2)
    weight_ref = _dequant_weight_with_fp8_scales(packed, n)

    scale_a = _compact_scale_to_swizzled_fp8(act_scale, outer_dim=m, k=k)
    weight_effective_scale = (
        packed.weight_scale[:n].float()
        / packed.weight_global_scale.to(packed.weight_scale.device).float()
    )
    scale_b = _compact_scale_to_swizzled_fp8(weight_effective_scale, outer_dim=n, k=k)

    explicit_ref = F.linear(act_ref.to(torch.float16), weight_ref.to(torch.float16)).float()
    native = torch._scaled_mm(
        act_fp4,
        weight_fp4.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    ).float()
    comparison = _compare(native, explicit_ref)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p4-native-fp4-activation-quant-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "weight": args.weight,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "shape": {
            "m": m,
            "n": n,
            "k": k,
            "activation_storage": list(act_packed.shape),
            "weight_storage": list(weight_u8.shape),
            "scale_a_len": int(scale_a.numel()),
            "scale_b_len": int(scale_b.numel()),
        },
        "activation_quantization": {
            "group_size": 16,
            "scale_rule": "amax(abs(x_group)) / 6.0",
            "scale_dtype_for_native": str(scale_a.dtype),
            "packed_dtype": str(act_fp4.dtype),
        },
        "comparison": comparison,
        "thresholds": {
            "cosine": args.cosine_threshold,
            "rel_l2": args.rel_l2_threshold,
        },
        "notes": [
            "Reference dequantizes both activation and weight with the same float8-rounded scales used by torch._scaled_mm.",
            "This isolates native GEMM correctness from activation quantization quality against the original BF16 activation.",
            "Next step is replacing PackedNVFP4Linear's scalar bridge with this native path for decode and rerunning P3-K.",
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
