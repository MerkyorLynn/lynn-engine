#!/usr/bin/env python3
"""P4-B probe: native FP4 `_scaled_mm` numeric alignment on real checkpoint bytes.

P4-A proved torch/cuBLASLt accepts the v8-RTN `weight_packed` storage as
`torch.float4_e2m1fn_x2`. P4-B starts aligning numerics against the P3 scalar
bridge.

Scope of this first numeric probe:
  - use synthetic FP4 activations so activation quantization is not yet a factor;
  - repack real checkpoint compact weight scales into `_scaled_mm`'s swizzled scale_b;
  - compare native `_scaled_mm` output against explicit unpack/dequant matmul.

If this passes, the remaining P4 work is activation quantization + integrating
the native path behind `PackedNVFP4Linear`.
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

from engine.dequant import dequantize_nvfp4_v8_rtn_weight, unpack_fp4_e2m1_from_uint8
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
    # PyTorch 2.10 / cuBLASLt requires a padded blockwise scale layout for
    # torch.float4_e2m1fn_x2: outer dim >= 128 and K groups >= 4.
    return max(dim, 128), max(k // 16, 4)


def _ones_scale(dim: int, k: int, device: str) -> torch.Tensor:
    rows, groups = _scale_shape(dim, k)
    return torch.ones(rows * groups, device=device, dtype=torch.float32).to(torch.float8_e4m3fn)


def _weight_scale_row_major(packed: PackedNVFP4Linear, n: int) -> torch.Tensor:
    """Repack compact `[out, K/16]` effective scales into torch's flat vector."""
    k = packed.in_features
    rows, groups = _scale_shape(n, k)
    effective = (
        packed.weight_scale[:n].float()
        / packed.weight_global_scale.to(packed.weight_scale.device).float()
    )
    padded = torch.ones(rows, groups, device=effective.device, dtype=torch.float32)
    padded[:n, :] = effective
    return padded.contiguous().flatten().to(torch.float8_e4m3fn)


def _weight_scale_group_major(packed: PackedNVFP4Linear, n: int) -> torch.Tensor:
    """Alternative layout probe: groups-major flattening."""
    k = packed.in_features
    rows, groups = _scale_shape(n, k)
    effective = (
        packed.weight_scale[:n].float()
        / packed.weight_global_scale.to(packed.weight_scale.device).float()
    )
    padded = torch.ones(groups, rows, device=effective.device, dtype=torch.float32)
    padded[:, :n] = effective.t().contiguous()
    return padded.contiguous().flatten().to(torch.float8_e4m3fn)


def _torch_scaled_mm_scale_index(row: int, group: int, groups: int) -> int:
    """Index into torch._scaled_mm's expanded FP4 scale vector.

    This is empirical for PyTorch 2.10's Blackwell FP4 `_scaled_mm` path. The
    scale vector is neither plain row-major nor group-major. Within each 128-row
    tile, groups are laid out in 4-group chunks, rows are swizzled by 32-row
    stripes, and each chunk occupies 512 elements:

        idx = tile * (128 * groups)
            + (group // 4) * 512
            + (row % 32) * 16
            + (row // 32) * 4
            + (group % 4)

    P4-B discovered this with single-nonzero synthetic FP4 matrices on R6000
    sm_120. Keeping it here makes the native path reproducible instead of
    relying on a guessed layout.
    """
    tile = row // 128
    row_in_tile = row % 128
    return (
        tile * (128 * groups)
        + (group // 4) * 512
        + (row_in_tile % 32) * 16
        + (row_in_tile // 32) * 4
        + (group % 4)
    )


def _weight_scale_swizzled(packed: PackedNVFP4Linear, n: int) -> torch.Tensor:
    """Repack compact `[out, K/16]` scales into torch._scaled_mm scale_b."""
    k = packed.in_features
    rows, groups = _scale_shape(n, k)
    actual_groups = packed.weight_scale.shape[1]
    effective = (
        packed.weight_scale[:n].float()
        / packed.weight_global_scale.to(packed.weight_scale.device).float()
    )
    expanded = torch.ones(rows * groups, device=effective.device, dtype=torch.float32)
    for row in range(n):
        for group in range(actual_groups):
            expanded[_torch_scaled_mm_scale_index(row, group, groups)] = effective[row, group]
    return expanded.to(torch.float8_e4m3fn)


def _native_mm(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    return torch._scaled_mm(
        act_fp4,
        weight_fp4.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weight", default="linear_attn.in_proj_qkv.weight")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--out-features", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--cosine-threshold", type=float, default=0.994)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.12)
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
    act_u8 = torch.randint(0, 256, (m, k // 2), device=args.device, dtype=torch.uint8, generator=gen)
    act_fp4 = act_u8.view(torch.float4_e2m1fn_x2)
    act_ref = unpack_fp4_e2m1_from_uint8(act_u8, dtype=torch.float32)

    weight_u8 = packed.weight_packed[:n].contiguous()
    weight_fp4 = weight_u8.view(torch.float4_e2m1fn_x2)
    weight_ref = dequantize_nvfp4_v8_rtn_weight(
        weight_u8,
        packed.weight_scale[:n].contiguous(),
        packed.weight_global_scale,
        output_dtype=torch.float32,
    )
    explicit_ref = F.linear(act_ref, weight_ref)

    scale_a = _ones_scale(m, k, args.device)
    scale_b_layouts = {
        "row_major": _weight_scale_row_major(packed, n),
        "group_major": _weight_scale_group_major(packed, n),
        "torch_swizzled": _weight_scale_swizzled(packed, n),
    }

    layout_results: dict[str, Any] = {}
    for name, scale_b in scale_b_layouts.items():
        try:
            native = _native_mm(act_fp4, weight_fp4, scale_a, scale_b).float()
            layout_results[name] = {
                "ok": True,
                "comparison": _compare(native, explicit_ref),
                "native_shape": list(native.shape),
            }
        except Exception as exc:
            layout_results[name] = {"ok": False, "error": repr(exc)}

    successful = {
        name: item for name, item in layout_results.items()
        if item.get("ok") and "comparison" in item
    }
    best_name = None
    if successful:
        best_name = max(successful, key=lambda k_: successful[k_]["comparison"]["cosine"])
    best = successful.get(best_name) if best_name else None

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p4-native-fp4-numeric-probe-v1",
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
            "activation_storage": list(act_u8.shape),
            "weight_storage": list(weight_u8.shape),
            "scale_a_len": int(scale_a.numel()),
            "scale_b_len": int(next(iter(scale_b_layouts.values())).numel()),
        },
        "layouts": layout_results,
        "best_layout": best_name,
        "thresholds": {
            "cosine": args.cosine_threshold,
            "rel_l2": args.rel_l2_threshold,
        },
        "notes": [
            "Activation is synthetic FP4 with scale_a=1 to isolate weight scale repacking.",
            "Reference path explicitly unpacks activation and dequantizes checkpoint weight with compact v8 scales.",
            "torch_swizzled is the empirically discovered PyTorch 2.10 FP4 scale layout on sm_120.",
            "Native path converts checkpoint scales to float8_e4m3fn as required by torch._scaled_mm, so a small numeric gap vs FP32 explicit reference is expected.",
            "If this passes, next P4 step is real BF16 activation quantization.",
        ],
    }
    if best:
        cmp = best["comparison"]
        result["verdict"] = (
            "PASS" if cmp["cosine"] >= args.cosine_threshold and cmp["rel_l2"] <= args.rel_l2_threshold else "FAIL"
        )
    else:
        result["verdict"] = "FAIL"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
