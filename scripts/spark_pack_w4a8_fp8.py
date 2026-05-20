#!/usr/bin/env python3
"""Offline NVFP4 → FP8 E4M3 repack for Spark sm_121 FP8 MMA path.

Spark sm_121 (GB10) has FP8 E4M3/E5M2 MMA at 162 TFLOPS peak (1.64× BF16)
but lacks FP4 MMA. The naive runtime path (dequant NVFP4→BF16→matmul) only
hits ~21 TPS at decode; a naive `_scaled_mm` swap of NVFP4 weight to FP8
inline measured 14 TPS due to activation cast / layout / scale / launch
overhead (memory ``reference_spark_fp8_w4a8_design_strategy_20260519``).

The strategy is: do the FP4→FP8 cast **once offline** so the inference
kernel only handles FP8 × FP8 matmul + activation cast. This script is
the offline repack tool (Phase 2 task #1).

V0 scope (this file):
  * Function-level NVFP4 → FP8 conversion with per-row scale.
  * Verify dequant cos > 0.999 vs original NVFP4.
  * Self-test on synthetic NVFP4 data (no external model files needed).
  * CLI to repack a single safetensors weight key for an end-to-end smoke.

V1 scope (next iteration, separate commit):
  * Full Lynn-native model dir repack (40 layers × N experts + projections).
  * Output dir manifest update (``lynn_quant_manifest.json`` schema).
  * Per-tensor scale granularity option.

V2 scope (later):
  * Col-major storage layout for cuBLASLt FP8 GEMM sweet spot.
  * Fused gate+up concatenated weight packing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# E2M1 magnitude table (compressed-tensors / Lynn-native NVFP4 share this)
E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32,
)

# FP8 E4M3 max representable absolute value (= 448).
FP8_E4M3_MAX = 448.0


@dataclass(slots=True)
class RepackResult:
    """Output of one NVFP4 → FP8 repack call."""
    fp8_weight: torch.Tensor          # [out_features, in_features] in float8_e4m3fn
    fp8_scale: torch.Tensor           # [out_features] (per-row) or scalar (per-tensor) — float32
    scale_granularity: str            # "per_row" | "per_tensor"
    bf16_intermediate_norm: float     # ||W_bf16||_F (for diagnostic)
    diff_max_abs_vs_bf16: float       # FP8-roundtrip vs BF16 max abs diff
    cosine_vs_bf16: float             # cos between FP8-roundtrip and BF16 dequant


def unpack_fp4_e2m1_from_uint8(
    packed: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack NVFP4 E2M1 values from uint8 storage.

    Two FP4 values per byte: low nibble first, high nibble second.
    Bit 3 = sign, bits 0-2 = magnitude index into E2M1_TO_FLOAT.

    Mirrors ``engine/dequant.py::unpack_fp4_e2m1_from_uint8``.
    """
    if packed.dtype is not torch.uint8:
        raise TypeError(f"expected uint8 packed tensor, got {packed.dtype}")
    flat = packed.flatten()
    low = flat & 0x0F
    high = (flat & 0xF0) >> 4
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    magnitudes = (combined & 0x07).to(torch.long)
    table = E2M1_TO_FLOAT.to(device=packed.device)
    values = table[magnitudes] * torch.where(
        signs,
        torch.tensor(-1.0, device=packed.device),
        torch.tensor(1.0, device=packed.device),
    )
    unpacked_shape = list(packed.shape)
    unpacked_shape[-1] *= 2
    return values.reshape(unpacked_shape).to(dtype=dtype)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weight to BF16/FP32.

    Mirrors ``engine/dequant.py::dequantize_nvfp4_v8_rtn_weight`` —
    group_size = 16, per-group BF16 scale, optional global_scale
    (compressed-tensors v8-RTN). Lynn-native ``per16_variable`` uses
    ``weight_global_scale=None``.
    """
    unpacked = unpack_fp4_e2m1_from_uint8(weight_packed, dtype=torch.float32)
    scale = weight_scale.to(torch.float32)
    if unpacked.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"expected 2D weight + 2D scale; got weight={tuple(unpacked.shape)} "
            f"scale={tuple(scale.shape)}"
        )
    group_size = unpacked.shape[1] // scale.shape[1]
    if group_size != 16:
        raise ValueError(f"expected group_size=16, got {group_size}")
    if weight_global_scale is not None:
        scale = scale / weight_global_scale.to(torch.float32)
    scale_full = scale.repeat_interleave(group_size, dim=1)
    if scale_full.shape[0] == 1 and unpacked.shape[0] != 1:
        scale_full = scale_full.expand(unpacked.shape[0], -1)
    scale_full = scale_full[: unpacked.shape[0], : unpacked.shape[1]]
    return (unpacked * scale_full).to(output_dtype)


def repack_nvfp4_to_fp8(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    scale_granularity: str = "per_row",
) -> RepackResult:
    """Offline NVFP4 → FP8 E4M3 repack.

    Args:
        weight_packed: uint8 [out_features, in_features // 2] NVFP4 E2M1 packed.
        weight_scale: BF16/FP32 [out_features, in_features // 16] per-group scale.
        weight_global_scale: optional FP32 scalar (compressed-tensors v8-RTN).
            For Lynn-native ``per16_variable`` pass None.
        scale_granularity: ``per_row`` (per-output-channel) or ``per_tensor``.
            cuBLASLt _scaled_mm supports per-row scale_b; ``per_tensor`` is
            simpler but sacrifices ~3-5% quality on heavy-tailed weights.

    Returns:
        :class:`RepackResult` with fp8_weight, fp8_scale, and verification
        diff vs the BF16 dequant.
    """
    if scale_granularity not in {"per_row", "per_tensor"}:
        raise ValueError(f"unknown scale_granularity={scale_granularity!r}")

    # 1. Reference BF16 dequant.
    bf16 = dequantize_nvfp4(
        weight_packed, weight_scale, weight_global_scale, output_dtype=torch.bfloat16,
    )
    bf16_norm = float(bf16.float().flatten().norm().item())

    # 2. Derive FP8 scale.
    bf16_f = bf16.float()
    if scale_granularity == "per_row":
        per_row_max = bf16_f.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        fp8_scale_2d = per_row_max / FP8_E4M3_MAX  # [N, 1] f32
        fp8_scale_out = fp8_scale_2d.squeeze(-1).contiguous()  # [N] f32
    else:
        per_tensor_max = bf16_f.abs().amax().clamp_min(1.0e-12)
        fp8_scale_2d = (per_tensor_max / FP8_E4M3_MAX).view(1, 1)
        fp8_scale_out = fp8_scale_2d.flatten().contiguous()  # [1] f32

    # 3. Quantize BF16 → FP8 E4M3 with derived scale.
    fp8_weight = (bf16_f / fp8_scale_2d).to(torch.float8_e4m3fn).contiguous()

    # 4. Verify round-trip dequant. Use FP64 for cosine accumulation —
    # large tensors (e.g. lm_head 311M elements) overflow FP32 accumulator
    # precision and produce nonsense cos > 1.0.
    fp8_roundtrip = fp8_weight.to(torch.float32) * fp8_scale_2d
    diff = (fp8_roundtrip - bf16_f).flatten()
    max_abs = float(diff.abs().max().item())
    af = fp8_roundtrip.flatten().double()
    bf = bf16_f.flatten().double()
    dot = float((af * bf).sum().item())
    na = float(af.norm().item())
    nb = float(bf.norm().item())
    cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")

    return RepackResult(
        fp8_weight=fp8_weight,
        fp8_scale=fp8_scale_out,
        scale_granularity=scale_granularity,
        bf16_intermediate_norm=bf16_norm,
        diff_max_abs_vs_bf16=max_abs,
        cosine_vs_bf16=cos,
    )


def synthetic_nvfp4(
    out_features: int,
    in_features: int,
    *,
    seed: int = 1234,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic NVFP4 (packed uint8 + BF16 per-16 scale) tensor.

    Used by the self-test. Produces a deterministic packed pattern that
    decodes to a realistic-ish weight distribution.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # Packed weight: random uint8 of shape [N, K/2]
    if in_features % 2 != 0:
        raise ValueError("in_features must be even for NVFP4 packing")
    if in_features % 16 != 0:
        raise ValueError("in_features must be divisible by group_size 16")
    packed = torch.randint(
        0, 256, (out_features, in_features // 2), generator=g, dtype=torch.uint8, device=device,
    )
    # Scale: per-row per-16-group, BF16, small positive values
    scale = (
        0.01 + 0.05 * torch.rand(
            (out_features, in_features // 16), generator=g, dtype=torch.float32, device=device,
        )
    ).to(torch.bfloat16)
    return packed, scale


def self_test() -> int:
    """Self-test: synthetic NVFP4 → FP8 repack, verify cos > 0.999."""
    torch.manual_seed(0)
    print("[spark_pack_w4a8_fp8] running self-test on synthetic NVFP4...")
    shapes = [
        (256, 2048),     # MoE expert gate/up size
        (2048, 6144),    # shared expert fan-in
        (2048, 2048),    # q/k/v_proj-style square
        (151936, 2048),  # lm_head-style large
    ]
    overall_ok = True
    for out_features, in_features in shapes:
        for granularity in ("per_row", "per_tensor"):
            packed, scale = synthetic_nvfp4(out_features, in_features)
            result = repack_nvfp4_to_fp8(packed, scale, scale_granularity=granularity)
            cos_ok = result.cosine_vs_bf16 > 0.999
            print(
                f"  shape=({out_features}, {in_features}) granularity={granularity}: "
                f"cos={result.cosine_vs_bf16:.6f} max_abs={result.diff_max_abs_vs_bf16:.4e} "
                f"fp8_scale_shape={tuple(result.fp8_scale.shape)} "
                f"bf16_norm={result.bf16_intermediate_norm:.2e} "
                f"{'OK' if cos_ok else 'FAIL'}"
            )
            if not cos_ok:
                overall_ok = False

    print(f"[spark_pack_w4a8_fp8] self-test {'PASSED' if overall_ok else 'FAILED'}")
    return 0 if overall_ok else 1


def repack_safetensors_weight(
    input_path: Path,
    output_path: Path,
    weight_key: str,
    *,
    scale_granularity: str = "per_row",
) -> dict[str, Any]:
    """Repack a single weight from a safetensors file.

    Looks for ``{weight_key}.weight_packed`` and ``{weight_key}.weight_scale``
    (Lynn-native naming) in the input file, repacks, and writes FP8 result
    + scale + verification report.

    Returns a JSON-serializable manifest dict.
    """
    from safetensors.torch import load_file, save_file

    src = load_file(str(input_path))
    packed_key = f"{weight_key}.weight_packed"
    scale_key = f"{weight_key}.weight_scale"
    global_scale_key = f"{weight_key}.weight_global_scale"
    if packed_key not in src or scale_key not in src:
        raise KeyError(
            f"input file missing required keys {packed_key!r} / {scale_key!r}; "
            f"available keys: {sorted(src.keys())[:10]}..."
        )
    packed = src[packed_key]
    scale = src[scale_key]
    global_scale = src.get(global_scale_key)

    result = repack_nvfp4_to_fp8(
        packed, scale, global_scale, scale_granularity=scale_granularity,
    )

    out_tensors = {
        f"{weight_key}.weight_fp8": result.fp8_weight,
        f"{weight_key}.weight_fp8_scale": result.fp8_scale,
    }
    save_file(out_tensors, str(output_path))

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "weight_key": weight_key,
        "input_packed_shape": list(packed.shape),
        "input_packed_dtype": str(packed.dtype),
        "input_scale_shape": list(scale.shape),
        "input_scale_dtype": str(scale.dtype),
        "output_fp8_shape": list(result.fp8_weight.shape),
        "output_fp8_dtype": str(result.fp8_weight.dtype),
        "output_scale_shape": list(result.fp8_scale.shape),
        "scale_granularity": result.scale_granularity,
        "cosine_vs_bf16": result.cosine_vs_bf16,
        "max_abs_vs_bf16": result.diff_max_abs_vs_bf16,
        "bf16_intermediate_norm": result.bf16_intermediate_norm,
    }
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline NVFP4 → FP8 E4M3 repack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_test = sub.add_parser("self-test", help="Run self-test on synthetic NVFP4")  # noqa: F841

    sp_one = sub.add_parser("one", help="Repack a single weight from a safetensors file")
    sp_one.add_argument("--input", required=True, type=Path)
    sp_one.add_argument("--output", required=True, type=Path)
    sp_one.add_argument(
        "--weight-key", required=True,
        help='e.g. "mlp.experts.0.gate_proj" or "self_attn.q_proj"',
    )
    sp_one.add_argument(
        "--scale-granularity", default="per_row", choices=["per_row", "per_tensor"],
    )
    sp_one.add_argument(
        "--manifest-out", default=None, type=Path,
        help="optional JSON manifest write path",
    )

    args = ap.parse_args()

    if args.cmd == "self-test":
        return self_test()
    if args.cmd == "one":
        m = repack_safetensors_weight(
            args.input, args.output, args.weight_key,
            scale_granularity=args.scale_granularity,
        )
        print(json.dumps(m, indent=2))
        if args.manifest_out is not None:
            args.manifest_out.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        return 0
    print(f"unknown cmd {args.cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
