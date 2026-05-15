"""Slow, explicit dequantization helpers for Lynn engine correctness gates.

These helpers are intentionally boring CPU/PyTorch code. They are not the final
runtime kernel. Their job is to make packed quantized checkpoints readable and
testable before any native NVFP4 GEMM work starts.
"""
from __future__ import annotations

import torch


E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def unpack_fp4_e2m1_from_uint8(
    packed: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack NVFP4 E2M1 values from uint8 storage.

    compressed-tensors stores two FP4 values per byte. The low nibble is the
    first value and the high nibble is the second value, matching
    `compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8`.

    The 4-bit value layout is:

    - bit 3: sign
    - bits 0-2: magnitude index into [0, .5, 1, 1.5, 2, 3, 4, 6]
    """
    if packed.dtype is not torch.uint8:
        raise TypeError(f"expected uint8 packed tensor, got {packed.dtype}")
    if packed.ndim < 1:
        raise ValueError("packed tensor must have at least one dimension")

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


def dequantize_nvfp4_v8_rtn_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one compressed-tensors NVFP4 v8-RTN weight.

    This mirrors compressed-tensors' reference path:

    1. unpack uint8 -> FP4 E2M1 values,
    2. infer group size from `weight_scale.shape`,
    3. broadcast per-group scale over columns,
    4. if present, divide local scale by `weight_global_scale`,
    5. multiply unpacked values by the effective scale.

    The global scale direction is easy to get wrong; compressed-tensors'
    `_dequantize` helper uses `scale = scale / global_scale`, so Lynn engine's
    slow path follows that exactly.
    """
    unpacked = unpack_fp4_e2m1_from_uint8(weight_packed, dtype=torch.float32)
    scale = weight_scale.to(torch.float32)

    if unpacked.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            "P2 slow path currently expects 2D weight and 2D per-group scale; "
            f"got weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}"
        )
    if scale.shape[0] not in (1, unpacked.shape[0]):
        raise ValueError(
            "scale row count must be 1 or match weight output dim; "
            f"got weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}"
        )
    if scale.shape[1] == 0 or unpacked.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            "weight input dim must be divisible by scale group columns; "
            f"got weight={tuple(unpacked.shape)} scale={tuple(scale.shape)}"
        )

    group_size = unpacked.shape[1] // scale.shape[1]
    if group_size != 16:
        raise ValueError(f"expected NVFP4 group_size=16, got {group_size}")

    if weight_global_scale is not None:
        scale = scale / weight_global_scale.to(torch.float32)

    scale_full = scale.repeat_interleave(group_size, dim=1)
    if scale_full.shape[0] == 1 and unpacked.shape[0] != 1:
        scale_full = scale_full.expand(unpacked.shape[0], -1)
    scale_full = scale_full[: unpacked.shape[0], : unpacked.shape[1]]
    return (unpacked * scale_full).to(output_dtype)

