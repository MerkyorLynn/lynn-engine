"""Native packed NVFP4 linear probes.

This module is the first Lynn engine P3 step: consume compressed-tensors
NVFP4 v8-RTN tensors in their packed representation and run a matvec without
materializing the full BF16/FP32 weight matrix.

It is intentionally scoped to single-token matvec first. The kernel is a
correctness and integration bridge, not the final Blackwell tensor-core FP4
GEMM. Once this path is wired into layer forward, we can replace the inner
kernel with a true FP4 GEMM while keeping the loader/runtime contract stable.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on non-GPU dev hosts.
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for native NVFP4 packed kernels")


if HAS_TRITON:

    @triton.jit
    def _e2m1_from_nibble(nibble):
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        val = tl.where(
            mag == 0,
            0.0,
            tl.where(
                mag == 1,
                0.5,
                tl.where(
                    mag == 2,
                    1.0,
                    tl.where(
                        mag == 3,
                        1.5,
                        tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                    ),
                ),
            ),
        )
        return tl.where(sign, -val, val)


    @triton.jit
    def _nearest_e2m1_mag_code(abs_norm):
        # E2M1 magnitudes: [0, .5, 1, 1.5, 2, 3, 4, 6].
        # Boundaries are midpoints between adjacent values.
        # Match torch.argmin tie-breaking in the reference quantizer: when a
        # value sits exactly on a midpoint, pick the lower magnitude code.
        return tl.where(
            abs_norm <= 0.25,
            0,
            tl.where(
                abs_norm <= 0.75,
                1,
                tl.where(
                    abs_norm <= 1.25,
                    2,
                    tl.where(
                        abs_norm <= 1.75,
                        3,
                        tl.where(
                            abs_norm <= 2.5,
                            4,
                            tl.where(abs_norm <= 3.5, 5, tl.where(abs_norm <= 5.0, 6, 7)),
                        ),
                    ),
                ),
            ),
        )


    @triton.jit
    def _quantize_fp4_m1_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        K: tl.constexpr,
        GROUPS: tl.constexpr,
    ):
        group = tl.program_id(0)
        pair = tl.arange(0, 8)
        low_x = tl.load(x_ptr + group * 16 + pair * 2, mask=group < GROUPS, other=0.0).to(tl.float32)
        high_x = tl.load(x_ptr + group * 16 + pair * 2 + 1, mask=group < GROUPS, other=0.0).to(tl.float32)
        low_abs = tl.abs(low_x)
        high_abs = tl.abs(high_x)
        max_abs = tl.maximum(tl.max(low_abs, axis=0), tl.max(high_abs, axis=0))
        scale = tl.maximum(max_abs / 6.0, 1.0e-8)
        low_mag = _nearest_e2m1_mag_code(low_abs / scale).to(tl.uint8)
        high_mag = _nearest_e2m1_mag_code(high_abs / scale).to(tl.uint8)
        low_sign = tl.where(low_x < 0.0, 8, 0).to(tl.uint8)
        high_sign = tl.where(high_x < 0.0, 8, 0).to(tl.uint8)
        low_code = low_mag | low_sign
        high_code = high_mag | high_sign
        packed = low_code | (high_code << 4)
        tl.store(packed_ptr + group * 8 + pair, packed, mask=group < GROUPS)

        # torch._scaled_mm FP4 scale layout for M=1:
        # idx = (group // 4) * 512 + (group % 4)
        scale_offset = (group // 4) * 512 + (group % 4)
        tl.store(scale_ptr + scale_offset, scale, mask=group < GROUPS)


    @triton.jit
    def _nvfp4_matvec_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        global_scale_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        packed_stride_m: tl.constexpr,
        packed_stride_n: tl.constexpr,
        scale_stride_m: tl.constexpr,
        scale_stride_g: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_GLOBAL_SCALE: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        global_scale = tl.full((), 1.0, dtype=tl.float32)
        if HAS_GLOBAL_SCALE:
            global_scale = tl.load(global_scale_ptr).to(tl.float32)

        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N

            # compressed-tensors packs two FP4 values per byte. Low nibble is
            # the even column, high nibble is the odd column.
            packed_cols = cols // 2
            packed_offsets = (
                rows[:, None] * packed_stride_m
                + packed_cols[None, :] * packed_stride_n
            )
            packed = tl.load(
                packed_ptr + packed_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0,
            )
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            nibble = tl.where((cols[None, :] & 1) == 0, low, high)
            w_fp4 = _e2m1_from_nibble(nibble)

            scale_cols = cols // 16
            scale_offsets = (
                rows[:, None] * scale_stride_m
                + scale_cols[None, :] * scale_stride_g
            )
            scale = tl.load(
                scale_ptr + scale_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            effective_scale = scale / global_scale

            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
            acc += tl.sum(w_fp4 * effective_scale * x[None, :], axis=1)

        tl.store(out_ptr + rows, acc, mask=row_mask)


    @triton.jit
    def _nvfp4_dual_matvec_kernel(
        x_ptr,
        packed_a_ptr,
        scale_a_ptr,
        global_a_ptr,
        packed_b_ptr,
        scale_b_ptr,
        global_b_ptr,
        out_a_ptr,
        out_b_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        packed_stride_m: tl.constexpr,
        packed_stride_n: tl.constexpr,
        scale_stride_m: tl.constexpr,
        scale_stride_g: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_GLOBAL_SCALE: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M

        acc_a = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc_b = tl.zeros((BLOCK_M,), dtype=tl.float32)
        global_a = tl.full((), 1.0, dtype=tl.float32)
        global_b = tl.full((), 1.0, dtype=tl.float32)
        if HAS_GLOBAL_SCALE:
            global_a = tl.load(global_a_ptr).to(tl.float32)
            global_b = tl.load(global_b_ptr).to(tl.float32)

        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            packed_cols = cols // 2
            scale_cols = cols // 16

            packed_offsets = (
                rows[:, None] * packed_stride_m
                + packed_cols[None, :] * packed_stride_n
            )
            scale_offsets = (
                rows[:, None] * scale_stride_m
                + scale_cols[None, :] * scale_stride_g
            )
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            packed_a = tl.load(
                packed_a_ptr + packed_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0,
            )
            low_a = packed_a & 0x0F
            high_a = (packed_a >> 4) & 0x0F
            nibble_a = tl.where((cols[None, :] & 1) == 0, low_a, high_a)
            w_a = _e2m1_from_nibble(nibble_a)
            scale_a = tl.load(
                scale_a_ptr + scale_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_a += tl.sum(w_a * (scale_a / global_a) * x[None, :], axis=1)

            packed_b = tl.load(
                packed_b_ptr + packed_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0,
            )
            low_b = packed_b & 0x0F
            high_b = (packed_b >> 4) & 0x0F
            nibble_b = tl.where((cols[None, :] & 1) == 0, low_b, high_b)
            w_b = _e2m1_from_nibble(nibble_b)
            scale_b = tl.load(
                scale_b_ptr + scale_offsets,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_b += tl.sum(w_b * (scale_b / global_b) * x[None, :], axis=1)

        tl.store(out_a_ptr + rows, acc_a, mask=row_mask)
        tl.store(out_b_ptr + rows, acc_b, mask=row_mask)


def nvfp4_matvec_packed(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    block_m: int = 16,
    block_n: int = 128,
) -> torch.Tensor:
    """Run `weight @ x` directly from packed NVFP4 v8-RTN tensors.

    Args:
        x: 1D activation vector `[in_features]`, CUDA tensor.
        weight_packed: packed uint8 tensor `[out_features, in_features / 2]`.
        weight_scale: per-group scale tensor `[out_features, in_features / 16]`.
            Pass it as float32/float16/bfloat16. Callers may convert from FP8
            once at load time; scales are tiny relative to weights.
        weight_global_scale: optional scalar tensor. compressed-tensors uses
            `effective_scale = weight_scale / weight_global_scale`.
    """
    _require_triton()
    if not x.is_cuda or not weight_packed.is_cuda or not weight_scale.is_cuda:
        raise ValueError("x, weight_packed, and weight_scale must be CUDA tensors")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got shape={tuple(x.shape)}")
    if weight_packed.dtype != torch.uint8:
        raise TypeError(f"weight_packed must be uint8, got {weight_packed.dtype}")
    if weight_packed.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError(
            "weight_packed and weight_scale must be 2D; "
            f"got packed={tuple(weight_packed.shape)} scale={tuple(weight_scale.shape)}"
        )

    out_features = weight_packed.shape[0]
    in_features = weight_packed.shape[1] * 2
    if x.numel() != in_features:
        raise ValueError(f"x has {x.numel()} elements, expected {in_features}")
    if weight_scale.shape[0] != out_features or weight_scale.shape[1] * 16 != in_features:
        raise ValueError(
            "weight_scale must be [out_features, in_features / 16]; "
            f"got packed={tuple(weight_packed.shape)} scale={tuple(weight_scale.shape)}"
        )

    x = x.contiguous()
    weight_packed = weight_packed.contiguous()
    weight_scale = weight_scale.contiguous()
    out = torch.empty((out_features,), device=x.device, dtype=torch.float32)

    has_global_scale = weight_global_scale is not None
    if weight_global_scale is not None:
        weight_global_scale = weight_global_scale.to(device=x.device).contiguous()
    else:
        weight_global_scale = torch.empty((1,), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(out_features, block_m),)
    _nvfp4_matvec_kernel[grid](
        x,
        weight_packed,
        weight_scale,
        weight_global_scale,
        out,
        out_features,
        in_features,
        weight_packed.stride(0),
        weight_packed.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        block_m,
        block_n,
        has_global_scale,
        num_warps=4,
    )
    return out


def nvfp4_dual_matvec_packed(
    x: torch.Tensor,
    weight_a_packed: torch.Tensor,
    weight_a_scale: torch.Tensor,
    weight_a_global_scale: torch.Tensor,
    weight_b_packed: torch.Tensor,
    weight_b_scale: torch.Tensor,
    weight_b_global_scale: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two same-shaped packed NVFP4 matvecs in one Triton launch."""
    _require_triton()
    if weight_a_packed.shape != weight_b_packed.shape:
        raise ValueError("dual matvec requires matching packed weight shapes")
    if weight_a_scale.shape != weight_b_scale.shape:
        raise ValueError("dual matvec requires matching scale shapes")
    out_a = torch.empty((weight_a_packed.shape[0],), device=x.device, dtype=torch.float32)
    out_b = torch.empty((weight_b_packed.shape[0],), device=x.device, dtype=torch.float32)
    x = x.contiguous()
    weight_a_packed = weight_a_packed.contiguous()
    weight_b_packed = weight_b_packed.contiguous()
    weight_a_scale = weight_a_scale.contiguous()
    weight_b_scale = weight_b_scale.contiguous()
    weight_a_global_scale = weight_a_global_scale.to(device=x.device).contiguous()
    weight_b_global_scale = weight_b_global_scale.to(device=x.device).contiguous()

    out_features = weight_a_packed.shape[0]
    in_features = weight_a_packed.shape[1] * 2
    if x.numel() != in_features:
        raise ValueError(f"x has {x.numel()} elements, expected {in_features}")

    grid = (triton.cdiv(out_features, block_m),)
    _nvfp4_dual_matvec_kernel[grid](
        x,
        weight_a_packed,
        weight_a_scale,
        weight_a_global_scale,
        weight_b_packed,
        weight_b_scale,
        weight_b_global_scale,
        out_a,
        out_b,
        out_features,
        in_features,
        weight_a_packed.stride(0),
        weight_a_packed.stride(1),
        weight_a_scale.stride(0),
        weight_a_scale.stride(1),
        block_m,
        block_n,
        True,
        num_warps=4,
    )
    return out_a, out_b


def quantize_fp4_m1_native(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one activation row for torch._scaled_mm native FP4.

    Returns packed activation bytes `[1, K/2]` and swizzled FP8 scale_a.
    This is a decode-only M=1 helper for P5-A; batched activation quantization
    is a later kernel.
    """
    _require_triton()
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError(f"quantize_fp4_m1_native expects [K] or [1,K], got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("x must be CUDA tensor")
    if x.shape[1] % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {x.shape[1]}")
    k = x.shape[1]
    groups = k // 16
    packed = torch.empty((1, k // 2), device=x.device, dtype=torch.uint8)
    scale_len = max(1, 128) * max(groups, 4)
    scale = torch.ones((scale_len,), device=x.device, dtype=torch.float8_e4m3fn)
    grid = (groups,)
    _quantize_fp4_m1_kernel[grid](
        x,
        packed,
        scale,
        k,
        groups,
        num_warps=1,
    )
    return packed, scale
