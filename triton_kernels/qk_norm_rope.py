"""Fused per-head Q/K RMSNorm(+1 weight) + partial RoPE for decode."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _qk_norm_rope_kernel(
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        stride_x_h: tl.constexpr,
        stride_out_h: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        x = tl.load(x_ptr + h * stride_x_h + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        y = x * tl.rsqrt(var + 1.0e-6) * (1.0 + weight)

        half = ROTARY_DIM // 2
        is_first = offs < half
        is_second = (offs >= half) & (offs < ROTARY_DIM)
        pair = tl.where(is_second, offs - half, offs)
        cos = tl.load(cos_ptr + pair, mask=(is_first | is_second), other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + pair, mask=(is_first | is_second), other=0.0).to(tl.float32)
        y_pair = tl.load(
            x_ptr + h * stride_x_h + tl.where(is_first, offs + half, offs - half),
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        w_pair = tl.load(
            weight_ptr + tl.where(is_first, offs + half, offs - half),
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        # The paired value must use the same RMS denominator.
        y_pair = y_pair * tl.rsqrt(var + 1.0e-6) * (1.0 + w_pair)
        rotated = tl.where(is_first, y * cos - y_pair * sin, y * cos + y_pair * sin)
        out = tl.where(is_first | is_second, rotated, y)
        tl.store(out_ptr + h * stride_out_h + offs, out.to(tl.bfloat16), mask=mask)


    @triton.jit
    def _qk_norm_rope_pair_kernel(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_h: tl.constexpr,
        k_stride_h: tl.constexpr,
        q_out_stride_h: tl.constexpr,
        k_out_stride_h: tl.constexpr,
        Q_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        head = tl.program_id(0)
        is_q = head < Q_HEADS
        local_head = tl.where(is_q, head, head - Q_HEADS)
        x_ptr = tl.where(is_q, q_ptr, k_ptr)
        weight_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)
        out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
        stride_h = tl.where(is_q, q_stride_h, k_stride_h)
        out_stride_h = tl.where(is_q, q_out_stride_h, k_out_stride_h)

        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        x = tl.load(x_ptr + local_head * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        inv = tl.rsqrt(var + 1.0e-6)
        y = x * inv * (1.0 + weight)

        half = ROTARY_DIM // 2
        is_first = offs < half
        is_second = (offs >= half) & (offs < ROTARY_DIM)
        pair = tl.where(is_second, offs - half, offs)
        pair_offs = tl.where(is_first, offs + half, offs - half)
        cos = tl.load(cos_ptr + pair, mask=(is_first | is_second), other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + pair, mask=(is_first | is_second), other=0.0).to(tl.float32)
        x_pair = tl.load(
            x_ptr + local_head * stride_h + pair_offs,
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        w_pair = tl.load(
            weight_ptr + pair_offs,
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        y_pair = x_pair * inv * (1.0 + w_pair)
        rotated = tl.where(is_first, y * cos - y_pair * sin, y * cos + y_pair * sin)
        out = tl.where(is_first | is_second, rotated, y)
        tl.store(out_ptr + local_head * out_stride_h + offs, out.to(tl.bfloat16), mask=mask)


def qk_norm_rope_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[2] != 1:
        raise ValueError("qk_norm_rope_triton expects [1, H, 1, D]")
    heads = x.shape[1]
    head_dim = x.shape[-1]
    x2 = x.reshape(heads, head_dim)
    out = torch.empty_like(x2)
    cos1 = cos.reshape(-1)
    sin1 = sin.reshape(-1)
    block = triton.next_power_of_2(head_dim)
    _qk_norm_rope_kernel[(heads,)](
        x2,
        weight,
        cos1,
        sin1,
        out,
        x2.stride(0),
        out.stride(0),
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK=block,
    )
    return out.reshape_as(x)


def qk_norm_rope_pair_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1 or q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("qk_norm_rope_pair_triton expects q/k as [1, H, 1, D]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k head_dim mismatch")
    q_heads = q.shape[1]
    k_heads = k.shape[1]
    head_dim = q.shape[-1]
    q2 = q.reshape(q_heads, head_dim)
    k2 = k.reshape(k_heads, head_dim)
    q_out = torch.empty_like(q2)
    k_out = torch.empty_like(k2)
    cos1 = cos.reshape(-1)
    sin1 = sin.reshape(-1)
    block = triton.next_power_of_2(head_dim)
    _qk_norm_rope_pair_kernel[(q_heads + k_heads,)](
        q2,
        k2,
        q_weight,
        k_weight,
        cos1,
        sin1,
        q_out,
        k_out,
        q2.stride(0),
        k2.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        Q_HEADS=q_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK=block,
    )
    return q_out.reshape_as(q), k_out.reshape_as(k)


if HAS_TRITON:

    _SP08_QK_NORM_ROPE_PAIR_CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ]

    @triton.autotune(
        configs=_SP08_QK_NORM_ROPE_PAIR_CONFIGS,
        key=["Q_HEADS", "HEAD_DIM", "ROTARY_DIM", "BLOCK"],
    )
    @triton.jit
    def _qk_norm_rope_pair_kernel_sp08_autotuned(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        q_stride_h: tl.constexpr,
        k_stride_h: tl.constexpr,
        q_out_stride_h: tl.constexpr,
        k_out_stride_h: tl.constexpr,
        Q_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        head = tl.program_id(0)
        is_q = head < Q_HEADS
        local_head = tl.where(is_q, head, head - Q_HEADS)
        x_ptr = tl.where(is_q, q_ptr, k_ptr)
        weight_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)
        out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
        stride_h = tl.where(is_q, q_stride_h, k_stride_h)
        out_stride_h = tl.where(is_q, q_out_stride_h, k_out_stride_h)

        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        x = tl.load(x_ptr + local_head * stride_h + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        inv = tl.rsqrt(var + 1.0e-6)
        y = x * inv * (1.0 + weight)

        half = ROTARY_DIM // 2
        is_first = offs < half
        is_second = (offs >= half) & (offs < ROTARY_DIM)
        pair = tl.where(is_second, offs - half, offs)
        pair_offs = tl.where(is_first, offs + half, offs - half)
        cos = tl.load(cos_ptr + pair, mask=(is_first | is_second), other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + pair, mask=(is_first | is_second), other=0.0).to(tl.float32)
        x_pair = tl.load(
            x_ptr + local_head * stride_h + pair_offs,
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        w_pair = tl.load(
            weight_ptr + pair_offs,
            mask=(is_first | is_second),
            other=0.0,
        ).to(tl.float32)
        y_pair = x_pair * inv * (1.0 + w_pair)
        rotated = tl.where(is_first, y * cos - y_pair * sin, y * cos + y_pair * sin)
        out = tl.where(is_first | is_second, rotated, y)
        tl.store(out_ptr + local_head * out_stride_h + offs, out.to(tl.bfloat16), mask=mask)


import os as _os
_SP08_AUTOTUNE = _os.environ.get("LYNN_SP_TRITON_AUTOTUNE", "0") == "1"


def qk_norm_rope_pair_triton_sp08_autotuned(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1 or q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("expects q/k as [1, H, 1, D]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k head_dim mismatch")
    q_heads = q.shape[1]
    k_heads = k.shape[1]
    head_dim = q.shape[-1]
    q2 = q.reshape(q_heads, head_dim)
    k2 = k.reshape(k_heads, head_dim)
    q_out = torch.empty_like(q2)
    k_out = torch.empty_like(k2)
    cos1 = cos.reshape(-1)
    sin1 = sin.reshape(-1)
    block = triton.next_power_of_2(head_dim)
    _qk_norm_rope_pair_kernel_sp08_autotuned[(q_heads + k_heads,)](
        q2,
        k2,
        q_weight,
        k_weight,
        cos1,
        sin1,
        q_out,
        k_out,
        q2.stride(0),
        k2.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        Q_HEADS=q_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        BLOCK=block,
    )
    return q_out.reshape_as(q), k_out.reshape_as(k)


__all__ = [
    "HAS_TRITON",
    "qk_norm_rope_pair_triton",
    "qk_norm_rope_pair_triton_sp08_autotuned",
    "qk_norm_rope_triton",
]
