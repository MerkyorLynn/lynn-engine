"""Fused full-attention decode launch-reducers (LYNN_FULL_ATTN_FUSED=1).

Two single-launch kernels for the T=1 full-attention decode hot path
(``engine/incremental_decode.py::decode_full_attn``):

1. ``qk_norm_rope_cache_write_triton`` — per-head Q/K RMSNorm(+1) + partial
   GPT-NeoX RoPE, AND the post-RoPE K + raw-V cache writes, in ONE launch.
   Returns Q only (for SDPA); K_cache/V_cache are written in place at
   ``write_pos``. Replaces: rope-pair launch + K cache-write + V cache-write
   (3 → 1). Numerics mirror ``_qk_norm_rope_pair_kernel`` exactly (fp32 math,
   bf16 store); the V copy is bit-exact (bf16→fp32→bf16 of an already-bf16
   value is identity).

2. ``gate_sigmoid_fold_triton`` — sigmoid(gate) ⊙ attn_out folded with the
   transpose/contiguous reshape into ONE launch, emitting the ``[1, 1, H*D]``
   tensor that feeds o_proj directly. Replaces: float-cast + sigmoid +
   to(dtype) + mul + contiguous (~5 → 1). The sigmoid is rounded to bf16
   *before* the multiply to match ``attn_out * sigmoid(gate.float()).to(dtype)``
   bit-for-bit (bf16×bf16, fp32 accumulate, bf16 round).

The o_proj GEMV itself stays a cuBLAS/packed-NVFP4 call — not worth
re-implementing in Triton at decode shapes.
"""
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
    def _qk_norm_rope_cache_write_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        kcache_ptr,
        vcache_ptr,
        write_pos_ptr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_d: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_d: tl.constexpr,
        q_out_stride_h: tl.constexpr,
        kc_stride_head: tl.constexpr,
        kc_stride_seq: tl.constexpr,
        vc_stride_head: tl.constexpr,
        vc_stride_seq: tl.constexpr,
        Q_HEADS: tl.constexpr,
        K_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Program grid = Q_HEADS + 2*K_HEADS. First Q_HEADS programs are Q
        # heads (→ q_out), next K_HEADS are K heads (norm+rope → K_cache), last
        # K_HEADS are V heads (raw copy → V_cache).
        head = tl.program_id(0)
        is_q = head < Q_HEADS
        is_k = (head >= Q_HEADS) & (head < Q_HEADS + K_HEADS)
        is_v = head >= Q_HEADS + K_HEADS
        local = tl.where(is_q, head, tl.where(is_k, head - Q_HEADS, head - Q_HEADS - K_HEADS))

        in_ptr = tl.where(is_q, q_ptr, tl.where(is_k, k_ptr, v_ptr))
        in_stride_h = tl.where(is_q, q_stride_h, tl.where(is_k, k_stride_h, v_stride_h))
        in_stride_d = tl.where(is_q, q_stride_d, tl.where(is_k, k_stride_d, v_stride_d))
        # V has no q/k_norm weight; point at k_weight (its `y` is discarded below).
        weight_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)

        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        x = tl.load(in_ptr + local * in_stride_h + offs * in_stride_d, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / HEAD_DIM
        inv = tl.rsqrt(var + 1.0e-6)
        y = x * inv * (1.0 + weight)

        half = ROTARY_DIM // 2
        is_first = offs < half
        is_second = (offs >= half) & (offs < ROTARY_DIM)
        in_rot = is_first | is_second
        pair = tl.where(is_second, offs - half, offs)
        pair_offs = tl.where(is_first, offs + half, offs - half)
        cos = tl.load(cos_ptr + pair, mask=in_rot, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + pair, mask=in_rot, other=0.0).to(tl.float32)
        x_pair = tl.load(
            in_ptr + local * in_stride_h + pair_offs * in_stride_d,
            mask=in_rot,
            other=0.0,
        ).to(tl.float32)
        w_pair = tl.load(weight_ptr + pair_offs, mask=in_rot, other=0.0).to(tl.float32)
        y_pair = x_pair * inv * (1.0 + w_pair)
        rotated = tl.where(is_first, y * cos - y_pair * sin, y * cos + y_pair * sin)
        out_qk = tl.where(in_rot, rotated, y)
        # V is a raw copy (no norm, no rope); Q/K get norm+partial-rope.
        out = tl.where(is_v, x, out_qk)

        write_pos = tl.load(write_pos_ptr)
        out_base = tl.where(
            is_q,
            local * q_out_stride_h,
            tl.where(
                is_k,
                local * kc_stride_head + write_pos * kc_stride_seq,
                local * vc_stride_head + write_pos * vc_stride_seq,
            ),
        )
        out_ptr = tl.where(is_q, q_out_ptr, tl.where(is_k, kcache_ptr, vcache_ptr))
        tl.store(out_ptr + out_base + offs, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gate_sigmoid_fold_kernel(
        attn_ptr,
        gate_ptr,
        out_ptr,
        attn_stride_h: tl.constexpr,
        attn_stride_d: tl.constexpr,
        gate_stride_h: tl.constexpr,
        gate_stride_d: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One program per Q head; emits the [1, 1, H*D] contiguous row that
        # feeds o_proj (transpose+contiguous folded in via the h*HEAD_DIM base).
        h = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < HEAD_DIM
        a = tl.load(attn_ptr + h * attn_stride_h + offs * attn_stride_d, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + h * gate_stride_h + offs * gate_stride_d, mask=mask, other=0.0).to(tl.float32)
        # Match torch: sigmoid in fp32, round to bf16, THEN bf16×bf16 multiply.
        sig_b = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
        out = (a * sig_b).to(tl.bfloat16)
        tl.store(out_ptr + h * HEAD_DIM + offs, out, mask=mask)


def qk_norm_rope_cache_write_triton(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    write_pos: torch.Tensor,
) -> torch.Tensor:
    """q/k norm + partial RoPE + post-RoPE K / raw-V cache write, one launch.

    q:        [1, H_Q, 1, D]  (raw, post q_proj+chunk+transpose)
    k_new:    [1, H_KV, 1, D] (raw, post k_proj)
    v_new:    [1, H_KV, 1, D] (raw, post v_proj)
    K_cache,
    V_cache:  [1, H_KV, max_T, D] bf16, written in place at write_pos
    write_pos: long tensor (≥1 elem) on device; element 0 is the cache row.

    Returns: q_out [1, H_Q, 1, D] (normed+roped). K/V are in the caches.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] != 1:
        raise ValueError(f"expected q=[1,H_Q,1,D], got {tuple(q.shape)}")
    if k_new.shape[0] != 1 or k_new.shape[2] != 1 or v_new.shape[0] != 1 or v_new.shape[2] != 1:
        raise ValueError("expected k_new/v_new = [1,H_KV,1,D]")
    if K_cache.dtype != torch.bfloat16 or V_cache.dtype != torch.bfloat16:
        raise ValueError("qk_norm_rope_cache_write_triton requires bf16 K/V caches")
    H_Q = q.shape[1]
    H_KV = k_new.shape[1]
    D = q.shape[-1]
    q_out = torch.empty_like(q)  # contiguous [1, H_Q, 1, D]
    cos1 = cos.reshape(-1)
    sin1 = sin.reshape(-1)
    write_pos = write_pos.reshape(-1)[:1].to(torch.int64)
    block = triton.next_power_of_2(D)
    grid = (H_Q + 2 * H_KV,)
    _qk_norm_rope_cache_write_kernel[grid](
        q,
        k_new,
        v_new,
        q_weight,
        k_weight,
        cos1,
        sin1,
        q_out,
        K_cache,
        V_cache,
        write_pos,
        q.stride(1),
        q.stride(3),
        k_new.stride(1),
        k_new.stride(3),
        v_new.stride(1),
        v_new.stride(3),
        q_out.stride(1),
        K_cache.stride(1),
        K_cache.stride(2),
        V_cache.stride(1),
        V_cache.stride(2),
        Q_HEADS=H_Q,
        K_HEADS=H_KV,
        HEAD_DIM=D,
        ROTARY_DIM=rotary_dim,
        BLOCK=block,
    )
    return q_out


def gate_sigmoid_fold_triton(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``(attn_out * sigmoid(gate)).transpose(1,2).reshape(1,1,H*D)`` in one launch.

    attn_out, gate: [1, H_Q, 1, D]. Returns [1, 1, H_Q*D] (bf16), ready for o_proj.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if attn_out.ndim != 4 or attn_out.shape[0] != 1 or attn_out.shape[2] != 1:
        raise ValueError(f"expected attn_out=[1,H_Q,1,D], got {tuple(attn_out.shape)}")
    if attn_out.dtype != torch.bfloat16:
        raise ValueError("gate_sigmoid_fold_triton requires bf16 attn_out")
    H_Q = attn_out.shape[1]
    D = attn_out.shape[-1]
    out = torch.empty((1, 1, H_Q * D), device=attn_out.device, dtype=attn_out.dtype)
    block = triton.next_power_of_2(D)
    _gate_sigmoid_fold_kernel[(H_Q,)](
        attn_out,
        gate,
        out,
        attn_out.stride(1),
        attn_out.stride(3),
        gate.stride(1),
        gate.stride(3),
        HEAD_DIM=D,
        BLOCK=block,
    )
    return out


__all__ = [
    "HAS_TRITON",
    "qk_norm_rope_cache_write_triton",
    "gate_sigmoid_fold_triton",
]
