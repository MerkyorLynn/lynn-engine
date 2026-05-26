"""Row-wise prefix attention kernels for K=2 verifier parity experiments.

The kernel computes one or two decode query rows in a single launch while
keeping independent online-softmax accumulators per row. For ``M=2`` the first
row attends to ``K[:N-1]`` and the second row attends to ``K[:N]``, matching
the prefix-causal K2 verifier shape.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - non-GPU dev hosts.
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for rowwise attention kernels")


if HAS_TRITON:

    @triton.jit
    def _rowwise_prefix_attention_kernel(
        Q,
        K,
        V,
        O,
        H_Q: tl.constexpr,
        H_KV: tl.constexpr,
        M: tl.constexpr,
        N,
        D: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_od: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        b = pid_bh // H_Q
        hq = pid_bh % H_Q
        hkv = hq * H_KV // H_Q

        offs_d = tl.arange(0, D)
        offs_n = tl.arange(0, BLOCK_N)
        q_base = Q + b * stride_qb + hq * stride_qh
        k_base = K + b * stride_kb + hkv * stride_kh
        v_base = V + b * stride_vb + hkv * stride_vh
        o_base = O + b * stride_ob + hq * stride_oh

        q0 = tl.load(q_base + offs_d * stride_qd)
        m0 = tl.full((), -float("inf"), dtype=tl.float32)
        l0 = tl.full((), 0.0, dtype=tl.float32)
        acc0 = tl.zeros((D,), dtype=tl.float32)
        sm_scale = 1.0 / tl.sqrt(D + 0.0)

        if M == 2:
            q1 = tl.load(q_base + stride_qm + offs_d * stride_qd)
            m1 = tl.full((), -float("inf"), dtype=tl.float32)
            l1 = tl.full((), 0.0, dtype=tl.float32)
            acc1 = tl.zeros((D,), dtype=tl.float32)

        n0 = 0
        while n0 < N:
            n_idx = n0 + offs_n
            k = tl.load(
                k_base + n_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=n_idx[:, None] < N,
                other=0.0,
            )
            v = tl.load(
                v_base + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=n_idx[:, None] < N,
                other=0.0,
            )

            qk0 = tl.sum(k * q0[None, :], axis=1) * sm_scale
            if M == 2:
                qk0 = tl.where(n_idx < N - 1, qk0, -float("inf"))
            else:
                qk0 = tl.where(n_idx < N, qk0, -float("inf"))
            m0_new = tl.maximum(m0, tl.max(qk0, axis=0))
            p0 = tl.exp(qk0 - m0_new)
            alpha0 = tl.exp(m0 - m0_new)
            l0 = l0 * alpha0 + tl.sum(p0, axis=0)
            acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v, axis=0)
            m0 = m0_new

            if M == 2:
                qk1 = tl.sum(k * q1[None, :], axis=1) * sm_scale
                qk1 = tl.where(n_idx < N, qk1, -float("inf"))
                m1_new = tl.maximum(m1, tl.max(qk1, axis=0))
                p1 = tl.exp(qk1 - m1_new)
                alpha1 = tl.exp(m1 - m1_new)
                l1 = l1 * alpha1 + tl.sum(p1, axis=0)
                acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v, axis=0)
                m1 = m1_new
            n0 += BLOCK_N

        out0 = acc0 / l0
        tl.store(o_base + offs_d * stride_od, out0.to(O.dtype.element_ty))
        if M == 2:
            out1 = acc1 / l1
            tl.store(o_base + stride_om + offs_d * stride_od, out1.to(O.dtype.element_ty))


def rowwise_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, block_n: int = 64) -> torch.Tensor:
    """Prefix-causal decode attention for one or two query rows.

    Args:
        q: CUDA tensor ``[B, H_Q, M, D]`` where ``M`` is 1 or 2.
        k/v: CUDA tensor ``[B, H_KV, N, D]``.
    """
    _require_triton()
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"expected q/k/v rank 4, got {q.shape=} {k.shape=} {v.shape=}")
    b, h_q, m, d = q.shape
    bk, h_kv, n, dk = k.shape
    bv, hv, nv, dv = v.shape
    if (bk, bv) != (b, b) or hv != h_kv or nv != n or dk != d or dv != d:
        raise ValueError(f"incompatible q/k/v shapes: {q.shape=} {k.shape=} {v.shape=}")
    if m not in (1, 2):
        raise ValueError(f"rowwise_prefix_attention supports M=1 or M=2, got {m}")
    if h_q % h_kv != 0:
        raise ValueError(f"H_Q must be divisible by H_KV, got {h_q=} {h_kv=}")
    if d not in (64, 128, 256):
        raise ValueError(f"unsupported head_dim={d}")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    out = torch.empty_like(q)
    grid = (b * h_q,)
    _rowwise_prefix_attention_kernel[grid](
        q,
        k,
        v,
        out,
        h_q,
        h_kv,
        m,
        n,
        d,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


__all__ = ["rowwise_prefix_attention"]
