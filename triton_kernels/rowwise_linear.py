"""Row-wise Linear kernels for K=2 verifier parity experiments.

These kernels deliberately compute each token row with an independent
accumulator. The goal is not to beat cuBLAS yet; it is to prove a dispatch
shape that can process K=2 rows in one launch while preserving the same
per-row accumulation contract as separate T=1 calls.
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
        raise RuntimeError("Triton is required for rowwise linear kernels")


if HAS_TRITON:

    @triton.jit
    def _rowwise_linear_kernel(
        X,
        W,
        Y,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_ym: tl.constexpr,
        stride_yn: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < K
            w = tl.load(
                W + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
                mask=(offs_n[:, None] < N) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                X + k_idx * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            acc0 += tl.sum(w * x0[None, :], axis=1)
            if M == 2:
                x1 = tl.load(
                    X + stride_xm + k_idx * stride_xk,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                acc1 += tl.sum(w * x1[None, :], axis=1)

        tl.store(Y + offs_n * stride_yn, acc0, mask=offs_n < N)
        if M == 2:
            tl.store(Y + stride_ym + offs_n * stride_yn, acc1, mask=offs_n < N)


def rowwise_linear(x: torch.Tensor, weight: torch.Tensor, *, block_n: int = 32, block_k: int = 64) -> torch.Tensor:
    """Compute ``F.linear(x, weight)`` for one or two rows.

    Args:
        x: CUDA tensor ``[M, K]`` where ``M`` is 1 or 2.
        weight: CUDA tensor ``[N, K]``.
    """
    _require_triton()
    if x.ndim != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be [N, K], got {tuple(weight.shape)}")
    if x.shape[0] not in (1, 2):
        raise ValueError(f"rowwise_linear supports M=1 or M=2, got {x.shape[0]}")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(f"K mismatch: x={x.shape[1]} weight={weight.shape[1]}")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")

    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(weight.shape[0], block_n),)
    _rowwise_linear_kernel[grid](
        x,
        weight,
        out,
        x.shape[0],
        x.shape[1],
        weight.shape[0],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


__all__ = ["rowwise_linear"]
