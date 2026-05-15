"""Router top-k helpers for decode-time MoE."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for router top-k kernels")


if HAS_TRITON:

    @triton.jit
    def _router_topk_softmax_kernel(
        logits_ptr,
        weights_ptr,
        indices_ptr,
        N: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(logits_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        top_vals = tl.zeros((TOP_K,), dtype=tl.float32)
        top_ids = tl.zeros((TOP_K,), dtype=tl.int32)

        for slot in range(0, TOP_K):
            max_val = tl.max(vals, axis=0)
            # Stable enough for router use: choose the smallest expert id on
            # exact ties. torch.topk(sorted=False) does not promise stable
            # ordering, and active expert accumulation is commutative.
            candidate_ids = tl.where(vals == max_val, offsets, BLOCK_N + offsets)
            max_id = tl.min(candidate_ids, axis=0)
            top_vals = tl.where(tl.arange(0, TOP_K) == slot, max_val, top_vals)
            top_ids = tl.where(tl.arange(0, TOP_K) == slot, max_id.to(tl.int32), top_ids)
            vals = tl.where(offsets == max_id, -float("inf"), vals)

        top_max = tl.max(top_vals, axis=0)
        exp_vals = tl.exp(top_vals - top_max)
        denom = tl.sum(exp_vals, axis=0)
        weights = exp_vals / denom
        slots = tl.arange(0, TOP_K)
        tl.store(weights_ptr + slots, weights)
        tl.store(indices_ptr + slots, top_ids)


def router_topk_softmax_triton(logits_1d: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(routing_weights, expert_indices)` for one router logits vector.

    This fuses top-k selection and softmax normalization into one Triton launch.
    The router linear remains the normal BF16 `F.linear` path.
    """
    _require_triton()
    if logits_1d.ndim != 1:
        raise ValueError(f"logits_1d must be 1D, got {tuple(logits_1d.shape)}")
    n = int(logits_1d.numel())
    if not (1 <= top_k <= n):
        raise ValueError(f"top_k must be in [1, {n}], got {top_k}")
    block_n = triton.next_power_of_2(n)
    weights = torch.empty((top_k,), device=logits_1d.device, dtype=torch.float32)
    indices = torch.empty((top_k,), device=logits_1d.device, dtype=torch.int64)
    _router_topk_softmax_kernel[(1,)](
        logits_1d.contiguous(),
        weights,
        indices,
        N=n,
        TOP_K=top_k,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return weights, indices


__all__ = ["HAS_TRITON", "router_topk_softmax_triton"]
