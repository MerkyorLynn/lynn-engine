"""Triton helpers for the BF16 shared-expert scalar gate."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - import guard for CPU-only dev envs
    triton = None
    tl = None
    HAS_TRITON = False


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for shared expert gate kernels")


if HAS_TRITON:

    @triton.jit
    def _shared_expert_gate_apply_kernel(
        shared_ptr,
        hidden_ptr,
        gate_weight_ptr,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        hidden = tl.load(hidden_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(hidden * weight, axis=0)
        gate = tl.sigmoid(dot)
        shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, (shared * gate).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _shared_expert_gate_add_from_scalar_kernel(
        moe_ptr,
        shared_ptr,
        gate_ptr,
        out_ptr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_D)
        mask = offsets < D
        gate = tl.load(gate_ptr).to(tl.float32)
        moe = tl.load(moe_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        shared = tl.load(shared_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, (moe + shared * gate).to(tl.bfloat16), mask=mask)


def apply_shared_expert_gate_triton(
    shared: torch.Tensor,
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the Qwen shared-expert scalar gate in one Triton kernel.

    This preserves the BF16 shared expert path but replaces the tiny
    `F.linear -> sigmoid -> broadcast multiply` chain with one graph-capturable
    kernel for batch=1 decode.
    """
    _require_triton()
    if shared.ndim != 2 or shared.shape[0] != 1:
        raise ValueError(f"shared must be [1, D], got {tuple(shared.shape)}")
    if hidden.ndim != 2 or hidden.shape[0] != 1 or hidden.shape[1] != shared.shape[1]:
        raise ValueError(f"hidden must be [1, {shared.shape[1]}], got {tuple(hidden.shape)}")
    if gate_weight.ndim != 2 or gate_weight.shape[0] != 1 or gate_weight.shape[1] != shared.shape[1]:
        raise ValueError(f"gate_weight must be [1, {shared.shape[1]}], got {tuple(gate_weight.shape)}")
    d = int(shared.shape[1])
    block_d = 1 << (d - 1).bit_length()
    out = torch.empty_like(shared)
    _shared_expert_gate_apply_kernel[(1,)](
        shared.contiguous(),
        hidden.contiguous(),
        gate_weight.contiguous(),
        out,
        D=d,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out


def add_shared_expert_gate_from_scalar_triton(
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse `moe_out + shared * gate` when the scalar gate was computed by Torch.

    This is a research backend for decode-only batch=1.  Unlike
    `apply_shared_expert_gate_triton`, it intentionally leaves the scalar gate
    reduction on the default Torch path and only removes the final broadcast
    multiply/add boundary.
    """
    _require_triton()
    if moe_out.ndim != 2 or moe_out.shape[0] != 1:
        raise ValueError(f"moe_out must be [1, D], got {tuple(moe_out.shape)}")
    if shared.shape != moe_out.shape:
        raise ValueError(f"shared must match moe_out, got {tuple(shared.shape)} vs {tuple(moe_out.shape)}")
    if gate.numel() != 1:
        raise ValueError(f"gate must contain one scalar for batch=1 decode, got {tuple(gate.shape)}")
    d = int(moe_out.shape[1])
    block_d = 1 << (d - 1).bit_length()
    out = torch.empty_like(moe_out)
    _shared_expert_gate_add_from_scalar_kernel[(1,)](
        moe_out.contiguous(),
        shared.contiguous(),
        gate.contiguous(),
        out,
        D=d,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return out


__all__ = ["HAS_TRITON", "apply_shared_expert_gate_triton", "add_shared_expert_gate_from_scalar_triton"]
