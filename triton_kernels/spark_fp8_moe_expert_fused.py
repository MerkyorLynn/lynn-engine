"""Spark sm_121 FP8 fused gate/up + SwiGLU kernel for one MoE expert.

This variant is tuned for the small per-expert intermediate sizes used by MoE
layers.  It consumes the activation slice routed to one expert and that
expert's FP8 gate/up weights, then emits the BF16 intermediate consumed by the
expert down projection.
"""
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


FP8_E4M3_MAX = 448.0


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for FP8 MoE expert fused kernel")


def fp8_activation_scale(x_bf16: torch.Tensor) -> torch.Tensor:
    """Return per-token FP8 E4M3 activation scales for a [M, K] BF16 tensor."""
    if x_bf16.dtype != torch.bfloat16:
        raise ValueError(f"activation must be BF16, got {x_bf16.dtype}")
    return (x_bf16.abs().amax(dim=-1).clamp_min(1.0e-12) / FP8_E4M3_MAX).to(torch.float32)


if HAS_TRITON:

    @triton.jit
    def _fp8_moe_expert_gate_up_silu_kernel(
        # Pointers
        x_ptr,                     # [M, K] BF16 activation slice for one expert
        x_scale_ptr,               # [M] F32 per-token activation scale
        w_gate_ptr,                # [N, K] FP8 E4M3 gate weight for one expert
        w_up_ptr,                  # [N, K] FP8 E4M3 up weight for one expert
        w_gate_scale_ptr,          # [N] F32 per-row gate weight scale
        w_up_scale_ptr,            # [N] F32 per-row up weight scale
        out_ptr,                   # [M, N] BF16 output = silu(gate) * up
        # Sizes
        M, K, N,
        # Strides
        stride_xm, stride_xk,
        stride_wm, stride_wk,
        stride_om, stride_on,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, K, BLOCK_K):
            k_offs = k_block + offs_k
            mask_k = k_offs < K

            x_block = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)
            x_fp8 = (x_block.to(tl.float32) / x_scale[:, None]).to(tl.float8e4nv)

            w_gate_block = tl.load(
                w_gate_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )
            w_up_block = tl.load(
                w_up_ptr + offs_n[:, None] * stride_wm + k_offs[None, :] * stride_wk,
                mask=mask_n[:, None] & mask_k[None, :],
                other=tl.zeros((1,), dtype=tl.float8e4nv),
            )

            acc_gate += tl.dot(x_fp8, w_gate_block.trans(), out_dtype=tl.float32)
            acc_up += tl.dot(x_fp8, w_up_block.trans(), out_dtype=tl.float32)

        w_gate_scale = tl.load(w_gate_scale_ptr + offs_n, mask=mask_n, other=1.0)
        w_up_scale = tl.load(w_up_scale_ptr + offs_n, mask=mask_n, other=1.0)
        x_scale = tl.load(x_scale_ptr + offs_m, mask=mask_m, other=1.0)

        gate = acc_gate * (x_scale[:, None] * w_gate_scale[None, :])
        up = acc_up * (x_scale[:, None] * w_up_scale[None, :])
        inter = (gate * tl.sigmoid(gate)) * up

        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            inter.to(tl.bfloat16),
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _select_expert_tensor(t: torch.Tensor, expert_id: int | None, name: str) -> torch.Tensor:
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        if expert_id is None:
            raise ValueError(f"expert_id is required when {name} has shape {tuple(t.shape)}")
        return t[expert_id]
    raise ValueError(f"{name} must have shape [N, K] or [E, N, K], got {tuple(t.shape)}")


def _select_expert_scale(t: torch.Tensor, expert_id: int | None, name: str) -> torch.Tensor:
    if t.ndim == 1:
        return t
    if t.ndim == 2:
        if expert_id is None:
            raise ValueError(f"expert_id is required when {name} has shape {tuple(t.shape)}")
        return t[expert_id]
    raise ValueError(f"{name} must have shape [N] or [E, N], got {tuple(t.shape)}")


def fp8_moe_expert_gate_up_silu_fused(
    x_bf16: torch.Tensor,             # [M, K], tokens routed to one expert
    w_gate_fp8: torch.Tensor,         # [N, K] or [E, N, K] FP8 E4M3
    w_up_fp8: torch.Tensor,           # [N, K] or [E, N, K] FP8 E4M3
    w_gate_scale: torch.Tensor,       # [N] or [E, N] F32
    w_up_scale: torch.Tensor,         # [N] or [E, N] F32
    *,
    expert_id: int | None = None,
    x_scale: torch.Tensor | None = None,
    block_m: int = 16,
    block_k: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """Run fused FP8 gate/up + SwiGLU for the tokens routed to one expert.

    Returns BF16 intermediate [M, N] ready for the selected expert's down_proj.
    If stacked [E, ...] weights/scales are provided, pass ``expert_id`` to select
    the expert without copying.  ``x_scale`` can be precomputed once per routed
    activation slice and reused by benchmark/integration callers.
    """
    _require_triton()
    if not x_bf16.is_cuda:
        raise ValueError("activation must be CUDA tensor")
    if x_bf16.dtype != torch.bfloat16:
        raise ValueError(f"activation must be BF16, got {x_bf16.dtype}")

    w_gate_fp8 = _select_expert_tensor(w_gate_fp8, expert_id, "w_gate_fp8")
    w_up_fp8 = _select_expert_tensor(w_up_fp8, expert_id, "w_up_fp8")
    w_gate_scale = _select_expert_scale(w_gate_scale, expert_id, "w_gate_scale")
    w_up_scale = _select_expert_scale(w_up_scale, expert_id, "w_up_scale")

    if w_gate_fp8.dtype != torch.float8_e4m3fn or w_up_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError("weights must be float8_e4m3fn")
    if w_gate_fp8.shape != w_up_fp8.shape:
        raise ValueError("gate and up weights must have same shape")
    if w_gate_scale.dtype != torch.float32 or w_up_scale.dtype != torch.float32:
        raise ValueError("scales must be float32")

    M, K = x_bf16.shape
    N, K_w = w_gate_fp8.shape
    if K != K_w:
        raise ValueError(f"K mismatch: act K={K}, weight K={K_w}")
    if w_gate_scale.shape != (N,) or w_up_scale.shape != (N,):
        raise ValueError("scale tensors must match weight row count")

    if x_scale is None:
        x_scale = fp8_activation_scale(x_bf16)
    else:
        if x_scale.shape != (M,):
            raise ValueError(f"x_scale must have shape ({M},), got {tuple(x_scale.shape)}")
        if x_scale.dtype != torch.float32:
            raise ValueError(f"x_scale must be float32, got {x_scale.dtype}")
        if x_scale.device != x_bf16.device:
            raise ValueError("x_scale must be on the same device as activation")

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_bf16.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _fp8_moe_expert_gate_up_silu_kernel[grid](
        x_bf16,
        x_scale,
        w_gate_fp8,
        w_up_fp8,
        w_gate_scale,
        w_up_scale,
        out,
        M, K, N,
        x_bf16.stride(0), x_bf16.stride(1),
        w_gate_fp8.stride(0), w_gate_fp8.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
    )
    return out


def fp8_moe_expert_gate_up_silu_reference(
    x_bf16: torch.Tensor,
    w_gate_bf16: torch.Tensor,     # [N, K] BF16 reference weight
    w_up_bf16: torch.Tensor,       # [N, K] BF16 reference weight
) -> torch.Tensor:
    """BF16 reference for one expert: silu(x @ gate^T) * (x @ up^T)."""
    gate = torch.nn.functional.linear(x_bf16, w_gate_bf16)
    up = torch.nn.functional.linear(x_bf16, w_up_bf16)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


__all__ = [
    "HAS_TRITON",
    "fp8_activation_scale",
    "fp8_moe_expert_gate_up_silu_fused",
    "fp8_moe_expert_gate_up_silu_reference",
]
