"""Triton RMSNormGated for Qwen3.6 linear-attention decode."""
from __future__ import annotations

import torch
import torch.nn.functional as F

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
    def _rmsnorm_gated_kernel(
        x_ptr,
        gate_ptr,
        weight_ptr,
        out_ptr,
        stride_x_row: tl.constexpr,
        stride_gate_row: tl.constexpr,
        stride_out_row: tl.constexpr,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row * stride_gate_row + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        x_norm = x * tl.rsqrt(var + EPS)
        silu_gate = gate * tl.sigmoid(gate)
        out = (x_norm * weight * silu_gate).to(tl.bfloat16)
        tl.store(out_ptr + row * stride_out_row + offs, out, mask=mask)


def rms_norm_gated_triton(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if not HAS_TRITON:
        x_f = x.float()
        y = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        return (y.to(x.dtype) * weight * F.silu(gate.float())).to(x.dtype)
    if x.ndim != 2 or gate.ndim != 2:
        raise ValueError("rms_norm_gated_triton expects flat [N, D] tensors")
    if x.shape != gate.shape:
        raise ValueError("x and gate shapes must match")
    n_rows, dim = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(dim)
    _rmsnorm_gated_kernel[(n_rows,)](
        x,
        gate,
        weight,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        D=dim,
        EPS=eps,
        BLOCK=block,
    )
    return out


__all__ = ["HAS_TRITON", "rms_norm_gated_triton"]
