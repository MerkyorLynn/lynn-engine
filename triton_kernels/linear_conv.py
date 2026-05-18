"""Triton single-token depthwise conv for Qwen3.6 linear-attention decode."""
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
    def _linear_conv1d_update_kernel(
        mixed_ptr,
        state_ptr,
        weight_ptr,
        out_ptr,
        new_state_ptr,
        C: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < C

        s0 = tl.load(state_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(state_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(state_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(mixed_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        w0 = tl.load(weight_ptr + offs * 4 + 0, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(weight_ptr + offs * 4 + 1, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(weight_ptr + offs * 4 + 2, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(weight_ptr + offs * 4 + 3, mask=mask, other=0.0).to(tl.float32)

        acc = s0 * w0 + s1 * w1 + s2 * w2 + x * w3
        if APPLY_SILU:
            out = acc * tl.sigmoid(acc)
        else:
            out = acc
        tl.store(out_ptr + offs, out, mask=mask)

        tl.store(new_state_ptr + offs * 3 + 0, s1, mask=mask)
        tl.store(new_state_ptr + offs * 3 + 1, s2, mask=mask)
        tl.store(new_state_ptr + offs * 3 + 2, x, mask=mask)


def linear_conv1d_update_triton(
    mixed_new: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    inplace: bool = False,
    torch_silu: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run decode conv update for one token.

    Args:
      mixed_new: `[1, conv_dim, 1]`
      conv_state: `[1, conv_dim, 3]`
      conv_weight: `[conv_dim, 1, 4]`

    Returns:
      out: `[1, 1, conv_dim]`
      new_state: `[1, conv_dim, 3]`, optionally the same tensor as
        `conv_state` when `inplace=True`.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for linear_conv1d_update_triton")
    if mixed_new.ndim != 3 or mixed_new.shape[0] != 1 or mixed_new.shape[2] != 1:
        raise ValueError(f"mixed_new must be [1,C,1], got {tuple(mixed_new.shape)}")
    if conv_state.ndim != 3 or conv_state.shape[0] != 1 or conv_state.shape[2] != 3:
        raise ValueError(f"conv_state must be [1,C,3], got {tuple(conv_state.shape)}")
    if conv_weight.ndim != 3 or conv_weight.shape[1:] != (1, 4):
        raise ValueError(f"conv_weight must be [C,1,4], got {tuple(conv_weight.shape)}")
    c = int(mixed_new.shape[1])
    if int(conv_state.shape[1]) != c or int(conv_weight.shape[0]) != c:
        raise ValueError(
            f"conv dim mismatch: mixed={tuple(mixed_new.shape)} "
            f"state={tuple(conv_state.shape)} weight={tuple(conv_weight.shape)}"
        )
    out = torch.empty((1, 1, c), device=mixed_new.device, dtype=mixed_new.dtype)
    new_state = conv_state if inplace else torch.empty_like(conv_state)
    block = 256
    _linear_conv1d_update_kernel[(triton.cdiv(c, block),)](
        mixed_new.reshape(c),
        conv_state.reshape(c, 3),
        conv_weight.reshape(c, 4),
        out.reshape(c),
        new_state.reshape(c, 3),
        C=c,
        APPLY_SILU=not torch_silu,
        BLOCK=block,
        num_warps=4,
    )
    if torch_silu:
        out = torch.nn.functional.silu(out)
    return out, new_state


__all__ = ["HAS_TRITON", "linear_conv1d_update_triton"]
