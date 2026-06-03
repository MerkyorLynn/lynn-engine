"""Decode-time fused BF16 shared-expert kernels (LYNN_SHARED_EXPERT_FUSED=1).

Two single-launch kernels for the T=1 / M=1 decode shared-expert tail in
``engine/moe_packed_nvfp4.py`` (the eager block at
``_moe_forward_decode_packed_nvfp4_fixed_triton`` lines ~826-834). They replace
the chain

    gate_up = F.linear(h, _gate_up_proj.weight)      # 1 GEMM launch
    gate, up = gate_up.chunk(2)                       # view
    inter   = F.silu(gate) * up                       # 2 launches (silu, mul)
    shared  = F.linear(inter, down_proj.weight)       # 1 GEMM launch
    shared  = shared * sigmoid(F.linear(h, gate.w))   # 3 launches (linear, sigmoid, mul)
    moe_out = moe_out + shared                        # 1 launch
                                                       # ── ~8 eager launches ──

with exactly TWO Triton launches:

1. ``_shared_gate_up_silu_kernel`` — a parallel GEMV over the intermediate dim
   that computes ``gate_up`` AND the SwiGLU ``silu(gate) * up`` in one launch,
   storing the BF16 intermediate ``[1, I]``. (grid = ceil(I / BLOCK_I))
2. ``_shared_down_gate_add_kernel`` — a parallel GEMV over the hidden dim that
   computes ``inter @ down.T``, the optional scalar gate
   ``sigmoid(h @ shared_expert_gate.T)``, and the residual add into ``moe_out``
   in one launch, writing BF16 in place. (grid = ceil(H / BLOCK_D))

Design notes (why 2, not 1): the down projection reduces over the FULL
intermediate vector, so it needs every ``inter[i]`` produced by kernel #1
before it can start. Triton has no cross-program barrier, so a true single
kernel would have to recompute ``gate_up`` per output block. We instead keep
the BF16 intermediate store and read it back (no recompute) — the same
trade-off the full-attn fusion made by leaving o_proj as a separate GEMM.

Numerics: BF16 in / BF16 out, fp32 accumulation (matching cuBLAS bf16-input /
fp32-accumulate). At each materialization boundary we round to BF16 to mirror
the eager torch path's dtype staging:
``gate_s``/``up_s`` (BF16 chunks) -> ``F.silu(gate_s)`` (BF16) ->
``F.silu(gate_s) * up_s`` (BF16 intermediate, the cross-kernel anchor) ->
``F.linear`` down output (BF16 "shared") -> ``+ moe_out`` (BF16). The scalar
gate is reduced in fp32 and its sigmoid taken in fp32, matching the proven
``apply_shared_expert_gate_triton`` convention. This is token-coherent
(cos ~ 1), not bit-exact: Triton's reduction order differs from cuBLAS.
"""
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
        raise RuntimeError("Triton is required for fused shared expert kernels")


def _check_pow2(name: str, value: int) -> None:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {value}")


if HAS_TRITON:

    @triton.jit
    def _shared_gate_up_silu_kernel(
        x_ptr,
        gate_up_ptr,
        inter_ptr,
        HIDDEN: tl.constexpr,
        INTER: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_gr: tl.constexpr,
        stride_gh: tl.constexpr,
        stride_ii: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        # One program computes BLOCK_I intermediate elements: a GEMV
        # row-block of gate_up [2I, H] dotted with x [H], then SwiGLU.
        pid_i = tl.program_id(0)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        mask_i = offs_i < INTER

        acc_g = tl.zeros((BLOCK_I,), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for h0 in range(0, HIDDEN, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask_h = offs_h < HIDDEN
            x = tl.load(x_ptr + offs_h * stride_xh, mask=mask_h, other=0.0).to(tl.float32)
            # gate rows are 0..I-1, up rows are I..2I-1 (cat([gate, up], dim=0)).
            wg = tl.load(
                gate_up_ptr + offs_i[:, None] * stride_gr + offs_h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            wu = tl.load(
                gate_up_ptr + (INTER + offs_i)[:, None] * stride_gr + offs_h[None, :] * stride_gh,
                mask=mask_i[:, None] & mask_h[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_g += tl.sum(x[None, :] * wg, axis=1)
            acc_u += tl.sum(x[None, :] * wu, axis=1)

        # Mirror eager dtype staging: gate_s/up_s are BF16 chunks of gate_up,
        # F.silu(gate_s) is BF16, and (F.silu(gate_s) * up_s) is the BF16
        # intermediate fed to the down projection.
        g_b = acc_g.to(tl.bfloat16).to(tl.float32)
        u_b = acc_u.to(tl.bfloat16).to(tl.float32)
        silu_b = (g_b * tl.sigmoid(g_b)).to(tl.bfloat16).to(tl.float32)
        inter = (silu_b * u_b).to(tl.bfloat16)
        tl.store(inter_ptr + offs_i * stride_ii, inter, mask=mask_i)

    @triton.jit
    def _shared_down_gate_add_kernel(
        inter_ptr,
        down_ptr,
        moe_ptr,
        x_ptr,
        gate_ptr,
        out_ptr,
        HIDDEN: tl.constexpr,
        INTER: tl.constexpr,
        stride_ii: tl.constexpr,
        stride_dr: tl.constexpr,
        stride_di: tl.constexpr,
        stride_mh: tl.constexpr,
        stride_xh: tl.constexpr,
        stride_gh: tl.constexpr,
        HAS_GATE: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # One program computes BLOCK_D hidden outputs: a GEMV row-block of
        # down [H, I] dotted with the stored intermediate [I], plus the
        # optional scalar gate and residual add into moe_out (in place).
        pid_d = tl.program_id(0)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < HIDDEN

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for i0 in range(0, INTER, BLOCK_I):
            offs_i = i0 + tl.arange(0, BLOCK_I)
            mask_i = offs_i < INTER
            inter = tl.load(inter_ptr + offs_i * stride_ii, mask=mask_i, other=0.0).to(tl.float32)
            wd = tl.load(
                down_ptr + offs_d[:, None] * stride_dr + offs_i[None, :] * stride_di,
                mask=mask_d[:, None] & mask_i[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(inter[None, :] * wd, axis=1)

        gate = 1.0
        if HAS_GATE:
            # Scalar gate sigmoid(h . gate_weight); reduced in fp32. Recomputed
            # per program (cheap: one H-length dot) to avoid a cross-kernel
            # scratch and an extra launch.
            gate_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for h0 in range(0, HIDDEN, BLOCK_H):
                offs_h = h0 + tl.arange(0, BLOCK_H)
                mask_h = offs_h < HIDDEN
                x = tl.load(x_ptr + offs_h * stride_xh, mask=mask_h, other=0.0).to(tl.float32)
                gw = tl.load(gate_ptr + offs_h * stride_gh, mask=mask_h, other=0.0).to(tl.float32)
                gate_acc += tl.where(mask_h, x * gw, 0.0)
            gate = tl.sigmoid(tl.sum(gate_acc, axis=0))

        # "shared" is the BF16 down output in the eager path; round to match,
        # then fp32 (moe + shared * gate) -> BF16 store.
        shared_b = acc.to(tl.bfloat16).to(tl.float32)
        moe = tl.load(moe_ptr + offs_d * stride_mh, mask=mask_d, other=0.0).to(tl.float32)
        out = (moe + shared_b * gate).to(tl.bfloat16)
        tl.store(out_ptr + offs_d * stride_mh, out, mask=mask_d)


def shared_expert_decode_fused_triton(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor | None = None,
    *,
    inter: torch.Tensor | None = None,
    block_hidden: int = 128,
    block_inter: int = 32,
    block_out: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fused BF16 shared-expert for M=1 decode in two Triton launches.

    h_flat:          [1, H]  BF16 decode hidden state
    moe_out:         [1, H]  BF16 routed-MoE output; the gated shared-expert
                              output is ADDED into this tensor in place.
    gate_up_weight:  [2I, H] BF16, cat([gate_proj, up_proj], dim=0)
    down_weight:     [H, I]  BF16 down projection
    gate_weight:     [1, H]  BF16 scalar gate (optional; None -> no gate scale)
    inter:           [1, I]  BF16 caller-owned scratch (optional; allocated if None)

    Returns ``moe_out`` (same tensor, written in place).
    """
    _require_triton()
    if h_flat.ndim != 2 or h_flat.shape[0] != 1:
        raise ValueError(f"h_flat must be [1, H], got {tuple(h_flat.shape)}")
    if h_flat.dtype != torch.bfloat16 or moe_out.dtype != torch.bfloat16:
        raise ValueError("h_flat and moe_out must be BF16")
    if moe_out.shape != h_flat.shape:
        raise ValueError(f"moe_out must match h_flat {tuple(h_flat.shape)}, got {tuple(moe_out.shape)}")
    hidden = int(h_flat.shape[1])
    if gate_up_weight.ndim != 2 or gate_up_weight.shape[1] != hidden or gate_up_weight.shape[0] % 2 != 0:
        raise ValueError(f"gate_up_weight must be [2 * I, {hidden}], got {tuple(gate_up_weight.shape)}")
    intermediate = int(gate_up_weight.shape[0] // 2)
    if down_weight.ndim != 2 or tuple(down_weight.shape) != (hidden, intermediate):
        raise ValueError(f"down_weight must be [{hidden}, {intermediate}], got {tuple(down_weight.shape)}")
    if gate_weight is not None and (
        gate_weight.ndim != 2 or tuple(gate_weight.shape) != (1, hidden)
    ):
        raise ValueError(f"gate_weight must be [1, {hidden}], got {tuple(gate_weight.shape)}")
    if gate_up_weight.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise ValueError("gate_up_weight and down_weight must be BF16")
    if gate_weight is not None and gate_weight.dtype != torch.bfloat16:
        raise ValueError("gate_weight must be BF16")
    _check_pow2("block_hidden", block_hidden)
    _check_pow2("block_inter", block_inter)
    _check_pow2("block_out", block_out)

    h_flat = h_flat.contiguous()
    moe_out = moe_out.contiguous()
    gate_up_weight = gate_up_weight.contiguous()
    down_weight = down_weight.contiguous()
    if inter is None:
        inter = torch.empty((1, intermediate), device=h_flat.device, dtype=torch.bfloat16)
    else:
        if inter.shape != (1, intermediate) or inter.dtype != torch.bfloat16:
            raise ValueError(f"inter scratch must be BF16 [1, {intermediate}]")
        inter = inter.contiguous()

    _shared_gate_up_silu_kernel[(triton.cdiv(intermediate, block_inter),)](
        h_flat,
        gate_up_weight,
        inter,
        HIDDEN=hidden,
        INTER=intermediate,
        stride_xh=h_flat.stride(1),
        stride_gr=gate_up_weight.stride(0),
        stride_gh=gate_up_weight.stride(1),
        stride_ii=inter.stride(1),
        BLOCK_H=block_hidden,
        BLOCK_I=block_inter,
        num_warps=num_warps,
    )

    has_gate = gate_weight is not None
    gate_arg = gate_weight.contiguous() if has_gate else h_flat
    _shared_down_gate_add_kernel[(triton.cdiv(hidden, block_out),)](
        inter,
        down_weight,
        moe_out,
        h_flat,
        gate_arg,
        moe_out,
        HIDDEN=hidden,
        INTER=intermediate,
        stride_ii=inter.stride(1),
        stride_dr=down_weight.stride(0),
        stride_di=down_weight.stride(1),
        stride_mh=moe_out.stride(1),
        stride_xh=h_flat.stride(1),
        stride_gh=gate_arg.stride(1),
        HAS_GATE=has_gate,
        BLOCK_I=block_inter,
        BLOCK_D=block_out,
        BLOCK_H=block_hidden,
        num_warps=num_warps,
    )
    return moe_out


__all__ = ["HAS_TRITON", "shared_expert_decode_fused_triton"]
