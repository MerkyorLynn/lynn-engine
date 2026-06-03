"""Triton kernels for Qwen3.6 Gated Delta Net decode.

P6 starts with the single-token recurrent update because profiling showed it is
by far the hottest segment of linear-attention decode. These helpers are opt-in;
the PyTorch reference path remains the production fallback until end-to-end gates
pass.
"""
from __future__ import annotations

import math
import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False


HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_V_HEADS = 32
NUM_K_HEADS = 16


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused recurrent gated-delta decode")


if HAS_TRITON:

    @triton.jit
    def _recurrent_fused_prepare_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        s_prev_ptr,
        out_ptr,
        s_new_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_V: tl.constexpr,
    ):
        head = tl.program_id(0)
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

        q_raw = tl.load(q_ptr + head * BLOCK_K + offs_k).to(tl.float32)
        k_raw = tl.load(k_ptr + head * BLOCK_K + offs_k).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1.0e-6)
        q = q_raw * q_norm * 0.08838834764831845  # 1 / sqrt(128)
        k = k_raw * k_norm
        v = tl.load(v_ptr + head * HEAD_V + offs_v).to(tl.float32)
        g_exp = tl.exp(tl.load(g_ptr + head).to(tl.float32))
        beta = tl.load(beta_ptr + head).to(tl.float32)

        s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
        s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
        s_decay = s_prev * g_exp
        kv_mem = tl.sum(s_decay * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s_new = s_decay + k[:, None] * delta[None, :]
        out = tl.sum(s_new * q[:, None], axis=0)
        tl.store(s_new_ptr + s_offsets, s_new)
        tl.store(out_ptr + head * HEAD_V + offs_v, out)

    @triton.jit
    def _recurrent_fused_prepare_gqa_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        s_prev_ptr,
        out_ptr,
        s_new_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_V: tl.constexpr,
        V_PER_K: tl.constexpr,
    ):
        head = tl.program_id(0)
        kv_head = head // V_PER_K
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

        q_raw = tl.load(q_ptr + kv_head * BLOCK_K + offs_k).to(tl.float32)
        k_raw = tl.load(k_ptr + kv_head * BLOCK_K + offs_k).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1.0e-6)
        q = q_raw * q_norm * 0.08838834764831845
        k = k_raw * k_norm
        v = tl.load(v_ptr + head * HEAD_V + offs_v).to(tl.float32)
        g_exp = tl.exp(tl.load(g_ptr + head).to(tl.float32))
        beta = tl.load(beta_ptr + head).to(tl.float32)

        s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
        s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
        s_decay = s_prev * g_exp
        kv_mem = tl.sum(s_decay * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s_new = s_decay + k[:, None] * delta[None, :]
        out = tl.sum(s_new * q[:, None], axis=0)
        tl.store(s_new_ptr + s_offsets, s_new)
        tl.store(out_ptr + head * HEAD_V + offs_v, out)

    @triton.jit
    def _recurrent_fused_prepare_from_outconv_gqa_kernel(
        out_conv_ptr,
        g_ptr,
        beta_ptr,
        s_prev_ptr,
        out_ptr,
        s_new_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_V: tl.constexpr,
        NUM_K: tl.constexpr,
        NUM_V: tl.constexpr,
        V_PER_K: tl.constexpr,
    ):
        head = tl.program_id(0)
        kv_head = head // V_PER_K
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

        q_base = kv_head * BLOCK_K
        k_base = NUM_K * BLOCK_K + kv_head * BLOCK_K
        v_base = NUM_K * BLOCK_K * 2 + head * HEAD_V
        q_raw = tl.load(out_conv_ptr + q_base + offs_k).to(tl.float32)
        k_raw = tl.load(out_conv_ptr + k_base + offs_k).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1.0e-6)
        q = q_raw * q_norm * 0.08838834764831845
        k = k_raw * k_norm
        v = tl.load(out_conv_ptr + v_base + offs_v).to(tl.float32)
        g_exp = tl.exp(tl.load(g_ptr + head).to(tl.float32))
        beta = tl.load(beta_ptr + head).to(tl.float32)

        s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
        s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
        s_decay = s_prev * g_exp
        kv_mem = tl.sum(s_decay * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s_new = s_decay + k[:, None] * delta[None, :]
        out = tl.sum(s_new * q[:, None], axis=0)
        tl.store(s_new_ptr + s_offsets, s_new)
        tl.store(out_ptr + head * HEAD_V + offs_v, out)

    @triton.jit
    def _recurrent_fused_prepare_from_outconv_ab_gqa_kernel(
        out_conv_ptr,
        a_ptr,
        b_ptr,
        neg_exp_A_log_ptr,
        dt_bias_ptr,
        s_prev_ptr,
        out_ptr,
        s_new_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_V: tl.constexpr,
        NUM_K: tl.constexpr,
        NUM_V: tl.constexpr,
        V_PER_K: tl.constexpr,
    ):
        head = tl.program_id(0)
        kv_head = head // V_PER_K
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

        q_base = kv_head * BLOCK_K
        k_base = NUM_K * BLOCK_K + kv_head * BLOCK_K
        v_base = NUM_K * BLOCK_K * 2 + head * HEAD_V
        q_raw = tl.load(out_conv_ptr + q_base + offs_k).to(tl.float32)
        k_raw = tl.load(out_conv_ptr + k_base + offs_k).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1.0e-6)
        q = q_raw * q_norm * 0.08838834764831845
        k = k_raw * k_norm
        v = tl.load(out_conv_ptr + v_base + offs_v).to(tl.float32)

        b_val = tl.load(b_ptr + head).to(tl.float32)
        beta = 1.0 / (1.0 + tl.exp(-b_val))
        a_dt = tl.load(a_ptr + head).to(tl.float32) + tl.load(dt_bias_ptr + head).to(tl.float32)
        softplus = tl.log(1.0 + tl.exp(a_dt))
        g_value = tl.load(neg_exp_A_log_ptr + head).to(tl.float32) * softplus
        g_exp = tl.exp(g_value)

        s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
        s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
        s_decay = s_prev * g_exp
        kv_mem = tl.sum(s_decay * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s_new = s_decay + k[:, None] * delta[None, :]
        out = tl.sum(s_new * q[:, None], axis=0)
        tl.store(s_new_ptr + s_offsets, s_new)
        tl.store(out_ptr + head * HEAD_V + offs_v, out)

    @triton.jit
    def _recurrent_fused_prepare_from_outconv_gqa_gbeta_kernel(
        out_conv_ptr,
        a_ptr,
        b_ptr,
        dt_bias_ptr,
        neg_exp_A_log_ptr,
        s_prev_ptr,
        out_ptr,
        s_new_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
        HEAD_V: tl.constexpr,
        NUM_K: tl.constexpr,
        NUM_V: tl.constexpr,
        V_PER_K: tl.constexpr,
    ):
        # Stage-3 launch-fold. Byte-for-byte identical to
        # `_recurrent_fused_prepare_from_outconv_gqa_kernel` (qk-l2norm, decay,
        # kv_mem, delta, state-store, out) EXCEPT g/beta are computed per-head in
        # float32 here from the raw projection outputs (a, b) + per-head dt_bias /
        # neg_exp_A_log, instead of being pre-computed in PyTorch and passed in.
        head = tl.program_id(0)
        kv_head = head // V_PER_K
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

        q_base = kv_head * BLOCK_K
        k_base = NUM_K * BLOCK_K + kv_head * BLOCK_K
        v_base = NUM_K * BLOCK_K * 2 + head * HEAD_V
        q_raw = tl.load(out_conv_ptr + q_base + offs_k).to(tl.float32)
        k_raw = tl.load(out_conv_ptr + k_base + offs_k).to(tl.float32)
        q_norm = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) + 1.0e-6)
        k_norm = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) + 1.0e-6)
        q = q_raw * q_norm * 0.08838834764831845
        k = k_raw * k_norm
        v = tl.load(out_conv_ptr + v_base + offs_v).to(tl.float32)

        # g/beta folded in (float32). Mirrors the PyTorch elementwise ops:
        #   beta = b.sigmoid()
        #   g    = neg_exp_A_log * softplus(a + dt_bias)
        #   g_exp = exp(g)  (the existing kernel's exp(g))
        beta = tl.sigmoid(tl.load(b_ptr + head).to(tl.float32))
        a_dt = tl.load(a_ptr + head).to(tl.float32) + tl.load(dt_bias_ptr + head).to(tl.float32)
        softplus = tl.log(1.0 + tl.exp(a_dt))
        g_value = tl.load(neg_exp_A_log_ptr + head).to(tl.float32) * softplus
        g_exp = tl.exp(g_value)

        s_offsets = head * BLOCK_K * HEAD_V + offs_k[:, None] * HEAD_V + offs_v[None, :]
        s_prev = tl.load(s_prev_ptr + s_offsets).to(tl.float32)
        s_decay = s_prev * g_exp
        kv_mem = tl.sum(s_decay * k[:, None], axis=0)
        delta = (v - kv_mem) * beta
        s_new = s_decay + k[:, None] * delta[None, :]
        out = tl.sum(s_new * q[:, None], axis=0)
        tl.store(s_new_ptr + s_offsets, s_new)
        tl.store(out_ptr + head * HEAD_V + offs_v, out)


def recurrent_gated_delta_fused_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token recurrent update with l2norm/scale/layout fused in Triton.

    Args use the engine decode layout:
      q/k:   [B=1, T=1, 32, 128]
      v:     [B=1, T=1, 32, 128]
      g/beta:[B=1, T=1, 32]
      state: [B=1, 32, 128, 128] FP32
    """
    _require_triton()
    if q.shape != (1, 1, NUM_V_HEADS, HEAD_K_DIM):
        raise ValueError(f"q shape must be [1,1,{NUM_V_HEADS},{HEAD_K_DIM}], got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != (1, 1, NUM_V_HEADS, HEAD_V_DIM):
        raise ValueError(f"unexpected k/v shape: k={tuple(k.shape)} v={tuple(v.shape)}")
    if g.shape != (1, 1, NUM_V_HEADS) or beta.shape != (1, 1, NUM_V_HEADS):
        raise ValueError(f"unexpected g/beta shape: g={tuple(g.shape)} beta={tuple(beta.shape)}")
    if s_prev.shape != (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM):
        raise ValueError(f"state shape must be [1,{NUM_V_HEADS},{HEAD_K_DIM},{HEAD_V_DIM}], got {tuple(s_prev.shape)}")
    initial_dtype = q.dtype
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=q.device, dtype=torch.float32)
    inplace_state = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"
    s_new = s_prev if inplace_state else torch.empty_like(s_prev)
    _recurrent_fused_prepare_kernel[(NUM_V_HEADS, 4)](
        q.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM),
        k.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM),
        v.contiguous().reshape(NUM_V_HEADS, HEAD_V_DIM),
        g.contiguous().reshape(NUM_V_HEADS),
        beta.contiguous().reshape(NUM_V_HEADS),
        s_prev.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM).to(initial_dtype), s_new


def recurrent_gated_delta_fused_prepare_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA-aware recurrent update that avoids q/k repeat_interleave.

    q/k use `[B=1,T=1,16,128]`; v/g/beta/state still use 32 value heads.
    Each pair of value heads reads the same q/k head via `head // 2`.
    """
    _require_triton()
    if q.shape != (1, 1, NUM_K_HEADS, HEAD_K_DIM):
        raise ValueError(f"q shape must be [1,1,{NUM_K_HEADS},{HEAD_K_DIM}], got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != (1, 1, NUM_V_HEADS, HEAD_V_DIM):
        raise ValueError(f"unexpected k/v shape: k={tuple(k.shape)} v={tuple(v.shape)}")
    if g.shape != (1, 1, NUM_V_HEADS) or beta.shape != (1, 1, NUM_V_HEADS):
        raise ValueError(f"unexpected g/beta shape: g={tuple(g.shape)} beta={tuple(beta.shape)}")
    if s_prev.shape != (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM):
        raise ValueError(f"state shape must be [1,{NUM_V_HEADS},{HEAD_K_DIM},{HEAD_V_DIM}], got {tuple(s_prev.shape)}")
    initial_dtype = q.dtype
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=q.device, dtype=torch.float32)
    inplace_state = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"
    s_new = s_prev if inplace_state else torch.empty_like(s_prev)
    _recurrent_fused_prepare_gqa_kernel[(NUM_V_HEADS, 4)](
        q.contiguous().reshape(NUM_K_HEADS, HEAD_K_DIM),
        k.contiguous().reshape(NUM_K_HEADS, HEAD_K_DIM),
        v.contiguous().reshape(NUM_V_HEADS, HEAD_V_DIM),
        g.contiguous().reshape(NUM_V_HEADS),
        beta.contiguous().reshape(NUM_V_HEADS),
        s_prev.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        V_PER_K=2,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM).to(initial_dtype), s_new


def recurrent_gated_delta_fused_prepare_from_outconv_gqa(
    out_conv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA recurrent update that reads q/k/v directly from conv output.

    This preserves the existing recurrent math while avoiding q/k/v split,
    reshape, and contiguous materialization in Python. `out_conv` uses the
    linear-attention decode layout `[1,1,8192]`:

      q [0:2048], k [2048:4096], v [4096:8192].
    """
    _require_triton()
    conv_dim = NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM
    if out_conv.shape != (1, 1, conv_dim):
        raise ValueError(f"out_conv shape must be [1,1,{conv_dim}], got {tuple(out_conv.shape)}")
    if g.shape != (1, 1, NUM_V_HEADS) or beta.shape != (1, 1, NUM_V_HEADS):
        raise ValueError(f"unexpected g/beta shape: g={tuple(g.shape)} beta={tuple(beta.shape)}")
    if s_prev.shape != (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM):
        raise ValueError(f"state shape must be [1,{NUM_V_HEADS},{HEAD_K_DIM},{HEAD_V_DIM}], got {tuple(s_prev.shape)}")
    initial_dtype = out_conv.dtype
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=out_conv.device, dtype=torch.float32)
    inplace_state = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"
    s_new = s_prev if inplace_state else torch.empty_like(s_prev)
    _recurrent_fused_prepare_from_outconv_gqa_kernel[(NUM_V_HEADS, 4)](
        out_conv.contiguous().reshape(conv_dim),
        g.contiguous().reshape(NUM_V_HEADS),
        beta.contiguous().reshape(NUM_V_HEADS),
        s_prev.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        NUM_K=NUM_K_HEADS,
        NUM_V=NUM_V_HEADS,
        V_PER_K=2,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM).to(initial_dtype), s_new


def recurrent_gated_delta_fused_prepare_from_outconv_ab_gqa(
    out_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    neg_exp_A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    s_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA recurrent update from out_conv while computing beta/g in Triton.

    This is a larger-boundary research candidate. It is expected to be more
    exactness-sensitive than `from_outconv_gqa` because sigmoid/softplus move
    from PyTorch to Triton.
    """
    _require_triton()
    conv_dim = NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM
    if out_conv.shape != (1, 1, conv_dim):
        raise ValueError(f"out_conv shape must be [1,1,{conv_dim}], got {tuple(out_conv.shape)}")
    if a.shape != (1, 1, NUM_V_HEADS) or b.shape != (1, 1, NUM_V_HEADS):
        raise ValueError(f"unexpected a/b shape: a={tuple(a.shape)} b={tuple(b.shape)}")
    if neg_exp_A_log.shape != (NUM_V_HEADS,) or dt_bias.shape != (NUM_V_HEADS,):
        raise ValueError(f"unexpected A/dt shape: A={tuple(neg_exp_A_log.shape)} dt={tuple(dt_bias.shape)}")
    if s_prev.shape != (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM):
        raise ValueError(f"state shape must be [1,{NUM_V_HEADS},{HEAD_K_DIM},{HEAD_V_DIM}], got {tuple(s_prev.shape)}")
    initial_dtype = out_conv.dtype
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=out_conv.device, dtype=torch.float32)
    inplace_state = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"
    s_new = s_prev if inplace_state else torch.empty_like(s_prev)
    _recurrent_fused_prepare_from_outconv_ab_gqa_kernel[(NUM_V_HEADS, 4)](
        out_conv.contiguous().reshape(conv_dim),
        a.contiguous().reshape(NUM_V_HEADS),
        b.contiguous().reshape(NUM_V_HEADS),
        neg_exp_A_log.contiguous().reshape(NUM_V_HEADS),
        dt_bias.contiguous().reshape(NUM_V_HEADS),
        s_prev.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        NUM_K=NUM_K_HEADS,
        NUM_V=NUM_V_HEADS,
        V_PER_K=2,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM).to(initial_dtype), s_new


def recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta(
    out_conv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    neg_exp_A_log: torch.Tensor,
    s_prev: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GQA recurrent update from out_conv with g/beta folded INTO the kernel.

    Stage-3 launch-fold: identical in result to
    ``recurrent_gated_delta_fused_prepare_from_outconv_gqa`` except beta/g are
    computed per-head in float32 INSIDE Triton from the raw projection outputs
    (``a`` = in_proj_a, ``b`` = in_proj_b) plus the per-head ``dt_bias`` /
    ``neg_exp_A_log``, instead of being pre-computed in PyTorch:

        beta = b.sigmoid()
        g    = neg_exp_A_log * softplus(a + dt_bias)

    This removes ~4 tiny elementwise launches per linear-attn layer (sigmoid,
    add, softplus, mul) — the recurrent kernel already does ``exp(g)``. The
    recurrent math (qk-l2norm / decay / kv_mem / delta / state-store / out) is
    byte-for-byte identical to the precomputed-g/beta kernel; only the source of
    g/beta changes, so token-exactness vs the existing path is the target.

    ``a``/``b`` are pre-sigmoid / pre-softplus ``[1, 1, NUM_V_HEADS]`` activations.
    ``dt_bias``/``neg_exp_A_log`` are per-head ``[NUM_V_HEADS]`` constants.
    """
    _require_triton()
    conv_dim = NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM
    if out_conv.shape != (1, 1, conv_dim):
        raise ValueError(f"out_conv shape must be [1,1,{conv_dim}], got {tuple(out_conv.shape)}")
    if a.shape != (1, 1, NUM_V_HEADS) or b.shape != (1, 1, NUM_V_HEADS):
        raise ValueError(f"unexpected a/b shape: a={tuple(a.shape)} b={tuple(b.shape)}")
    if neg_exp_A_log.shape != (NUM_V_HEADS,) or dt_bias.shape != (NUM_V_HEADS,):
        raise ValueError(f"unexpected A/dt shape: A={tuple(neg_exp_A_log.shape)} dt={tuple(dt_bias.shape)}")
    if s_prev.shape != (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM):
        raise ValueError(f"state shape must be [1,{NUM_V_HEADS},{HEAD_K_DIM},{HEAD_V_DIM}], got {tuple(s_prev.shape)}")
    initial_dtype = out_conv.dtype
    out = torch.empty((NUM_V_HEADS, HEAD_V_DIM), device=out_conv.device, dtype=torch.float32)
    inplace_state = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "0") == "1"
    s_new = s_prev if inplace_state else torch.empty_like(s_prev)
    _recurrent_fused_prepare_from_outconv_gqa_gbeta_kernel[(NUM_V_HEADS, 4)](
        out_conv.contiguous().reshape(conv_dim),
        a.contiguous().reshape(NUM_V_HEADS),
        b.contiguous().reshape(NUM_V_HEADS),
        dt_bias.contiguous().reshape(NUM_V_HEADS),
        neg_exp_A_log.contiguous().reshape(NUM_V_HEADS),
        s_prev.contiguous().reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        out,
        s_new.reshape(NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
        BLOCK_K=HEAD_K_DIM,
        BLOCK_V=32,
        HEAD_V=HEAD_V_DIM,
        NUM_K=NUM_K_HEADS,
        NUM_V=NUM_V_HEADS,
        V_PER_K=2,
        num_warps=4,
    )
    return out.reshape(1, 1, NUM_V_HEADS, HEAD_V_DIM).to(initial_dtype), s_new
