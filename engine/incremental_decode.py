"""
Lynn Engine · Phase 3.1 · Incremental decode primitives.

Refactored prefill / decode versions of:
  - _full_attn_forward  → prefill_full_attn / decode_full_attn (uses KV cache)
  - lynn_linear_attn    → prefill_linear_attn / decode_linear_attn (uses recurrent state)

Existing `engine/full_forward.py` remains unchanged for the brute-force greedy
path (Phase 2). This module is the new path used by `generate_incremental` (TBD).
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import PackedNVFP4Linear
from engine.qwen36_linear_attn_block import (
    HIDDEN_SIZE, NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_KERNEL,
    KEY_DIM, VALUE_DIM, V_PER_K, RMS_EPS,
    chunk_gated_delta_rule_torch,
    rms_norm_gated,
    l2norm,
)

try:
    from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare
except Exception:  # pragma: no cover - optional acceleration path.
    recurrent_gated_delta_fused_prepare = None

try:
    from triton_kernels.rmsnorm_gated import rms_norm_gated_triton
except Exception:  # pragma: no cover - optional acceleration path.
    rms_norm_gated_triton = None

try:
    from triton_kernels.qk_norm_rope import qk_norm_rope_pair_triton, qk_norm_rope_triton
except Exception:  # pragma: no cover - optional acceleration path.
    qk_norm_rope_pair_triton = None
    qk_norm_rope_triton = None


def _linear(x: torch.Tensor, weight) -> torch.Tensor:
    """Linear dispatch with optional packed NVFP4 decode-path support."""
    if isinstance(weight, PackedNVFP4Linear):
        if weight.default_backend == "native_fast_2d":
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] != 1:
                raise NotImplementedError("native_fast_2d currently supports one token")
            out = weight.forward_native_fast_2d(flat)
            return out.to(x.dtype).reshape(*x.shape[:-1], weight.out_features)
        return weight(x)
    return F.linear(x, weight)


def _decode_weight(w: dict, key: str):
    """Return an optional packed decode alias for a linear weight.

    Prefill still uses the BF16/dequantized tensor under `key`; decode may opt
    into `key + ".packed"` so we can validate packed NVFP4 kernels end-to-end
    without destabilizing the multi-token prefill path.
    """
    if os.environ.get("LYNN_PACKED_DECODE", "0") == "1":
        return w.get(key + ".packed", w[key])
    if key.startswith("linear_attn.") and os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1":
        return w.get(key + ".packed", w[key])
    if key.startswith("self_attn.") and os.environ.get("LYNN_PACKED_DECODE_FULL_ATTN", "0") == "1":
        return w.get(key + ".packed", w[key])
    return w[key]


def _rms_norm_gated_decode(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    backend = os.environ.get("LYNN_RMSNORM_GATED_BACKEND", "torch")
    if backend == "triton":
        if rms_norm_gated_triton is None:
            raise RuntimeError("LYNN_RMSNORM_GATED_BACKEND=triton requested but kernel is unavailable")
        return rms_norm_gated_triton(x, weight, gate, eps=RMS_EPS)
    if backend != "torch":
        raise ValueError(f"unknown LYNN_RMSNORM_GATED_BACKEND={backend!r}")
    return rms_norm_gated(x, weight, gate, eps=RMS_EPS)


def _qk_norm_rope_decode(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    backend = os.environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch")
    if backend == "triton":
        if qk_norm_rope_triton is None:
            raise RuntimeError("LYNN_QK_NORM_ROPE_BACKEND=triton requested but kernel is unavailable")
        return qk_norm_rope_triton(x, weight, cos, sin, rotary_dim)
    if backend != "torch":
        raise ValueError(f"unknown LYNN_QK_NORM_ROPE_BACKEND={backend!r}")
    return _apply_partial_rope(_rms_norm(x, weight), cos, sin, rotary_dim)


def _qk_norm_rope_pair_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    backend = os.environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch")
    if backend == "triton_pair":
        if qk_norm_rope_pair_triton is None:
            raise RuntimeError("LYNN_QK_NORM_ROPE_BACKEND=triton_pair requested but kernel is unavailable")
        return qk_norm_rope_pair_triton(q, k, q_weight, k_weight, cos, sin, rotary_dim)
    return (
        _qk_norm_rope_decode(q, q_weight, cos, sin, rotary_dim),
        _qk_norm_rope_decode(k, k_weight, cos, sin, rotary_dim),
    )


# Standard RMSNorm (with the (1.0 + weight) trick — Qwen3 specific).
def _rms_norm(x, weight, eps=1e-6):
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + eps)
    return (x_n * (1.0 + weight.float())).to(in_dtype)


# ============================================================================
# RoPE primitives — supports both prefill (range of positions) and decode
# (single position).
# ============================================================================

def _build_rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float,
                       device, dtype):
    """positions: [B, T] long. Returns cos, sin: [B, 1, T, rotary_dim/2].

    F2.1 cat-free: returns half-width cos/sin (instead of doubled via cat).
    GPT-NeoX layout: cos[:half] and cos[half:] would be identical in old impl,
    so storing only half is byte-exact equivalent and saves the cat op.
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.float()[:, :, None] * inv_freq[None, None, :]
    cos = freqs.cos()[:, None, :, :].to(dtype)  # [B, 1, T, half]
    sin = freqs.sin()[:, None, :, :].to(dtype)
    return cos, sin


def _apply_partial_rope(x, cos, sin, rotary_dim):
    """GPT-NeoX style partial RoPE, cat-free implementation.

    x:    [B, H, M, head_dim]
    cos:  [B, 1, M, rotary_dim/2]  (half-width per F2.1)
    sin:  [B, 1, M, rotary_dim/2]
    Returns: rotated x [B, H, M, head_dim]

    Equivalent to:
      x_rotated = x_rot * cos_full + rotate_half(x_rot) * sin_full
      out = cat([x_rotated, x_pass], dim=-1)
    where rotate_half flips first/second halves of rotary_dim. Since cos_full and
    sin_full are doubled (cos[:half] == cos[half:]), we use half-width cos/sin
    directly:
      out[:half]              = x_first * cos - x_second * sin
      out[half:rotary_dim]    = x_second * cos + x_first * sin
      out[rotary_dim:]        = x[rotary_dim:]  (pass-through)
    Eliminates 3 cat ops (rotate_half, x_rotated+x_pass, freq doubling).
    """
    half = rotary_dim // 2
    out = x.clone()  # copy entire x (pass-through region included)
    x_first = x[..., :half]
    x_second = x[..., half:rotary_dim]
    out[..., :half] = x_first * cos - x_second * sin
    out[..., half:rotary_dim] = x_second * cos + x_first * sin
    return out


# ============================================================================
# full_attention prefill / decode
# ============================================================================

def prefill_full_attn(h, position_ids, w, cfg):
    """Prefill: process T tokens, return attn_out + K, V to write into cache.

    h: [B, T, HIDDEN]  (post input_layernorm)
    position_ids: [B, T] long
    w: dict of layer weights
    cfg: dict with rope_theta, partial_rotary_factor, num_attention_heads, etc.

    Returns:
        attn_out: [B, T, HIDDEN]
        K_to_cache: [B, NUM_KV_HEADS, T, HEAD_DIM]   (the K that should go into cache)
        V_to_cache: [B, NUM_KV_HEADS, T, HEAD_DIM]
    """
    B, T, _ = h.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])

    # 1. Q/K/V projection (Q is 2× for attn_output_gate)
    q_full = F.linear(h, w["self_attn.q_proj.weight"])
    k = F.linear(h, w["self_attn.k_proj.weight"])
    v = F.linear(h, w["self_attn.v_proj.weight"])

    # Per-head reshape THEN chunk on q_full
    q_full_view = q_full.view(B, T, H_Q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)                             # [B, H_Q, T, head_dim]
    gate = gate.transpose(1, 2)
    k = k.view(B, T, H_KV, head_dim).transpose(1, 2)  # [B, H_KV, T, head_dim]
    v = v.view(B, T, H_KV, head_dim).transpose(1, 2)

    # 2. q_norm, k_norm
    q = _rms_norm(q, w["self_attn.q_norm.weight"])
    k = _rms_norm(k, w["self_attn.k_norm.weight"])

    # 3. RoPE on q and k (both will be cached as post-RoPE for K)
    cos, sin = _build_rope_cos_sin(position_ids, rotary_dim, rope_theta,
                                   h.device, h.dtype)
    q = _apply_partial_rope(q, cos, sin, rotary_dim)
    k = _apply_partial_rope(k, cos, sin, rotary_dim)

    # 4. K, V to be cached are the post-RoPE K and post-projection V.
    # NOTE: cache K is post-RoPE so subsequent decode just attends with current Q.
    K_to_cache = k.contiguous()
    V_to_cache = v.contiguous()

    # 5. GQA repeat for attention
    if H_KV != H_Q:
        k_attn = k.repeat_interleave(H_Q // H_KV, dim=1)
        v_attn = v.repeat_interleave(H_Q // H_KV, dim=1)
    else:
        k_attn = k
        v_attn = v

    # 6. Causal attention
    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=True)

    # 7. attn_output_gate
    attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)

    # 8. o_proj
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H_Q * head_dim)
    out = F.linear(attn_out, w["self_attn.o_proj.weight"])
    return out, K_to_cache, V_to_cache


def decode_full_attn(h_new, new_position_id, w, cfg, K_cache_full, V_cache_full,
                    cached_seq_len: int):
    """Decode 1 new token using cached K/V.

    h_new: [B, 1, HIDDEN]  (post input_layernorm)
    new_position_id: scalar int (position of the new token, 0-indexed)
    K_cache_full, V_cache_full: [B, NUM_KV_HEADS, max_T, HEAD_DIM] (pre-allocated full size)
    cached_seq_len: how many positions are already populated in cache (= new_position_id for fresh decode)

    Side effect: writes new K/V into K_cache_full[:, :, cached_seq_len:cached_seq_len+1, :] etc.

    Returns: attn_out [B, 1, HIDDEN]
    """
    B, _, _ = h_new.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])

    # 1. Q/K/V projection on the single new token
    q_full = _linear(h_new, _decode_weight(w, "self_attn.q_proj.weight"))
    k_new = _linear(h_new, _decode_weight(w, "self_attn.k_proj.weight"))
    v_new = _linear(h_new, _decode_weight(w, "self_attn.v_proj.weight"))

    q_full_view = q_full.view(B, 1, H_Q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)
    gate = gate.transpose(1, 2)
    k_new = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    v_new = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)

    # 2+3. q/k norm + RoPE on the new position.
    # CUDA graph capture cannot include fresh torch.tensor allocations, so
    # benchmark/serving paths may pass a preallocated [[position]] tensor.
    if torch.is_tensor(new_position_id):
        pos_tensor = new_position_id
    else:
        pos_tensor = torch.tensor([[new_position_id]], device=h_new.device, dtype=torch.long)
    cos, sin = _build_rope_cos_sin(pos_tensor, rotary_dim, rope_theta,
                                   h_new.device, h_new.dtype)
    q, k_new = _qk_norm_rope_pair_decode(
        q,
        k_new,
        w["self_attn.q_norm.weight"],
        w["self_attn.k_norm.weight"],
        cos,
        sin,
        rotary_dim,
    )

    # 4. Append to cache at position cached_seq_len
    K_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = k_new
    V_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = v_new

    # 5. Slice cache up to and including new token
    new_total = cached_seq_len + 1
    K_used = K_cache_full[:, :, :new_total, :]
    V_used = V_cache_full[:, :, :new_total, :]

    full_attn_backend = os.environ.get("LYNN_FULL_ATTN_DECODE_BACKEND", "sdpa")
    if full_attn_backend == "manual_gqa":
        # Decode uses a single query token. Avoid SDPA launch/dispatch overhead
        # by doing grouped-query attention explicitly without materializing
        # repeated KV heads. This is opt-in until parity + latency are proven.
        group = H_Q // H_KV
        q_grouped = q.view(B, H_KV, group, 1, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped.float(), K_used.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(V_used.dtype)
        attn_out = torch.einsum("bhgqk,bhkd->bhgqd", probs, V_used)
        attn_out = attn_out.reshape(B, H_Q, 1, head_dim)
    elif full_attn_backend == "sdpa":
        # SDPA with enable_gqa=True (PyTorch 2.5+) — internal broadcast,
        # no memory expansion. Math equivalent to explicit repeat_interleave+SDPA.
        # Replaces 2× repeat_interleave (8x mem copy on H_Q/H_KV=8) with view-only.
        attn_out = F.scaled_dot_product_attention(q, K_used, V_used, is_causal=False, enable_gqa=(H_KV != H_Q))
    else:
        raise ValueError(f"Unknown LYNN_FULL_ATTN_DECODE_BACKEND: {full_attn_backend}")

    # 8. attn_output_gate
    attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)

    # 9. o_proj
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
    return _linear(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))


# ============================================================================
# linear_attention prefill / decode
# ============================================================================

def _recurrent_gated_delta_rule(q, k, v, g, beta, S_prev):
    """Single-token (T=1) recurrent path. Port of HF torch_recurrent_gated_delta_rule.

    q, k: [B, T=1, num_v_heads, head_k_dim]
    v: [B, T=1, num_v_heads, head_v_dim]
    g, beta: [B, T=1, num_v_heads]
    S_prev: [B, num_v_heads, head_k_dim, head_v_dim]  FP32

    Returns:
        out: [B, T=1, num_v_heads, head_v_dim]
        S_new: [B, num_v_heads, head_k_dim, head_v_dim] FP32
    """
    initial_dtype = q.dtype

    # l2norm Q, K (inside-kernel,跟 chunk path 同款)
    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)

    # → FP32 for stable accumulation, shape [B, num_v_heads, T=1, head_dim]
    q = q.transpose(1, 2).contiguous().to(torch.float32)
    k = k.transpose(1, 2).contiguous().to(torch.float32)
    v = v.transpose(1, 2).contiguous().to(torch.float32)
    g = g.transpose(1, 2).contiguous().to(torch.float32)
    beta = beta.transpose(1, 2).contiguous().to(torch.float32)

    # Squeeze the T=1 dim
    q = q.squeeze(2)      # [B, num_v_heads, head_k_dim]
    k = k.squeeze(2)
    v = v.squeeze(2)      # [B, num_v_heads, head_v_dim]
    g = g.squeeze(2)      # [B, num_v_heads]
    beta = beta.squeeze(2)

    # Scale q
    scale = 1.0 / math.sqrt(q.shape[-1])
    q = q * scale

    # Recurrent step (HF torch_recurrent_gated_delta_rule line 340):
    #   state = state * exp(g_t)
    #   kv_mem = sum(state * k_t.unsqueeze(-1), dim=-2)
    #   delta = (v_t - kv_mem) * beta_t
    #   state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    #   out_t = sum(state * q_t.unsqueeze(-1), dim=-2)
    g_exp = g.exp().unsqueeze(-1).unsqueeze(-1)              # [B, H, 1, 1]
    S = S_prev * g_exp                                       # decay
    kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)               # [B, H, head_v_dim]
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S * q.unsqueeze(-1)).sum(dim=-2)                  # [B, H, head_v_dim]

    # Restore [B, T=1, num_v_heads, head_v_dim]
    out = out.unsqueeze(1)
    return out.to(initial_dtype), S


def prefill_linear_attn(h, w, chunk_size: int = 64):
    """Prefill: chunk-recurrence over T tokens, return out + final state (recurrent + conv).

    h: [B, T, HIDDEN]  (post input_layernorm)
    Returns:
        out: [B, T, HIDDEN]
        new_recurrent_state: [B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM] FP32
        new_conv_state: [B, CONV_DIM, CONV_KERNEL-1] same dtype as h
    """
    B, T, _ = h.shape

    def W(k):
        return w[k]

    # 1. QKV proj
    mixed = F.linear(h, W("linear_attn.in_proj_qkv.weight"))     # [B, T, conv_dim]
    mixed = mixed.transpose(1, 2)                                # [B, conv_dim, T]

    # 2. Causal conv1d (left-pad kernel-1)
    conv_w = W("linear_attn.conv1d.weight")
    pad = CONV_KERNEL - 1
    mixed_padded = F.pad(mixed, (pad, 0))                        # [B, conv_dim, T+3]
    mixed_conv = F.conv1d(mixed_padded, conv_w, bias=None, padding=0,
                          groups=mixed.shape[1])
    mixed_conv = F.silu(mixed_conv)
    # Save last (kernel-1) tokens of the *input* to conv as new_conv_state.
    # The input to conv was mixed_padded; the last (kernel-1) positions of
    # mixed (un-padded) become the conv_state for next decode.
    new_conv_state = mixed[:, :, max(0, T - (CONV_KERNEL - 1)):].contiguous()
    if new_conv_state.shape[-1] < CONV_KERNEL - 1:
        # Pad on left so it's exactly kernel-1 wide
        pad_amt = (CONV_KERNEL - 1) - new_conv_state.shape[-1]
        new_conv_state = F.pad(new_conv_state, (pad_amt, 0))

    mixed_conv = mixed_conv.transpose(1, 2)                      # [B, T, conv_dim]

    # 3. Split q, k, v
    q, k, v = torch.split(mixed_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)

    # 4. z, beta, g
    z = F.linear(h, W("linear_attn.in_proj_z.weight")).reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)
    b = F.linear(h, W("linear_attn.in_proj_b.weight"))
    beta = b.sigmoid()
    a = F.linear(h, W("linear_attn.in_proj_a.weight"))
    dt_bias = W("linear_attn.dt_bias")
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -W("linear_attn.A_log").float().exp()
    g = neg_exp_A_log * F.softplus(a.float() + dt_bias.float())

    # 5. q, k repeat by V_PER_K
    if V_PER_K > 1:
        q = q.repeat_interleave(V_PER_K, dim=2)
        k = k.repeat_interleave(V_PER_K, dim=2)

    # 6. chunk_gated_delta_rule WITH output_final_state=True
    core_attn_out, last_state = _chunk_gated_delta_with_state(
        q, k, v, g, beta, chunk_size=chunk_size, use_qk_l2norm=True,
        initial_state=None, output_final_state=True,
    )

    # 7. RMSNormGated
    norm_w = W("linear_attn.norm.weight")
    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z.reshape(-1, HEAD_V_DIM)
    flat_y = _rms_norm_gated_decode(flat_x, norm_w, flat_z)
    core_attn_out = flat_y.reshape(B, T, NUM_V_HEADS * HEAD_V_DIM)

    # 8. out_proj
    out = F.linear(core_attn_out, W("linear_attn.out_proj.weight"))

    return out, last_state, new_conv_state


def decode_linear_attn(h_new, w, recurrent_state, conv_state, *, recurrent_backend: str = "torch"):
    """Decode 1 new token using cached recurrent_state + conv_state.

    h_new: [B, 1, HIDDEN] (post input_layernorm)
    recurrent_state: [B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM] FP32
    conv_state: [B, CONV_DIM, CONV_KERNEL-1]

    Returns:
        out: [B, 1, HIDDEN]
        new_recurrent_state: [B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM] FP32
        new_conv_state: [B, CONV_DIM, CONV_KERNEL-1]
    """
    B = h_new.shape[0]

    def W(k):
        return _decode_weight(w, k)

    # 1. Input projections on h_new.
    #
    # P6-O opt-in path: fuse qkv/z/b/a input projections into one GEMM.
    # Decode is dominated by many tiny launches across 30 linear-attn layers;
    # one larger matmul is usually better than four independent launches.
    use_packed_linear = (
        os.environ.get("LYNN_PACKED_DECODE", "0") == "1"
        or os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1"
    )
    if not use_packed_linear and "linear_attn._in_proj_qkv_z_b_a.weight" in w:
        proj_all = _linear(h_new, W("linear_attn._in_proj_qkv_z_b_a.weight"))
        mixed_new, z, b, a = torch.split(
            proj_all,
            [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
            dim=-1,
        )
    else:
        mixed_new = _linear(h_new, W("linear_attn.in_proj_qkv.weight"))    # [B, 1, conv_dim]
        z = _linear(h_new, W("linear_attn.in_proj_z.weight")).reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        b = _linear(h_new, W("linear_attn.in_proj_b.weight"))
        a = _linear(h_new, W("linear_attn.in_proj_a.weight"))
    mixed_new = mixed_new.transpose(1, 2)                              # [B, conv_dim, 1]

    # 2. Causal conv1d update: prepend conv_state to mixed_new, run conv, take last 1 output
    conv_input = torch.cat([conv_state, mixed_new], dim=-1)            # [B, conv_dim, kernel]
    conv_w = W("linear_attn.conv1d.weight")
    out_conv = F.conv1d(conv_input, conv_w, bias=None, padding=0,
                        groups=mixed_new.shape[1])                     # [B, conv_dim, 1]
    out_conv = F.silu(out_conv)
    out_conv = out_conv.transpose(1, 2)                                # [B, 1, conv_dim]

    # New conv_state = last (kernel-1) tokens of input (mixed_new is 1 new, conv_state is past)
    new_conv_state = conv_input[:, :, 1:].contiguous()                  # drop oldest

    # 3. Split q/k/v
    q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)

    # 4. z, beta, g (using h_new)
    z = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b.sigmoid()
    dt_bias = W("linear_attn.dt_bias")
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -W("linear_attn.A_log").float().exp()
    g = neg_exp_A_log * F.softplus(a.float() + dt_bias.float())

    # 5. q, k repeat by V_PER_K
    if V_PER_K > 1:
        q = q.repeat_interleave(V_PER_K, dim=2)
        k = k.repeat_interleave(V_PER_K, dim=2)

    # 6. recurrent gated delta rule (single-step)
    if recurrent_backend == "torch":
        core_attn_out, new_recurrent_state = _recurrent_gated_delta_rule(
            q, k, v, g, beta, recurrent_state
        )
    elif recurrent_backend == "triton_fused_prepare":
        if recurrent_gated_delta_fused_prepare is None:
            raise RuntimeError("triton_fused_prepare requested but triton kernel is unavailable")
        core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare(
            q, k, v, g, beta, recurrent_state
        )
    else:
        raise ValueError(f"unknown recurrent_backend={recurrent_backend!r}")

    # 7. RMSNormGated
    norm_w = W("linear_attn.norm.weight")
    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z.reshape(-1, HEAD_V_DIM)
    flat_y = _rms_norm_gated_decode(flat_x, norm_w, flat_z)
    core_attn_out = flat_y.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)

    # 8. out_proj
    out = _linear(core_attn_out, W("linear_attn.out_proj.weight"))
    return out, new_recurrent_state, new_conv_state


# ============================================================================
# Helper: chunk_gated_delta_rule with output_final_state=True
# (the version in qwen36_linear_attn_block.py discards last_state — we need it here)
# ============================================================================

def _chunk_gated_delta_with_state(query, key, value, g, beta, chunk_size=64,
                                   use_qk_l2norm=True, initial_state=None,
                                   output_final_state=False):
    """chunk_gated_delta_rule but ALWAYS returning the last_recurrent_state.

    Same math as engine.qwen36_linear_attn_block.chunk_gated_delta_rule_torch,
    but here we never set last_recurrent_state to None.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    B, H, T, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))
    T_pad = T + pad

    scale = 1.0 / (k_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)

    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    if initial_state is None:
        last_recurrent_state = torch.zeros(
            B, H, k_dim, v_dim, dtype=value.dtype, device=value.device,
        )
    else:
        last_recurrent_state = initial_state.to(value)

    core_attn_out = torch.zeros_like(value)

    n_chunk = T_pad // chunk_size
    for i in range(n_chunk):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a_within = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        a_inter = (q_i * g_cum[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = a_inter + a_within @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_cum[:, :, i, -1, None, None].exp()
            + (k_i * (g_cum[:, :, i, -1, None] - g_cum[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    core_attn_out = core_attn_out.reshape(B, H, -1, v_dim)
    core_attn_out = core_attn_out[:, :, :T]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


__all__ = [
    "prefill_full_attn",
    "decode_full_attn",
    "prefill_linear_attn",
    "decode_linear_attn",
]
