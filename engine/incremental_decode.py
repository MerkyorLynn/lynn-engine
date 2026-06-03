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

from engine.mtp_profile import section as profile_section
from engine.nvfp4_runtime import PackedNVFP4FusedLinear, PackedNVFP4Linear
from engine.qwen36_linear_attn_block import (
    HIDDEN_SIZE, NUM_K_HEADS, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, CONV_KERNEL,
    KEY_DIM, VALUE_DIM, V_PER_K, RMS_EPS,
    chunk_gated_delta_rule_torch,
    rms_norm_gated,
    l2norm,
)

try:
    from triton_kernels.gated_delta import (
        recurrent_gated_delta_fused_prepare,
        recurrent_gated_delta_fused_prepare_gqa,
        recurrent_gated_delta_fused_prepare_from_outconv_gqa,
    )
except Exception:  # pragma: no cover - optional acceleration path.
    recurrent_gated_delta_fused_prepare = None
    recurrent_gated_delta_fused_prepare_gqa = None
    recurrent_gated_delta_fused_prepare_from_outconv_gqa = None

try:
    # Stage-3 g/beta-fold variant (LYNN_LINEAR_ATTN_FUSE_GBETA). Imported
    # separately so its absence never disables the precomputed-g/beta path above.
    from triton_kernels.gated_delta import (
        recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta,
    )
except Exception:  # pragma: no cover - optional acceleration path.
    recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta = None

try:
    from triton_kernels.rmsnorm_gated import rms_norm_gated_triton
except Exception:  # pragma: no cover - optional acceleration path.
    rms_norm_gated_triton = None

try:
    from triton_kernels.linear_conv import linear_conv1d_update_triton
except Exception:  # pragma: no cover - optional acceleration path.
    linear_conv1d_update_triton = None

try:
    from triton_kernels.qk_norm_rope import qk_norm_rope_pair_triton, qk_norm_rope_triton
except Exception:  # pragma: no cover - optional acceleration path.
    qk_norm_rope_pair_triton = None
    qk_norm_rope_triton = None

try:
    from triton_kernels.rowwise_linear import rowwise_linear as rowwise_linear_triton
except Exception:  # pragma: no cover - optional acceleration path.
    rowwise_linear_triton = None

try:
    from triton_kernels.rowwise_attention import rowwise_prefix_attention as rowwise_prefix_attention_triton
except Exception:  # pragma: no cover - optional acceleration path.
    rowwise_prefix_attention_triton = None


_ROPE_TABLE_CACHE: dict[tuple[str, str, int, float, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _linear(x: torch.Tensor, weight) -> torch.Tensor:
    """Linear dispatch with optional packed NVFP4 decode-path support."""
    if isinstance(weight, (PackedNVFP4FusedLinear, PackedNVFP4Linear)):
        if weight.default_backend == "native_fast_2d":
            flat = x.reshape(-1, x.shape[-1])
            if flat.shape[0] != 1:
                raise NotImplementedError("native_fast_2d currently supports one token")
            out = weight.forward_native_fast_2d(flat)
            return out.to(x.dtype).reshape(*x.shape[:-1], weight.out_features)
        return weight(x)
    return F.linear(x, weight)


def _full_attn_o_proj(attn_out: torch.Tensor, weight) -> torch.Tensor:
    """Full-attention o_proj dispatch.

    ``LYNN_FULL_ATTN_O_PROJ_BACKEND=rowwise_triton`` is an experimental K2
    verifier path: T=1 and T=2 both use the same independent-row Triton
    accumulation contract. It is not the default because it does not match
    PyTorch/cuBLAS T=1 bit-for-bit.
    """
    if os.environ.get("LYNN_FULL_ATTN_O_PROJ_BACKEND", "") == "rowwise_triton":
        if rowwise_linear_triton is None:
            raise RuntimeError("LYNN_FULL_ATTN_O_PROJ_BACKEND=rowwise_triton requested but kernel is unavailable")
        if not torch.is_tensor(weight):
            raise TypeError("rowwise_triton o_proj backend requires a dense tensor weight")
        if attn_out.ndim != 3 or attn_out.shape[0] != 1 or attn_out.shape[1] not in (1, 2):
            return _linear(attn_out, weight)
        with profile_section("full_attn.o_proj.rowwise_triton"):
            out = rowwise_linear_triton(attn_out.reshape(attn_out.shape[1], attn_out.shape[2]), weight)
        return out.to(attn_out.dtype).reshape(1, attn_out.shape[1], weight.shape[0])
    with profile_section("full_attn.o_proj.default"):
        return _linear(attn_out, weight)


def _full_attn_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, attn_mask=None, enable_gqa: bool) -> torch.Tensor:
    """Full-attention dispatch for decode verifier experiments.

    ``LYNN_FULL_ATTN_ATTENTION_BACKEND=rowwise_triton`` uses the experimental
    prefix-attention kernel for T=1/T=2. The K2 kernel has the prefix-causal
    policy baked in: row 0 sees ``N-1`` keys and row 1 sees ``N`` keys.
    """
    backend = os.environ.get("LYNN_FULL_ATTN_ATTENTION_BACKEND", "")
    if backend == "rowwise_triton":
        if rowwise_prefix_attention_triton is None:
            raise RuntimeError("LYNN_FULL_ATTN_ATTENTION_BACKEND=rowwise_triton requested but kernel is unavailable")
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2] not in (1, 2):
            raise ValueError(f"rowwise_triton attention expects q=[1,H,1|2,D], got {tuple(q.shape)}")
        if attn_mask is not None and q.shape[2] != 2:
            raise ValueError("rowwise_triton attention only supports the implicit K2 prefix mask")
        with profile_section("full_attn.attention.rowwise_triton"):
            return rowwise_prefix_attention_triton(q, k, v)
    if backend not in {"", "sdpa"}:
        raise ValueError(f"Unknown LYNN_FULL_ATTN_ATTENTION_BACKEND: {backend}")
    with profile_section("full_attn.attention.sdpa"):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=enable_gqa,
        )


def _decode_weight(w: dict, key: str):
    """Return an optional packed decode alias for a linear weight.

    Prefill still uses the BF16/dequantized tensor under `key`; decode may opt
    into `key + ".packed"` so we can validate packed NVFP4 kernels end-to-end
    without destabilizing the multi-token prefill path.
    """
    packed_key = key + ".packed"
    if os.environ.get("LYNN_PACKED_DECODE", "0") == "1":
        return w[packed_key] if packed_key in w else w[key]
    if key.startswith("linear_attn.") and os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1":
        return w[packed_key] if packed_key in w else w[key]
    if key.startswith("self_attn.") and os.environ.get("LYNN_PACKED_DECODE_FULL_ATTN", "0") == "1":
        return w[packed_key] if packed_key in w else w[key]
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


def _linear_conv_update_decode(
    mixed_new: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    backend = os.environ.get("LYNN_LINEAR_ATTN_CONV_BACKEND", "torch")
    if backend in {"triton", "triton_inplace", "triton_torch_silu", "triton_torch_silu_inplace"}:
        if linear_conv1d_update_triton is None:
            raise RuntimeError("LYNN_LINEAR_ATTN_CONV_BACKEND requested but Triton kernel is unavailable")
        return linear_conv1d_update_triton(
            mixed_new,
            conv_state,
            conv_weight,
            inplace=backend.endswith("_inplace"),
            torch_silu=backend.startswith("triton_torch_silu"),
        )
    if backend != "torch":
        raise ValueError(f"unknown LYNN_LINEAR_ATTN_CONV_BACKEND={backend!r}")
    conv_input = torch.cat([conv_state, mixed_new], dim=-1)
    out_conv = F.conv1d(conv_input, conv_weight, bias=None, padding=0, groups=mixed_new.shape[1])
    out_conv = F.silu(out_conv).transpose(1, 2)
    new_conv_state = conv_input[:, :, 1:].contiguous()
    return out_conv, new_conv_state


def _qk_norm_rope_decode(x: torch.Tensor, weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    backend = os.environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch")
    if backend == "triton" and x.shape[0] == 1 and x.shape[2] == 1:
        if qk_norm_rope_triton is None:
            raise RuntimeError("LYNN_QK_NORM_ROPE_BACKEND=triton requested but kernel is unavailable")
        return qk_norm_rope_triton(x, weight, cos, sin, rotary_dim)
    if backend not in {"torch", "triton", "triton_pair"}:
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
        if q.shape[0] == 1 and k.shape[0] == 1 and q.shape[2] == 1 and k.shape[2] == 1:
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


def _build_rope_cos_sin_cached(positions: torch.Tensor, rotary_dim: int, theta: float, device, dtype):
    """Optional full-attention decode RoPE table cache.

    Rebuilding `arange -> inv_freq -> cos/sin` for every full-attention layer is
    launch-heavy during one-token decode. The cache is opt-in so gates can prove
    bitwise/greedy parity before it becomes a serving default.
    """
    max_seq = int(os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ", "65536"))
    half = rotary_dim // 2
    key = (str(torch.device(device)), str(dtype), int(rotary_dim), float(theta), max_seq)
    cached = _ROPE_TABLE_CACHE.get(key)
    if cached is None:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
        )
        seq = torch.arange(max_seq, device=device, dtype=torch.float32)
        freqs = seq[:, None] * inv_freq[None, :]
        cached = (freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous())
        _ROPE_TABLE_CACHE[key] = cached
    cos_table, sin_table = cached
    flat = positions.reshape(-1).to(device=device, dtype=torch.long)
    cos = cos_table.index_select(0, flat).reshape(*positions.shape, half).unsqueeze(1)
    sin = sin_table.index_select(0, flat).reshape(*positions.shape, half).unsqueeze(1)
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
    rope_builder = (
        _build_rope_cos_sin_cached
        if os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE", "0") == "1"
        else _build_rope_cos_sin
    )
    cos, sin = rope_builder(position_ids, rotary_dim, rope_theta, h.device, h.dtype)
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
    with profile_section("full_attn_t1.qkv"):
        if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1" and "self_attn._qkv_proj.weight" in w:
            q_out = int(w["self_attn.q_proj.weight"].shape[0])
            k_out = int(w["self_attn.k_proj.weight"].shape[0])
            v_out = int(w["self_attn.v_proj.weight"].shape[0])
            qkv = _linear(h_new, w["self_attn._qkv_proj.weight"])
            q_full, k_new, v_new = qkv.split((q_out, k_out, v_out), dim=-1)
        else:
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
    rope_builder = (
        _build_rope_cos_sin_cached
        if os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE", "0") == "1"
        else _build_rope_cos_sin
    )
    # LYNN_FULL_ATTN_FUSED=1: collapse q/k norm + RoPE + the K/V cache writes
    # into ONE Triton launch (3 launches → 1). bf16 caches only; the gate/o_proj
    # epilogue is fused separately below. write_pos == pos_tensor for every
    # current caller (T=1 decode and the t1_loop K2/block verifiers), so the
    # written cache row matches both the fixed-shape and variable-slice paths.
    fused = (
        os.environ.get("LYNN_FULL_ATTN_FUSED", "0") == "1"
        and q.is_cuda
        and K_cache_full.dtype == torch.bfloat16
        and V_cache_full.dtype == torch.bfloat16
    )
    with profile_section("full_attn_t1.rope"):
        cos, sin = rope_builder(pos_tensor, rotary_dim, rope_theta, h_new.device, h_new.dtype)
        if fused:
            from triton_kernels.full_attn_fused import qk_norm_rope_cache_write_triton
            q = qk_norm_rope_cache_write_triton(
                q,
                k_new,
                v_new,
                w["self_attn.q_norm.weight"],
                w["self_attn.k_norm.weight"],
                cos,
                sin,
                rotary_dim,
                K_cache_full,
                V_cache_full,
                pos_tensor.reshape(-1)[:1],
            )
        else:
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
    #    M3 fixed-shape (graph-replayable) path: write K/V via index_copy_ (the
    #    write position is read from pos_tensor, not baked as a Python-int slice)
    #    and attend over the FULL fixed-length cache with a position mask. No
    #    tensor shape then depends on cached_seq_len, so the whole decode can be
    #    captured once and replayed. Bit-exact with the variable-slice path:
    #    masked (future/stale) key positions get zero attention weight.
    fixed_shape = os.environ.get("LYNN_FULL_ATTN_FIXED_SHAPE", "0") == "1"
    attn_valid_mask = None
    with profile_section("full_attn_t1.cache_write"):
        if fused:
            pass  # post-RoPE K and raw V were written by the fused kernel above
        elif fixed_shape:
            pos_idx = pos_tensor.reshape(-1)[:1]
            K_cache_full.index_copy_(2, pos_idx, k_new)
            V_cache_full.index_copy_(2, pos_idx, v_new)
        else:
            K_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = k_new
            V_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = v_new

    # 5. Select cache window: fixed full + position mask, or variable slice.
    if fixed_shape:
        K_used = K_cache_full
        V_used = V_cache_full
        max_T = K_cache_full.shape[2]
        positions = torch.arange(max_T, device=h_new.device).view(1, 1, 1, max_T)
        attn_valid_mask = positions <= pos_tensor.reshape(1, 1, 1, 1)
    else:
        new_total = cached_seq_len + 1
        K_used = K_cache_full[:, :, :new_total, :]
        V_used = V_cache_full[:, :, :new_total, :]

    full_attn_backend = os.environ.get("LYNN_FULL_ATTN_DECODE_BACKEND", "sdpa")
    if fixed_shape:
        # Direct masked SDPA (bool mask: True = attend). Bypasses
        # _full_attn_attention's q.shape[2]==2 K2 special-casing.
        with profile_section("full_attn_t1.attention.sdpa_fixed"):
            attn_out = F.scaled_dot_product_attention(
                q, K_used, V_used,
                attn_mask=attn_valid_mask,
                is_causal=False,
                enable_gqa=(H_KV != H_Q),
            )
    elif full_attn_backend == "manual_gqa":
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
        attn_out = _full_attn_attention(q, K_used, V_used, enable_gqa=(H_KV != H_Q))
    else:
        raise ValueError(f"Unknown LYNN_FULL_ATTN_DECODE_BACKEND: {full_attn_backend}")

    # 8. attn_output_gate (+ transpose/contiguous reshape for o_proj)
    if fused and attn_out.is_cuda and attn_out.dtype == torch.bfloat16:
        # FUSED: sigmoid(gate) ⊙ attn_out AND the transpose→contiguous→view
        # reshape in one Triton launch (~5 launches → 1). Emits [1,1,H_Q*D].
        with profile_section("full_attn_t1.gate"):
            from triton_kernels.full_attn_fused import gate_sigmoid_fold_triton
            attn_flat = gate_sigmoid_fold_triton(attn_out, gate)
        with profile_section("full_attn_t1.o_proj_total"):
            return _full_attn_o_proj(attn_flat, _decode_weight(w, "self_attn.o_proj.weight"))

    with profile_section("full_attn_t1.gate"):
        attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)

    # 9. o_proj
    with profile_section("full_attn_t1.o_proj_total"):
        # T=1 decode: transpose(1,2) swaps a size-1 dim, so for a contiguous
        # attn_out the transposed view is already contiguous and the explicit
        # .contiguous() is a redundant copy. LYNN_DECODE_OPROJ_NOCOPY=1 uses
        # .reshape() (view when possible, copy only if actually non-contiguous)
        # — value-identical, never crashes. Default keeps the forced copy so
        # behavior is unchanged.
        attn_flat = attn_out.transpose(1, 2)
        if os.environ.get("LYNN_DECODE_OPROJ_NOCOPY", "0") == "1":
            attn_flat = attn_flat.reshape(B, 1, H_Q * head_dim)
        else:
            attn_flat = attn_flat.contiguous().view(B, 1, H_Q * head_dim)
        return _full_attn_o_proj(attn_flat, _decode_weight(w, "self_attn.o_proj.weight"))


def decode_full_attn_k2(
    h_new_k2,
    new_position_ids,
    w,
    cfg,
    K_cache_full,
    V_cache_full,
    cached_seq_len: int,
):
    """Decode K=2 new tokens at once using cached K/V.

    Used by the M5 batched K=1 speculative path: ``[pending, draft]`` arrive
    as a 2-position batch and we want the base model's logits for BOTH
    positions in a SINGLE forward (instead of two sequential T=1 decodes).
    The whole point is memory-bound layers (weight loads) cost the same
    for T=1 and T=2 — only the per-position compute differs, and at decode
    scale that's a small fraction of total cost.

    h_new_k2: ``[B, 2, HIDDEN]`` (post input_layernorm).
    new_position_ids: ``[B, 2]`` long, typically ``[cached_seq_len, cached_seq_len+1]``.

    Side effect: writes new K/V into
    ``K_cache_full[:, :, cached_seq_len:cached_seq_len+2, :]`` etc.
    On REJECT (draft mismatches base's argmax at position 0), the caller
    must rewind ``state.seq_len`` to ``cached_seq_len + 1`` so the stale
    K/V at position ``cached_seq_len + 1`` gets overwritten on the next
    write. KV is append-only — no clone/restore needed.

    Returns: ``attn_out [B, 2, HIDDEN]``.
    """
    B = h_new_k2.shape[0]
    assert h_new_k2.shape[1] == 2, f"decode_full_attn_k2 expects T=2, got {h_new_k2.shape}"
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])

    rope_builder = (
        _build_rope_cos_sin_cached
        if os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE", "0") == "1"
        else _build_rope_cos_sin
    )
    k2_backend = os.environ.get("LYNN_FULL_ATTN_K2_BACKEND", "t1_loop")
    probe_mode = os.environ.get("LYNN_FULL_ATTN_K2_PROBE", "")
    if k2_backend == "rowwise_bridge" and not probe_mode:
        probe_mode = "rowwise_qkv_rowwise_t1"
    if k2_backend == "rowwise_gate_bridge" and not probe_mode:
        probe_mode = "rowwise_qkv_rowwise_attn_batched_gate_rowwise_o"
    if k2_backend == "rowwise_kernel_bridge" and not probe_mode:
        probe_mode = "rowwise_qkv_rowwise_attn_kernel_batched_gate_rowwise_o"
    rowwise_qkv_probe_modes = {
        "rowwise_qkv",
        "rowwise_qkv_rowwise_t1",
        "rowwise_qkv_rowwise_attn_batched_o",
        "rowwise_qkv_batched_attn_rowwise_o",
        "rowwise_qkv_rowwise_attn_batched_gate_rowwise_o",
        "rowwise_qkv_rowwise_attn_rowwise_gate_batched_o",
        "rowwise_qkv_rowwise_attn_kernel_batched_gate_rowwise_o",
    }
    if probe_mode in rowwise_qkv_probe_modes:
        q_pieces = []
        gate_pieces = []
        k_pieces = []
        v_pieces = []
        for idx in range(2):
            h_i = h_new_k2[:, idx:idx + 1, :].contiguous()
            with profile_section("full_attn_k2.rowwise_qkv"):
                if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1" and "self_attn._qkv_proj.weight" in w:
                    q_out = int(w["self_attn.q_proj.weight"].shape[0])
                    k_out = int(w["self_attn.k_proj.weight"].shape[0])
                    v_out = int(w["self_attn.v_proj.weight"].shape[0])
                    qkv_i = _linear(h_i, w["self_attn._qkv_proj.weight"])
                    q_full_i, k_i, v_i = qkv_i.split((q_out, k_out, v_out), dim=-1)
                else:
                    q_full_i = _linear(h_i, _decode_weight(w, "self_attn.q_proj.weight"))
                    k_i = _linear(h_i, _decode_weight(w, "self_attn.k_proj.weight"))
                    v_i = _linear(h_i, _decode_weight(w, "self_attn.v_proj.weight"))
                q_full_view_i = q_full_i.view(B, 1, H_Q, head_dim * 2)
                q_i, gate_i = q_full_view_i.chunk(2, dim=-1)
                q_i = q_i.transpose(1, 2)
                gate_i = gate_i.transpose(1, 2)
                k_i = k_i.view(B, 1, H_KV, head_dim).transpose(1, 2)
                v_i = v_i.view(B, 1, H_KV, head_dim).transpose(1, 2)
            with profile_section("full_attn_k2.rowwise_rope"):
                cos_i, sin_i = rope_builder(
                    new_position_ids[:, idx:idx + 1].contiguous(),
                    rotary_dim,
                    rope_theta,
                    h_new_k2.device,
                    h_new_k2.dtype,
                )
                q_i, k_i = _qk_norm_rope_pair_decode(
                    q_i,
                    k_i,
                    w["self_attn.q_norm.weight"],
                    w["self_attn.k_norm.weight"],
                    cos_i,
                    sin_i,
                    rotary_dim,
                )
            q_pieces.append(q_i)
            gate_pieces.append(gate_i)
            k_pieces.append(k_i)
            v_pieces.append(v_i)
        with profile_section("full_attn_k2.rowwise_cat"):
            q = torch.cat(q_pieces, dim=2)
            gate = torch.cat(gate_pieces, dim=2)
            k_new = torch.cat(k_pieces, dim=2)
            v_new = torch.cat(v_pieces, dim=2)
    else:
        # 1. Q/K/V projection
        with profile_section("full_attn_k2.batched_qkv"):
            if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1" and "self_attn._qkv_proj.weight" in w:
                q_out = int(w["self_attn.q_proj.weight"].shape[0])
                k_out = int(w["self_attn.k_proj.weight"].shape[0])
                v_out = int(w["self_attn.v_proj.weight"].shape[0])
                qkv = _linear(h_new_k2, w["self_attn._qkv_proj.weight"])
                q_full, k_new, v_new = qkv.split((q_out, k_out, v_out), dim=-1)
            else:
                q_full = _linear(h_new_k2, _decode_weight(w, "self_attn.q_proj.weight"))
                k_new = _linear(h_new_k2, _decode_weight(w, "self_attn.k_proj.weight"))
                v_new = _linear(h_new_k2, _decode_weight(w, "self_attn.v_proj.weight"))

            q_full_view = q_full.view(B, 2, H_Q, head_dim * 2)
            q, gate = q_full_view.chunk(2, dim=-1)
            q = q.transpose(1, 2)                                  # [B, H_Q, 2, head_dim]
            gate = gate.transpose(1, 2)
            k_new = k_new.view(B, 2, H_KV, head_dim).transpose(1, 2)  # [B, H_KV, 2, head_dim]
            v_new = v_new.view(B, 2, H_KV, head_dim).transpose(1, 2)

        # 2+3. q/k norm + RoPE on both positions.
        with profile_section("full_attn_k2.batched_rope"):
            cos, sin = rope_builder(new_position_ids, rotary_dim, rope_theta, h_new_k2.device, h_new_k2.dtype)
            q, k_new = _qk_norm_rope_pair_decode(
                q, k_new, w["self_attn.q_norm.weight"], w["self_attn.k_norm.weight"], cos, sin, rotary_dim,
            )

    # 4. Write both new positions into cache.
    with profile_section("full_attn_k2.cache_write"):
        K_cache_full[:, :, cached_seq_len:cached_seq_len + 2, :] = k_new
        V_cache_full[:, :, cached_seq_len:cached_seq_len + 2, :] = v_new

    # 5. Slice usable cache.
    new_total = cached_seq_len + 2
    K_used = K_cache_full[:, :, :new_total, :]
    V_used = V_cache_full[:, :, :new_total, :]

    rowwise_attn_probe_modes = {
        "rowwise_t1",
        "rowwise_qkv_rowwise_t1",
        "rowwise_qkv_rowwise_attn_batched_o",
        "rowwise_qkv_rowwise_attn_batched_gate_rowwise_o",
        "rowwise_qkv_rowwise_attn_rowwise_gate_batched_o",
    }
    if probe_mode in rowwise_attn_probe_modes:
        attn_pieces = []
        out_pieces = []
        for idx in range(2):
            q_i = q[:, :, idx:idx + 1, :].contiguous()
            gate_i = gate[:, :, idx:idx + 1, :].contiguous()
            k_i = K_cache_full[:, :, :cached_seq_len + idx + 1, :]
            v_i = V_cache_full[:, :, :cached_seq_len + idx + 1, :]
            attn_i = F.scaled_dot_product_attention(
                q_i,
                k_i,
                v_i,
                is_causal=False,
                enable_gqa=(H_KV != H_Q),
            )
            attn_pieces.append(attn_i)
            if probe_mode in {"rowwise_t1", "rowwise_qkv_rowwise_t1"}:
                attn_i = attn_i * torch.sigmoid(gate_i.float()).to(attn_i.dtype)
                attn_i = attn_i.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
                out_pieces.append(_full_attn_o_proj(attn_i, _decode_weight(w, "self_attn.o_proj.weight")))
        if probe_mode == "rowwise_qkv_rowwise_attn_batched_o":
            attn_out = torch.cat(attn_pieces, dim=2)
            attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, 2, H_Q * head_dim)
            return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))
        if probe_mode == "rowwise_qkv_rowwise_attn_batched_gate_rowwise_o":
            attn_out = torch.cat(attn_pieces, dim=2)
            attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
            pieces = []
            for idx in range(2):
                attn_i = attn_out[:, :, idx:idx + 1, :].contiguous()
                attn_i = attn_i.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
                pieces.append(_full_attn_o_proj(attn_i, _decode_weight(w, "self_attn.o_proj.weight")))
            return torch.cat(pieces, dim=1)
        if probe_mode == "rowwise_qkv_rowwise_attn_rowwise_gate_batched_o":
            gated_pieces = []
            for idx, attn_i in enumerate(attn_pieces):
                gate_i = gate[:, :, idx:idx + 1, :].contiguous()
                gated_pieces.append(attn_i * torch.sigmoid(gate_i.float()).to(attn_i.dtype))
            attn_out = torch.cat(gated_pieces, dim=2)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, 2, H_Q * head_dim)
            return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))
        return torch.cat(out_pieces, dim=1)

    if probe_mode == "rowwise_qkv_rowwise_attn_kernel_batched_gate_rowwise_o":
        if rowwise_prefix_attention_triton is None:
            raise RuntimeError("rowwise_kernel_bridge requested but rowwise attention kernel is unavailable")
        attn_out = rowwise_prefix_attention_triton(q, K_used, V_used)
        with profile_section("full_attn_k2.gate_batched"):
            attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
        pieces = []
        for idx in range(2):
            with profile_section("full_attn_k2.rowwise_o_prep"):
                attn_i = attn_out[:, :, idx:idx + 1, :].contiguous()
                attn_i = attn_i.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
            pieces.append(_full_attn_o_proj(attn_i, _decode_weight(w, "self_attn.o_proj.weight")))
        with profile_section("full_attn_k2.rowwise_o_cat"):
            return torch.cat(pieces, dim=1)

    # 6. Attention with prefix-causal mask:
    #    Q[0] (at pos cached_seq_len)   sees K[0..cached_seq_len]   (NOT cached_seq_len+1)
    #    Q[1] (at pos cached_seq_len+1) sees K[0..cached_seq_len+1]
    # PyTorch SDPA causal mode requires q_len == k_len; here q_len=2, k_len>=2.
    # Build a [2, new_total] additive bool mask, broadcast through SDPA attn_mask.
    causal_mask = torch.zeros(2, new_total, dtype=torch.bool, device=h_new_k2.device)
    causal_mask[0, : cached_seq_len + 1] = True
    causal_mask[1, : cached_seq_len + 2] = True
    attn_mask = torch.zeros(2, new_total, dtype=h_new_k2.dtype, device=h_new_k2.device)
    attn_mask.masked_fill_(~causal_mask, float("-inf"))
    attn_mask = attn_mask.view(1, 1, 2, new_total)

    full_attn_backend = os.environ.get("LYNN_FULL_ATTN_DECODE_BACKEND", "sdpa")
    if full_attn_backend == "manual_gqa":
        group = H_Q // H_KV
        q_grouped = q.view(B, H_KV, group, 2, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped.float(), K_used.float()) * scale
        scores = scores + attn_mask.unsqueeze(2).float()  # broadcast over group
        probs = torch.softmax(scores, dim=-1).to(V_used.dtype)
        attn_out = torch.einsum("bhgqk,bhkd->bhgqd", probs, V_used)
        attn_out = attn_out.reshape(B, H_Q, 2, head_dim)
    elif full_attn_backend == "sdpa":
        # PyTorch SDPA: attn_mask + enable_gqa together is supported since 2.5.
        attn_out = _full_attn_attention(
            q, K_used, V_used,
            attn_mask=attn_mask,
            enable_gqa=(H_KV != H_Q),
        )
    else:
        raise ValueError(f"Unknown LYNN_FULL_ATTN_DECODE_BACKEND: {full_attn_backend}")

    if probe_mode == "rowwise_qkv_batched_attn_rowwise_o":
        pieces = []
        for idx in range(2):
            attn_i = attn_out[:, :, idx:idx + 1, :].contiguous()
            gate_i = gate[:, :, idx:idx + 1, :].contiguous()
            attn_i = attn_i * torch.sigmoid(gate_i.float()).to(attn_i.dtype)
            attn_i = attn_i.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
            pieces.append(_full_attn_o_proj(attn_i, _decode_weight(w, "self_attn.o_proj.weight")))
        return torch.cat(pieces, dim=1)

    # 7. attn_output_gate
    attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)

    # 8. o_proj
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, 2, H_Q * head_dim)
    return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))


def decode_full_attn_block(
    h_new_block,
    new_position_ids,
    w,
    cfg,
    K_cache_full,
    V_cache_full,
    cached_seq_len: int,
):
    """Decode an arbitrary block of new tokens against the prefix KV cache.

    This is the K=N generalization of ``decode_full_attn_k2``. It is intended
    for experimental MTP block verification, where the base model verifies
    ``[pending, draft_1, ..., draft_k]`` in one forward. Rejected suffix KV is
    invalidated by rewinding ``state.seq_len``; stale cache slots beyond
    ``seq_len`` are ignored and overwritten by later writes.
    """
    B, block_len, _ = h_new_block.shape
    if block_len == 1:
        return decode_full_attn(
            h_new_block,
            new_position_ids[:, 0:1].contiguous(),
            w,
            cfg,
            K_cache_full,
            V_cache_full,
            cached_seq_len=cached_seq_len,
        )
    if block_len == 2:
        return decode_full_attn_k2(
            h_new_block,
            new_position_ids,
            w,
            cfg,
            K_cache_full,
            V_cache_full,
            cached_seq_len=cached_seq_len,
        )

    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])

    if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1" and "self_attn._qkv_proj.weight" in w:
        q_out = int(w["self_attn.q_proj.weight"].shape[0])
        k_out = int(w["self_attn.k_proj.weight"].shape[0])
        v_out = int(w["self_attn.v_proj.weight"].shape[0])
        qkv = _linear(h_new_block, w["self_attn._qkv_proj.weight"])
        q_full, k_new, v_new = qkv.split((q_out, k_out, v_out), dim=-1)
    else:
        q_full = _linear(h_new_block, _decode_weight(w, "self_attn.q_proj.weight"))
        k_new = _linear(h_new_block, _decode_weight(w, "self_attn.k_proj.weight"))
        v_new = _linear(h_new_block, _decode_weight(w, "self_attn.v_proj.weight"))

    q_full_view = q_full.view(B, block_len, H_Q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)
    gate = gate.transpose(1, 2)
    k_new = k_new.view(B, block_len, H_KV, head_dim).transpose(1, 2)
    v_new = v_new.view(B, block_len, H_KV, head_dim).transpose(1, 2)

    rope_builder = (
        _build_rope_cos_sin_cached
        if os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE", "0") == "1"
        else _build_rope_cos_sin
    )
    cos, sin = rope_builder(new_position_ids, rotary_dim, rope_theta, h_new_block.device, h_new_block.dtype)
    q, k_new = _qk_norm_rope_pair_decode(
        q, k_new, w["self_attn.q_norm.weight"], w["self_attn.k_norm.weight"], cos, sin, rotary_dim,
    )

    K_cache_full[:, :, cached_seq_len:cached_seq_len + block_len, :] = k_new
    V_cache_full[:, :, cached_seq_len:cached_seq_len + block_len, :] = v_new

    new_total = cached_seq_len + block_len
    K_used = K_cache_full[:, :, :new_total, :]
    V_used = V_cache_full[:, :, :new_total, :]

    row = torch.arange(block_len, device=h_new_block.device).view(block_len, 1)
    col = torch.arange(new_total, device=h_new_block.device).view(1, new_total)
    causal_mask = col <= (cached_seq_len + row)
    attn_mask = torch.zeros(block_len, new_total, dtype=h_new_block.dtype, device=h_new_block.device)
    attn_mask.masked_fill_(~causal_mask, float("-inf"))
    attn_mask = attn_mask.view(1, 1, block_len, new_total)

    full_attn_backend = os.environ.get("LYNN_FULL_ATTN_DECODE_BACKEND", "sdpa")
    if full_attn_backend == "manual_gqa":
        group = H_Q // H_KV
        q_grouped = q.view(B, H_KV, group, block_len, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped.float(), K_used.float()) * scale
        scores = scores + attn_mask.unsqueeze(2).float()
        probs = torch.softmax(scores, dim=-1).to(V_used.dtype)
        attn_out = torch.einsum("bhgqk,bhkd->bhgqd", probs, V_used)
        attn_out = attn_out.reshape(B, H_Q, block_len, head_dim)
    elif full_attn_backend == "sdpa":
        attn_out = F.scaled_dot_product_attention(
            q, K_used, V_used,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=(H_KV != H_Q),
        )
    else:
        raise ValueError(f"Unknown LYNN_FULL_ATTN_DECODE_BACKEND: {full_attn_backend}")

    attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, block_len, H_Q * head_dim)
    return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))


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

    # 5. q, k repeat by V_PER_K. Prefill keeps the reference tensor layout; the
    # P10-D GQA shortcut is decode-only.
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


def decode_linear_attn_k2(
    h_new_k2,
    w,
    recurrent_state,
    conv_state,
    *,
    recurrent_backend: str = "torch",
):
    """Decode K=2 tokens through one linear-attention layer.

    Gated Delta-Net is a stateful SSM — each token's output depends on the
    recurrent state produced by the previous token. There is no way to run
    two steps in parallel within a single layer; the cost is just 2× a T=1
    decode for the linear-attn layer. Across the 30/40 hybrid SSM stack
    that adds ~16% to a K=2 forward, but the win comes from full-attn +
    MoE layers being memory-bound (K=2 ≈ free there).

    Returns: ``out [B, 2, HIDDEN]``, plus updated recurrent/conv states.
    """
    assert h_new_k2.shape[1] == 2, f"decode_linear_attn_k2 expects T=2, got {h_new_k2.shape}"
    out0, recurrent_state, conv_state = decode_linear_attn(
        h_new_k2[:, 0:1, :].contiguous(),
        w, recurrent_state, conv_state,
        recurrent_backend=recurrent_backend,
    )
    out1, recurrent_state, conv_state = decode_linear_attn(
        h_new_k2[:, 1:2, :].contiguous(),
        w, recurrent_state, conv_state,
        recurrent_backend=recurrent_backend,
    )
    out = torch.cat([out0, out1], dim=1)
    return out, recurrent_state, conv_state


def decode_linear_attn_block(
    h_new_block,
    w,
    recurrent_state,
    conv_state,
    *,
    recurrent_backend: str = "torch",
):
    """Decode an arbitrary block through one linear-attention layer.

    Gated Delta-Net state is inherently sequential, so this intentionally loops
    over positions while preserving the same state-update contract as T=1.
    The block-level speed win comes from full-attention and MoE layers, not
    from parallelizing the recurrent update.
    """
    outs = []
    for idx in range(int(h_new_block.shape[1])):
        out, recurrent_state, conv_state = decode_linear_attn(
            h_new_block[:, idx:idx + 1, :].contiguous(),
            w,
            recurrent_state,
            conv_state,
            recurrent_backend=recurrent_backend,
        )
        outs.append(out)
    return torch.cat(outs, dim=1), recurrent_state, conv_state


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

    # 2. Causal conv1d update: prepend conv_state to mixed_new, run conv, take last 1 output.
    # Triton opt-in can fuse cat + depthwise conv + silu + state shift.
    out_conv, new_conv_state = _linear_conv_update_decode(
        mixed_new,
        conv_state,
        W("linear_attn.conv1d.weight"),
    )

    # 3. z, beta, g (using h_new)
    z = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    dt_bias = W("linear_attn.dt_bias")
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -W("linear_attn.A_log").float().exp()

    # 4. q, k repeat by V_PER_K. P10-D can avoid this materialization for the
    # Triton recurrent path by reading q/k as grouped-query heads directly.
    use_gqa_recurrent = (
        recurrent_backend == "triton_fused_prepare"
        and V_PER_K > 1
        and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"
    )

    use_outconv_recurrent = (
        use_gqa_recurrent
        and os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV", "0") == "1"
    )

    # P-Stage3 launch-fold: on the outconv recurrent path, compute g/beta
    # per-head INSIDE the recurrent Triton kernel from raw a/b/dt_bias/
    # neg_exp_A_log instead of pre-computing them here — folds away ~4 tiny
    # elementwise launches/layer (sigmoid, add, softplus, mul). Math identical;
    # token-exactness is the target. Default OFF; falls back to the precomputed
    # g/beta path when the flag is unset or the fused kernel is unavailable.
    fuse_gbeta = (
        use_outconv_recurrent
        and os.environ.get("LYNN_LINEAR_ATTN_FUSE_GBETA", "0") == "1"
        and recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta is not None
    )

    # Pre-compute g/beta outside only when NOT folding them into the kernel.
    # (When folding, a/dt_bias stay raw and are cast to fp32 inside Triton, so we
    # also skip the per-token a.float()/dt_bias.float() casts here.)
    if not fuse_gbeta:
        beta = b.sigmoid()
        g = neg_exp_A_log * F.softplus(a.float() + dt_bias.float())

    if use_outconv_recurrent:
        if fuse_gbeta:
            core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare_from_outconv_gqa_gbeta(
                out_conv, a, b, dt_bias, neg_exp_A_log, recurrent_state
            )
        else:
            if recurrent_gated_delta_fused_prepare_from_outconv_gqa is None:
                raise RuntimeError("outconv recurrent requested but Triton kernel is unavailable")
            core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare_from_outconv_gqa(
                out_conv, g, beta, recurrent_state
            )
    else:
        q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)

    # 5. recurrent gated delta rule (single-step)
    if use_outconv_recurrent:
        pass
    elif recurrent_backend == "torch":
        core_attn_out, new_recurrent_state = _recurrent_gated_delta_rule(
            q, k, v, g, beta, recurrent_state
        )
    elif recurrent_backend == "triton_fused_prepare":
        if recurrent_gated_delta_fused_prepare is None:
            raise RuntimeError("triton_fused_prepare requested but triton kernel is unavailable")
        if use_gqa_recurrent:
            if recurrent_gated_delta_fused_prepare_gqa is None:
                raise RuntimeError("GQA recurrent requested but triton kernel is unavailable")
            core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare_gqa(
                q, k, v, g, beta, recurrent_state
            )
        else:
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
    "decode_full_attn_k2",
    "decode_full_attn_block",
    "prefill_linear_attn",
    "decode_linear_attn",
    "decode_linear_attn_k2",
    "decode_linear_attn_block",
]
