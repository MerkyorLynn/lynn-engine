"""
Lynn Engine · Qwen 3.6 35B-A3B linear_attention (Gated Delta Net) block.

Mirrors HF transformers' `Qwen3_5MoeGatedDeltaNet` math, hooked to Lynn's
loader / RMSNorm kernel. Used at 30 of the 40 transformer layers (indices
0,1,2,4,5,6,8,9,10,12,13,14,...).

Reference: transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
  - `torch_chunk_gated_delta_rule` (line ~235): the prefill math we port
  - `torch_recurrent_gated_delta_rule` (line ~315): single-token decode math
  - `Qwen3_5MoeRMSNormGated` (line ~176): norm with silu-gated value path
  - `Qwen3_5MoeGatedDeltaNet.forward` (line ~425): the wrapper this module mirrors

This is a torch-only port for correctness validation against HF — no Triton
kernels yet. The chunk math is small per-chunk (chunk_size=64) so torch ops
are not a bottleneck. Triton-fuse later in Phase 3 for throughput.

Block dimensions (Qwen 3.6 config, fixed):
  hidden_size               = 2048
  linear_num_key_heads      = 16     k_dim per head 128 → 2048 total
  linear_num_value_heads    = 32     v_dim per head 128 → 4096 total
  linear_conv_kernel_dim    = 4
  conv_dim                  = 2*key_dim + value_dim = 8192
  v_per_k                   = 32 / 16 = 2  (q,k repeat-interleaved by 2)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# -- Config (hard-coded for Qwen 3.6 35B-A3B; matches text_config) -------------
HIDDEN_SIZE = 2048
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_KERNEL = 4
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM       # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM     # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM        # 8192
V_PER_K = NUM_V_HEADS // NUM_K_HEADS      # 2
RMS_EPS = 1e-6


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """RMSNormGated, bit-equivalent to HF's `Qwen3_5MoeRMSNormGated.forward`.

    The intermediate-precision dance is significant — HF rounds the normed
    value BACK to in_dtype before multiplying by weight, then promotes to
    FP32 for the silu-gate, then truncates to in_dtype. Doing it all in FP32
    diverges from HF by ~10x at the output magnitude.
    """
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_norm = x_f * torch.rsqrt(var + eps)
    # weight (BF16) * x_norm.to(BF16) → BF16, then BF16 * silu(gate FP32) → FP32.
    x_normed_in_dtype = x_norm.to(in_dtype)
    out_low = weight * x_normed_in_dtype  # BF16
    out_fp = out_low * F.silu(gate.float())  # FP32 (BF16 * FP32 promotes)
    return out_fp.to(in_dtype)


def chunk_gated_delta_rule_torch(query, key, value, g, beta,
                                 chunk_size: int = 64,
                                 use_qk_l2norm: bool = True):
    """Direct port of HF's `torch_chunk_gated_delta_rule`.

    Inputs (after the q/k repeat_interleave applied by caller):
      query, key:  [B, T, num_v_heads, head_k_dim]
      value:       [B, T, num_v_heads, head_v_dim]
      g, beta:     [B, T, num_v_heads]

    Returns:
      core_attn_out [B, T, num_v_heads, head_v_dim],
      last_recurrent_state [B, num_v_heads, head_k_dim, head_v_dim]
    """
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # → [B, num_heads, T, dim]
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
    # reshape to chunks: [B, H, n_chunk, chunk_size, dim]
    query, key, value, k_beta, v_beta = [
        x.reshape(B, H, -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(B, H, -1, chunk_size)

    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # within-chunk decay
    g_cum = g.cumsum(dim=-1)
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)

    # online inverse delta accumulation (HF's per-chunk loop)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    last_recurrent_state = torch.zeros(
        B, H, k_dim, v_dim, dtype=value.dtype, device=value.device
    )
    core_attn_out = torch.zeros_like(value)
    # NOTE: HF defines a causal mask `torch.triu(..., diagonal=1)` here but
    # never applies it — `decay_mask` from line 277 has `.tril()` baked in
    # which already enforces causality. We intentionally mirror the dead-code
    # quirk: the within-chunk attention is masked solely by decay_mask.

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


def lynn_linear_attn_forward(hidden_states: torch.Tensor, weights: dict,
                             chunk_size: int = 64) -> torch.Tensor:
    """Forward pass for one linear_attention layer.

    Args:
        hidden_states: [B, T, hidden_size] (BF16)
        weights: dict of dequantized BF16 tensors with keys (loader output):
            linear_attn.in_proj_qkv.weight   [conv_dim, hidden] = [8192, 2048]
            linear_attn.conv1d.weight        [conv_dim, 1, 4]
            linear_attn.in_proj_z.weight     [value_dim, hidden]
            linear_attn.in_proj_b.weight     [num_v_heads, hidden]
            linear_attn.in_proj_a.weight     [num_v_heads, hidden]
            linear_attn.A_log                [num_v_heads]
            linear_attn.dt_bias              [num_v_heads]
            linear_attn.norm.weight          [head_v_dim]
            linear_attn.out_proj.weight      [hidden, value_dim]

    Returns:
        output: [B, T, hidden_size]
    """
    B, T, _ = hidden_states.shape

    def W(k):
        return weights[k]

    # 1. QKV projection
    mixed = F.linear(hidden_states, W("linear_attn.in_proj_qkv.weight"))
    # [B, T, conv_dim] → transpose to [B, conv_dim, T] for conv1d
    mixed = mixed.transpose(1, 2)

    # 2. depthwise causal conv1d (left-pad kernel-1)
    conv_w = W("linear_attn.conv1d.weight")  # [conv_dim, 1, kernel]
    pad = CONV_KERNEL - 1
    mixed_padded = F.pad(mixed, (pad, 0))
    mixed = F.conv1d(mixed_padded, conv_w, bias=None, padding=0,
                     groups=CONV_DIM)
    # F.silu activation
    mixed = F.silu(mixed)
    mixed = mixed.transpose(1, 2)  # [B, T, conv_dim]

    # 3. split q, k, v
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(B, T, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)

    # 4. z (value-path gate input)
    z = F.linear(hidden_states, W("linear_attn.in_proj_z.weight"))
    z = z.reshape(B, T, NUM_V_HEADS, HEAD_V_DIM)

    # 5. beta (delta gate)
    b = F.linear(hidden_states, W("linear_attn.in_proj_b.weight"))
    beta = b.sigmoid()

    # 6. g (decay)
    a = F.linear(hidden_states, W("linear_attn.in_proj_a.weight"))
    A_log = W("linear_attn.A_log")
    dt_bias = W("linear_attn.dt_bias")
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())

    # 7. repeat q, k by V_PER_K so all share num_v_heads
    if V_PER_K > 1:
        q = q.repeat_interleave(V_PER_K, dim=2)
        k = k.repeat_interleave(V_PER_K, dim=2)

    # 8. chunk gated delta rule (the math)
    core_attn_out, _ = chunk_gated_delta_rule_torch(
        q, k, v, g, beta,
        chunk_size=chunk_size,
        use_qk_l2norm=True,
    )
    # core_attn_out: [B, T, NUM_V_HEADS, HEAD_V_DIM]

    # 9. RMSNormGated with z gate (per head_v_dim, not per token)
    # HF flattens to [B*T*num_v_heads, head_v_dim] and broadcasts the per-head_dim weight.
    norm_w = W("linear_attn.norm.weight")  # [head_v_dim]
    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z.reshape(-1, HEAD_V_DIM)
    flat_y = rms_norm_gated(flat_x, norm_w, flat_z, eps=RMS_EPS)
    core_attn_out = flat_y.reshape(B, T, NUM_V_HEADS * HEAD_V_DIM)

    # 10. output projection
    output = F.linear(core_attn_out, W("linear_attn.out_proj.weight"))
    return output


__all__ = ["lynn_linear_attn_forward", "chunk_gated_delta_rule_torch"]
