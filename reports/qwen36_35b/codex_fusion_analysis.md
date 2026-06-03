Reading additional input from stdin...
OpenAI Codex v0.136.0-alpha.2
--------
workdir: /Users/lynn/Downloads/Lynn/lynn-engine
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 019e8b9c-4bf9-7c13-aa2a-8468031f396b
--------
user
The Lynn NVFP4 Qwen3.6-35B-A3B decode is kernel-LAUNCH-OVERHEAD bound: ~140 tiny CUDA/Triton kernel launches per token, 40 tok/s vs llama.cpp Q4_K_M 70 tok/s on the SAME GPU (sm_121, no FP4 MMA). We measured it is NOT memory-bound (~37% of 240 GB/s) and NOT bytes (cutting weight bytes gave ~0). The whole gap is per-op launch/dispatch overhead.

Read the decode hot path in this repo: engine/incremental_decode.py (decode_full_attn, decode_linear_attn, _decode_weight, _linear), engine/full_forward.py (_decode_layer / _ffn_forward), engine/moe_packed_nvfp4.py (moe_forward_decode_packed_nvfp4 + _moe_forward_decode_packed_nvfp4_fixed_triton), triton_kernels/nvfp4_moe.py.

Produce a PRIORITIZED markdown list of the top 4-6 kernel-FUSION opportunities to cut per-token launch COUNT, focused on per-layer and per-expert launches. For each: (1) which launches merge, (2) concrete Triton fusion approach, (3) est. launches saved per token, (4) token-exactness risk + mitigation, (5) effort S/M/L. Cite exact functions/kernels. Do NOT write kernels yet — analysis + plan only. End with your single highest-ROI pick and why.
codex
I’ll inspect the decode path and kernel call sites first, then turn that into a launch-count reduction plan. I’ll keep this to analysis only and won’t edit files.
exec
/bin/zsh -lc "sed -n '1,260p' engine/incremental_decode.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '1,260p' engine/full_forward.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""
Lynn Engine · 40-layer full forward (memory-bounded, layer-by-layer load).

The Lynn-engine end-to-end forward without any HF dependency. Architecture:

    input_ids → embed_tokens → h0
    for layer i in 0..39:
        residual = h
        h = input_layernorm(h)
        if layer_types[i] == 'linear_attention':
            h = lynn_linear_attn_forward(h, layer_i_weights)
        else:  # 'full_attention'
            h = lynn_full_attn_forward(h, layer_i_weights)  # (P1.1 path)
        h = residual + h
        residual = h
        h = post_attention_layernorm(h)
        h = MoE_forward(h, layer_i_weights)  # 256 experts top-8 + shared
        h = residual + h
        # free layer_i_weights
    h = final_norm(h)
    logits = h @ lm_head.T
    return logits

Memory profile:
  embeddings + lm_head:          1.0 GB BF16   (kept resident)
  per-layer weights, peak:       1.7 GB BF16   (loaded then freed)
  hidden state activation:       few MB        (B=1, T<=256)
  Total GPU peak:               ~3 GB

Doesn't disturb running vLLM (which uses 60 GB at mem-fraction 0.5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.mtp_profile import section as profile_section


_DENSE_FP4XFP8_EXT = None


def _quantize_to_fp8_e4m3_per16(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single decode activation to E4M3 with per-16 FP32 scale."""
    x_flat = x.view(-1).float()
    if x_flat.numel() % 16 != 0:
        raise ValueError(f"FP4xFP8 activation length must be divisible by 16, got {x_flat.numel()}")
    grouped = x_flat.view(x_flat.numel() // 16, 16)
    scale = (grouped.abs().amax(dim=-1) / 448.0).clamp_min(1e-8)
    fp8 = (grouped / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8).view(-1).contiguous(), scale.contiguous()


def _dense_fp4xfp8_extension():
    global _DENSE_FP4XFP8_EXT
    if _DENSE_FP4XFP8_EXT is None:
        from engine.native_cuda import load_lynn_native_extension

        _DENSE_FP4XFP8_EXT = load_lynn_native_extension(verbose=False)
        if not hasattr(_DENSE_FP4XFP8_EXT, "dense_fp4xfp8_mma_scaled_probe"):
            raise RuntimeError("dense_fp4xfp8_mma_scaled_probe missing from native extension")
    return _DENSE_FP4XFP8_EXT


def _dense_fp4xfp8_project(x: torch.Tensor, w: dict, proj: str) -> torch.Tensor:
    """Run one dense FFN projection through the R6000 native E4M3xE2M1 MMA bridge."""
    prefix = f"mlp._fp4xfp8.{proj}."
    packed = w[prefix + "weight_packed"]
    scale = w[prefix + "weight_scale"].float()
    global_scale = w[prefix + "weight_global_scale"].float().view(-1)
    n, k_half = packed.shape
    k = k_half * 2
    act_fp8, act_scale = _quantize_to_fp8_e4m3_per16(x.reshape(-1)[:k])
    out = _dense_fp4xfp8_extension().dense_fp4xfp8_mma_scaled_probe(
        act_fp8,
        act_scale,
        packed.contiguous(),
        scale.contiguous(),
        global_scale.contiguous(),
        1,
        int(n),
        int(k),
    )
    return out.reshape(*x.shape[:-1], int(n))


def _has_dense_fp4xfp8_sidecar(w: dict) -> bool:
    required = (
        "mlp._fp4xfp8.gate_proj.weight_packed",
        "mlp._fp4xfp8.gate_proj.weight_scale",
        "mlp._fp4xfp8.gate_proj.weight_global_scale",
        "mlp._fp4xfp8.up_proj.weight_packed",
        "mlp._fp4xfp8.up_proj.weight_scale",
        "mlp._fp4xfp8.up_proj.weight_global_scale",
        "mlp._fp4xfp8.down_proj.weight_packed",
        "mlp._fp4xfp8.down_proj.weight_scale",
        "mlp._fp4xfp8.down_proj.weight_global_scale",
    )
    return all(k in w for k in required)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3_5MoeRMSNorm — note the `(1.0 + weight)` factor, not plain `weight`.

    From HF transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:806 ::
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # the +1 offset
        return output.type_as(x)

    See https://github.com/huggingface/transformers/pull/29402 — Qwen-family
    diverges from Llama-style RMSNorm (which is `weight * x` only).
    """
    in_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + eps)
    return (x_n * (1.0 + weight.float())).to(in_dtype)


def _full_attn_forward(h: torch.Tensor, position_ids: torch.Tensor,
                       w: dict, cfg: dict) -> torch.Tensor:
    """Full-attention forward (Qwen 3.6 specifics: GQA, attn_output_gate,
    q_norm/k_norm, partial-rotary GPT-NeoX-style RoPE with theta=1e7).

    Note on RoPE: Qwen 3.6 uses MROPE (multi-modal: T/H/W position grids)
    with `partial_rotary_factor=0.25` (only first 64 of 256 head dims rotate)
    and `rope_theta=1e7`. For text-only input, T=H=W positions, so MROPE
    collapses to standard GPT-NeoX RoPE on the first 64 dims. The remaining
    192 dims pass through unrotated.
    """
    B, M, D = h.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    partial = cfg["partial_rotary_factor"]
    rotary_dim = int(head_dim * partial)

    q_full = F.linear(h, w["self_attn.q_proj.weight"])
    k = F.linear(h, w["self_attn.k_proj.weight"])
    v = F.linear(h, w["self_attn.v_proj.weight"])

    # Critical: q_proj output is [B, M, H_Q*2*head_dim]. HF first reshapes to
    # [B, M, H_Q, 2*head_dim] (per-head 2x slot) then chunks along last dim
    # into [q, gate]. Doing chunk(2, dim=-1) on the flat representation
    # incorrectly mixes head0_gate into "q" and head_last_q into "gate".
    q_full_view = q_full.view(B, M, H_Q, head_dim * 2)
    q, attn_output_gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)                              # [B, H_Q, M, head_dim]
    attn_output_gate = attn_output_gate.transpose(1, 2)
    k = k.view(B, M, H_KV, head_dim).transpose(1, 2)
    v = v.view(B, M, H_KV, head_dim).transpose(1, 2)

    # q_norm and k_norm (Qwen3 trick)
    q = _rms_norm(q, w["self_attn.q_norm.weight"])
    k = _rms_norm(k, w["self_attn.k_norm.weight"])

    # RoPE — GPT-NeoX split-halves style on first `rotary_dim` channels
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, device=h.device, dtype=torch.float32) / rotary_dim)
    )  # [rotary_dim // 2]
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]  # [B, M, rotary_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, M, rotary_dim]
    cos = emb.cos()[:, None, :, :]  # [B, 1, M, rotary_dim] (broadcast over H)
    sin = emb.sin()[:, None, :, :]

    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def apply_partial_rope(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        c, s = cos.to(x.dtype), sin.to(x.dtype)
        x_rotated = (x_rot * c) + (rotate_half(x_rot) * s)
        return torch.cat([x_rotated, x_pass], dim=-1)

    q = apply_partial_rope(q)
    k = apply_partial_rope(k)

    # GQA: repeat k, v
    if H_KV != H_Q:
        k = k.repeat_interleave(H_Q // H_KV, dim=1)
        v = v.repeat_interleave(H_Q // H_KV, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attn_out = attn_out * torch.sigmoid(attn_output_gate.float()).to(attn_out.dtype)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, M, H_Q * head_dim)
    return F.linear(attn_out, w["self_attn.o_proj.weight"])


def _moe_forward(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """MoE forward: 256 experts, top-K=8 routing, shared expert with sigmoid gate.

    Phase 2 Spark FP8 path: keyed on ``mlp.experts.gate_up_proj.weight_fp8``
    presence. When present, per-expert FP8 fused gate/up + SwiGLU is
    dispatched via ``triton_kernels/spark_fp8_moe_expert_fused.py`` (the
    codebuddy MoE-expert variant kernel, 1.82-2.10x over BF16 at MoE
    intermediate shapes). The down_proj follows via ``torch._scaled_mm``
    per expert. Opt-out via ``LYNN_DISABLE_W4A8_FP8_PATH=1``.
    """
    B, M, D = h.shape
    E = int(cfg.get("num_experts", 0) or 0)
    if E <= 0:
        raise RuntimeError("MoE forward called for a dense FFN layer")
    K = cfg["num_experts_per_tok"]

    h_flat = h.view(B * M, D)
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)

    moe_out = torch.zeros_like(h_flat)

    # Phase 2 FP8 MoE path (auto-selected from weight dict).
    fp8_disabled = os.environ.get("LYNN_DISABLE_W4A8_FP8_PATH", "0").lower() in {"1", "true", "yes"}
    fp8_moe = (
        not fp8_disabled
        and "mlp.experts.gate_up_proj.weight_fp8" in w
        and "mlp.experts.down_proj.weight_fp8" in w
    )
    if fp8_moe:
        from triton_kernels.spark_fp8_moe_expert_fused import (
            fp8_moe_expert_gate_up_silu_fused,
        )
        # Stacked weight layout: gate_up_proj [E, 2*intermediate, hidden] →
        # split halves into gate [E, intermediate, hidden] + up [E, ...].
        w_gate_up_fp8 = w["mlp.experts.gate_up_proj.weight_fp8"]   # [E, 2I, K]
        w_gate_up_scale = w["mlp.experts.gate_up_proj.weight_fp8_scale"]  # [E, 2I]
        intermediate = w_gate_up_fp8.shape[1] // 2
        w_gate_fp8 = w_gate_up_fp8[:, :intermediate, :].contiguous()
        w_up_fp8 = w_gate_up_fp8[:, intermediate:, :].contiguous()
        w_gate_scale = w_gate_up_scale[:, :intermediate].contiguous().to(torch.float32)
        w_up_scale = w_gate_up_scale[:, intermediate:].contiguous().to(torch.float32)
        w_down_fp8 = w["mlp.experts.down_proj.weight_fp8"]          # [E, K, intermediate]
        w_down_scale = w["mlp.experts.down_proj.weight_fp8_scale"]  # [E, K]
        for e in range(E):
            mask = (expert_indices == e)
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            x_e = h_flat[token_idx]  # [M_e, K]
            # Fused FP8 gate+up+SwiGLU for this expert.
            inter_bf16 = fp8_moe_expert_gate_up_silu_fused(
                x_e,
                w_gate_fp8,
                w_up_fp8,
                w_gate_scale,
                w_up_scale,

 succeeded in 0ms:
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

exec
/bin/zsh -lc "sed -n '1,260p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '1,280p' triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
"""Packed NVFP4 MoE decode path."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from engine.nvfp4_runtime import _quantize_activation_to_fp4, dual_scalar_bridge
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_effective_scale,
    nvfp4_grouped_down_weighted_sum_prepared,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale,
    nvfp4_grouped_gate_up_silu_fast_decode_prepared,
)
from triton_kernels.shared_expert_gate import (
    HAS_TRITON as HAS_SHARED_EXPERT_GATE_TRITON,
    add_shared_expert_gate_from_scalar_triton,
    apply_shared_expert_gate_triton,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _env_first(names: tuple[str, ...]) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            return raw
    return None


def _w4a8_fake_quant_mode() -> str:
    mode = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE", "off").lower()
    if mode in {"0", "false", "no", "off", ""}:
        return "off"
    if mode not in {"gateup", "full"}:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_ACTIVE must be off, gateup, or full")
    return mode


def _use_moe_effective_scale(w: dict) -> bool:
    return (
        _env_bool("LYNN_MOE_EFFECTIVE_SCALE", False)
        and "mlp.experts._gate_up_effective_scale" in w
        and "mlp.experts._down_effective_scale" in w
    )


def _fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    """Research-only FP8 activation round-trip for W4A8 quality gates.

    This is intentionally controlled by `LYNN_W4A8_FAKE_QUANT_ACTIVE` and never
    used by default. It mirrors P104's best variant: E4M3, per-16 scaling.
    """
    fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3").lower()
    granularity = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16").lower()
    if fmt == "e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e4m3fn
    elif fmt == "e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is required for W4A8 fake quant")
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_FORMAT must be e4m3 or e5m2")

    max_fp8 = float(torch.finfo(fp8_dtype).max)
    x32 = x.float()
    if granularity == "tensor":
        scale = (x32.abs().amax() / max_fp8).clamp_min(1e-8)
        return ((x32 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
    if granularity == "row":
        x2 = x32.view(1, -1) if x32.ndim == 1 else x32
        scale = (x2.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        y = ((x2 / scale).to(fp8_dtype).float() * scale).to(x.dtype)
        return y.view_as(x) if x32.ndim == 1 else y
    if granularity == "per16":
        if x32.shape[-1] % 16 != 0:
            raise ValueError(f"W4A8 per16 fake quant requires last dim divisible by 16, got {tuple(x.shape)}")
        shape = x32.shape
        grouped = x32.reshape(-1, shape[-1] // 16, 16)
        scale = (grouped.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        return ((grouped / scale).to(fp8_dtype).float() * scale).reshape(shape).to(x.dtype)
    raise ValueError("LYNN_W4A8_FAKE_QUANT_GRANULARITY must be tensor, row, or per16")


def _topk_limit_from_env(top_k: int) -> int:
    raw = _env_first(("LYNN_MOE_TOPK_LIMIT", "LYNN_MOE_PROFILE_TOPK_LIMIT"))
    if raw is None:
        return top_k
    limit = int(raw)
    if not (1 <= limit <= top_k):
        raise ValueError(f"MoE top-k limit must be in [1, {top_k}], got {limit}")
    return limit


def _router_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    sorted: bool,
    scratch_owner: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run router top-k, optionally reusing caller-owned output buffers.

    P163 showed `torch.topk(..., out=...)` is bit-exact for the decode shape and
    saves a small but measurable boundary cost.  Keep it opt-in because the
    scratch tensors are stored on the mutable per-layer weight dict.
    """
    if not _env_bool("LYNN_ROUTER_TOPK_OUT_BUFFER", False):
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    if router_logits.ndim != 2 or router_logits.shape[0] != 1:
        return torch.topk(router_logits, top_k, dim=-1, sorted=sorted)
    values_key = "mlp.gate._topk_values_scratch"
    indices_key = "mlp.gate._topk_indices_scratch"
    values = scratch_owner.get(values_key)
    indices = scratch_owner.get(indices_key)
    expected_shape = (1, top_k)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != router_logits.device
        or values.dtype != router_logits.dtype
    ):
        values = torch.empty(expected_shape, device=router_logits.device, dtype=router_logits.dtype)
        scratch_owner[values_key] = values
    if (
        indices is None
        or tuple(indices.shape) != expected_shape
        or indices.device != router_logits.device
        or indices.dtype != torch.long
    ):
        indices = torch.empty(expected_shape, device=router_logits.device, dtype=torch.long)
        scratch_owner[indices_key] = indices
    torch.topk(router_logits, top_k, dim=-1, sorted=sorted, out=(values, indices))
    return values, indices


def _router_linear(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    """Run the MoE router projection, optionally with caller-owned output.

    P177 showed `torch.mm(hidden, gate_weight.t(), out=logits_scratch)` is
    bit-exact for the single-token decode router and removes a small allocation
    boundary. The transposed weight and scratch are attached at load time by the
    resident runner.
    """
    if not _env_bool("LYNN_ROUTER_LINEAR_OUT_BUFFER", False):
        return F.linear(h_flat, w["mlp.gate.weight"])
    weight_t = w.get("mlp.gate.weight_t")
    logits = w.get("mlp.gate._logits_scratch")
    if (
        weight_t is None
        or logits is None
        or h_flat.ndim != 2
        or h_flat.shape[0] != 1
        or logits.shape != (1, w["mlp.gate.weight"].shape[0])
        or logits.device != h_flat.device
        or logits.dtype != h_flat.dtype
    ):
        return F.linear(h_flat, w["mlp.gate.weight"])
    torch.mm(h_flat, weight_t, out=logits)
    return logits


def _router_softmax(
    routing_logits: torch.Tensor,
    *,
    scratch_owner: dict,
) -> torch.Tensor:
    """Run router softmax, optionally reusing a caller-owned float32 buffer."""
    if not _env_bool("LYNN_ROUTER_SOFTMAX_OUT_BUFFER", False):
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    if routing_logits.ndim != 2 or routing_logits.shape[0] != 1:
        return F.softmax(routing_logits, dim=-1, dtype=torch.float32)[0].contiguous()
    values_key = "mlp.gate._softmax_values_scratch"
    expected_shape = tuple(routing_logits.shape)
    values = scratch_owner.get(values_key)
    if (
        values is None
        or tuple(values.shape) != expected_shape
        or values.device != routing_logits.device
        or values.dtype != torch.float32
    ):
        values = torch.empty(expected_shape, device=routing_logits.device, dtype=torch.float32)
        scratch_owner[values_key] = values
    torch.softmax(routing_logits, dim=-1, dtype=torch.float32, out=values)
    return values[0]


def _skip_shared_from_env() -> bool:
    raw = _env_first(("LYNN_MOE_SKIP_SHARED", "LYNN_MOE_PROFILE_SKIP_SHARED"))
    if raw is None:
        return False
    return raw.lower() not in {"0", "false", "no", "off"}


def _apply_shared_expert_gate(h_flat: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    if "mlp.shared_expert_gate.weight" not in w:
        return shared
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch":
        return shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    if backend == "torch_inplace":
        shared.mul_(torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])))
        return shared
    if backend == "triton":
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=triton requires Triton")
        return apply_shared_expert_gate_triton(shared, h_flat, w["mlp.shared_expert_gate.weight"])
    raise ValueError("LYNN_SHARED_EXPERT_GATE_BACKEND must be 'torch', 'torch_inplace', or 'triton', got " f"{backend!r}")


def _add_shared_expert_output(moe_out: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    if _env_bool("LYNN_MOE_ADD_SHARED_INPLACE", False):
        moe_out.add_(shared)
        return moe_out
    return moe_out + shared


def _finalize_shared_expert_output(h_flat: torch.Tensor, moe_out: torch.Tensor, shared: torch.Tensor, w: dict) -> torch.Tensor:
    backend = os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch")
    if backend == "torch_scalar_add_triton" and "mlp.shared_expert_gate.weight" in w:
        if not HAS_SHARED_EXPERT_GATE_TRITON:
            raise RuntimeError("LYNN_SHARED_EXPERT_GATE_BACKEND=torch_scalar_add_triton requires Triton")
        gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        return add_shared_expert_gate_from_scalar_triton(moe_out, shared, gate)
    shared = _apply_shared_expert_gate(h_flat, shared, w)
    return _add_shared_expert_output(moe_out, shared)


def _layer_selected_for_native_cuda(cfg: dict) -> bool:
    spec = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_LAYERS")
    if not spec:
        return True
    layer_idx = cfg.get("layer_idx")
    if layer_idx is None:
        return False
    layer_idx = int(layer_idx)
    from engine.inference_state import LAYER_TYPES


 succeeded in 0ms:
"""Packed NVFP4 MoE kernels for Lynn variable-expert decode."""
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


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 512


def _require_triton() -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for packed NVFP4 MoE kernels")


if HAS_TRITON:

    @triton.jit
    def _e2m1_from_nibble(nibble):
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        val = tl.where(
            mag == 0,
            0.0,
            tl.where(
                mag == 1,
                0.5,
                tl.where(
                    mag == 2,
                    1.0,
                    tl.where(
                        mag == 3,
                        1.5,
                        tl.where(mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))),
                    ),
                ),
            ),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _e2m1_from_nibble_fast(nibble):
        """Same E2M1 table with a shallower expression tree.

        This keeps the exact compressed-tensors value table:
        [0, .5, 1, 1.5, 2, 3, 4, 6].
        """
        mag = nibble & 0x07
        sign = (nibble & 0x08) != 0
        mag_f = mag.to(tl.float32)
        val = tl.where(
            mag <= 4,
            mag_f * 0.5,
            tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
        )
        return tl.where(sign, -val, val)

    @triton.jit
    def _grouped_gate_up_silu_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
            cols = h0 + h_offsets
            col_mask = cols < HIDDEN
            packed_cols = cols // 2
            scale_cols = cols // 16
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            gate_rows = inter_offsets
            up_rows = INTERMEDIATE + inter_offsets
            gate_packed_offsets = (
                expert * PACKED_STRIDE_E
                + gate_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            up_packed_offsets = (
                expert * PACKED_STRIDE_E
                + up_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            gate_scale_offsets = (
                expert * SCALE_STRIDE_E
                + gate_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )
            up_scale_offsets = (
                expert * SCALE_STRIDE_E
                + up_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )

            gate_packed = tl.load(
                gate_up_packed_ptr + gate_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            up_packed = tl.load(
                gate_up_packed_ptr + up_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
            up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
            gate_w = _e2m1_from_nibble(gate_nibble)
            up_w = _e2m1_from_nibble(up_nibble)
            gate_scale = tl.load(
                gate_up_scale_ptr + gate_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_scale_ptr + up_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
            up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_gate_up_silu_scale_hoist_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        group_offsets = tl.arange(0, 16)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in tl.static_range(0, HIDDEN, BLOCK_HIDDEN):
            for hg in tl.static_range(0, BLOCK_HIDDEN, 16):
                cols = h0 + hg + group_offsets
                col_mask = cols < HIDDEN
                packed_cols = cols // 2
                scale_col = (h0 + hg) // 16
                x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

                gate_rows = inter_offsets
                up_rows = INTERMEDIATE + inter_offsets
                gate_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + gate_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                up_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + up_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                gate_packed = tl.load(
                    gate_up_packed_ptr + gate_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                up_packed = tl.load(
                    gate_up_packed_ptr + up_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
                up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
                gate_w = _e2m1_from_nibble_fast(gate_nibble)
                up_w = _e2m1_from_nibble_fast(up_nibble)
                gate_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + gate_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + up_rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=inter_mask,
                    other=0.0,
                ).to(tl.float32)
                gate_acc += tl.sum(gate_w * (gate_scale[:, None] / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale[:, None] / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_gate_up_silu_fast_decode_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        SCALE_EFFECTIVE: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        if SCALE_EFFECTIVE:
            global_scale = 1.0
        else:

codex
I’ve got the first pass through the main modules. The hot path has several existing “micro-fusions” already, so I’m now checking the exact decode layer orchestration and wrapper launch counts to avoid recommending work that’s already done.
exec
/bin/zsh -lc 'rg -n "def decode_full_attn|def decode_linear_attn|def _decode_layer|def _ffn_forward|moe_forward_decode_packed_nvfp4|_moe_forward_decode_packed_nvfp4_fixed_triton|nvfp4_grouped" engine/incremental_decode.py engine/full_forward.py engine/moe_packed_nvfp4.py triton_kernels/nvfp4_moe.py' in /Users/lynn/Downloads/Lynn/lynn-engine
 exited 127 in 0ms:
zsh:1: command not found: rg

exec
/bin/zsh -lc "sed -n '260,700p' triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        SCALE_EFFECTIVE: tl.constexpr,
    ):
        slot = tl.program_id(0)
        block_i = tl.program_id(1)
        expert = tl.load(expert_ids_ptr + slot)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        if SCALE_EFFECTIVE:
            global_scale = 1.0
        else:
            global_scale = tl.load(global_scale_ptr).to(tl.float32)

        gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

        for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
            cols = h0 + h_offsets
            col_mask = cols < HIDDEN
            packed_cols = cols // 2
            scale_cols = cols // 16
            x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

            gate_rows = inter_offsets
            up_rows = INTERMEDIATE + inter_offsets
            gate_packed_offsets = (
                expert * PACKED_STRIDE_E
                + gate_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            up_packed_offsets = (
                expert * PACKED_STRIDE_E
                + up_rows[:, None] * PACKED_STRIDE_M
                + packed_cols[None, :] * PACKED_STRIDE_N
            )
            gate_scale_offsets = (
                expert * SCALE_STRIDE_E
                + gate_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )
            up_scale_offsets = (
                expert * SCALE_STRIDE_E
                + up_rows[:, None] * SCALE_STRIDE_M
                + scale_cols[None, :] * SCALE_STRIDE_G
            )

            gate_packed = tl.load(
                gate_up_packed_ptr + gate_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            up_packed = tl.load(
                gate_up_packed_ptr + up_packed_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0,
            )
            gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
            up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
            gate_w = _e2m1_from_nibble_fast(gate_nibble)
            up_w = _e2m1_from_nibble_fast(up_nibble)
            gate_scale = tl.load(
                gate_up_scale_ptr + gate_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up_scale = tl.load(
                gate_up_scale_ptr + up_scale_offsets,
                mask=inter_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if SCALE_EFFECTIVE:
                gate_acc += tl.sum(gate_w * gate_scale * x[None, :], axis=1)
                up_acc += tl.sum(up_w * up_scale * x[None, :], axis=1)
            else:
                gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        inter = gate_silu * up_acc
        tl.store(inter_ptr + slot * INTER_STRIDE_K + inter_offsets * INTER_STRIDE_I, inter.to(tl.bfloat16), mask=inter_mask)

    @triton.jit
    def _grouped_down_weighted_sum_kernel(
        inter_ptr,
        expert_ids_ptr,
        routing_weights_ptr,
        down_packed_ptr,
        down_scale_ptr,
        global_scale_ptr,
        out_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        SCALE_EFFECTIVE: tl.constexpr,
    ):
        hidden_block = tl.program_id(0)
        rows = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        row_mask = rows < HIDDEN
        inter_offsets = tl.arange(0, BLOCK_INTER)
        if SCALE_EFFECTIVE:
            global_scale = 1.0
        else:
            global_scale = tl.load(global_scale_ptr).to(tl.float32)
        acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            route = tl.load(routing_weights_ptr + slot).to(tl.float32)
            slot_acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
            for i0 in range(0, INTERMEDIATE, BLOCK_INTER):
                cols = i0 + inter_offsets
                col_mask = cols < INTERMEDIATE
                packed_cols = cols // 2
                scale_cols = cols // 16
                inter = tl.load(
                    inter_ptr + slot * INTER_STRIDE_K + cols * INTER_STRIDE_I,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)
                packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )
                packed = tl.load(
                    down_packed_ptr + packed_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                nibble = tl.where((cols[None, :] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
                w = _e2m1_from_nibble(nibble)
                scale = tl.load(
                    down_scale_ptr + scale_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if SCALE_EFFECTIVE:
                    slot_acc += tl.sum(w * scale * inter[None, :], axis=1)
                else:
                    slot_acc += tl.sum(w * (scale / global_scale) * inter[None, :], axis=1)
            acc += route * slot_acc

        tl.store(out_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)

    @triton.jit
    def _grouped_down_weighted_sum_scale_hoist_kernel(
        inter_ptr,
        expert_ids_ptr,
        routing_weights_ptr,
        down_packed_ptr,
        down_scale_ptr,
        global_scale_ptr,
        out_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_K: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        TOP_K: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
    ):
        hidden_block = tl.program_id(0)
        rows = hidden_block * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        row_mask = rows < HIDDEN
        group_offsets = tl.arange(0, 16)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
        acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            route = tl.load(routing_weights_ptr + slot).to(tl.float32)
            slot_acc = tl.zeros((BLOCK_HIDDEN,), dtype=tl.float32)
            for i0 in tl.static_range(0, INTERMEDIATE, 16):
                cols = i0 + group_offsets
                col_mask = cols < INTERMEDIATE
                packed_cols = cols // 2
                scale_col = i0 // 16
                inter = tl.load(
                    inter_ptr + slot * INTER_STRIDE_K + cols * INTER_STRIDE_I,
                    mask=col_mask,
                    other=0.0,
                ).to(tl.float32)
                packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                packed = tl.load(
                    down_packed_ptr + packed_offsets,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                nibble = tl.where((cols[None, :] & 1) == 0, packed & 0x0F, (packed >> 4) & 0x0F)
                w = _e2m1_from_nibble_fast(nibble)
                scale = tl.load(
                    down_scale_ptr
                    + expert * SCALE_STRIDE_E
                    + rows * SCALE_STRIDE_M
                    + scale_col * SCALE_STRIDE_G,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                slot_acc += tl.sum(w * (scale[:, None] / global_scale) * inter[None, :], axis=1)
            acc += route * slot_acc

        tl.store(out_ptr + rows, acc.to(tl.bfloat16), mask=row_mask)

    @triton.jit
    def _grouped_gate_up_silu_merged_topk_kernel(
        x_ptr,
        expert_ids_ptr,
        gate_up_packed_ptr,
        gate_up_scale_ptr,
        global_scale_ptr,
        inter_ptr,
        PACKED_STRIDE_E: tl.constexpr,
        PACKED_STRIDE_M: tl.constexpr,
        PACKED_STRIDE_N: tl.constexpr,
        SCALE_STRIDE_E: tl.constexpr,
        SCALE_STRIDE_M: tl.constexpr,
        SCALE_STRIDE_G: tl.constexpr,
        INTER_STRIDE_I: tl.constexpr,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_INTER: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        block_i = tl.program_id(0)
        inter_offsets = block_i * BLOCK_INTER + tl.arange(0, BLOCK_INTER)
        inter_mask = inter_offsets < INTERMEDIATE
        h_offsets = tl.arange(0, BLOCK_HIDDEN)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        for slot in range(0, TOP_K):
            expert = tl.load(expert_ids_ptr + slot)
            gate_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_INTER,), dtype=tl.float32)

            for h0 in range(0, HIDDEN, BLOCK_HIDDEN):
                cols = h0 + h_offsets
                col_mask = cols < HIDDEN
                packed_cols = cols // 2
                scale_cols = cols // 16
                x = tl.load(x_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

                gate_rows = inter_offsets
                up_rows = INTERMEDIATE + inter_offsets
                gate_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + gate_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                up_packed_offsets = (
                    expert * PACKED_STRIDE_E
                    + up_rows[:, None] * PACKED_STRIDE_M
                    + packed_cols[None, :] * PACKED_STRIDE_N
                )
                gate_scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + gate_rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )
                up_scale_offsets = (
                    expert * SCALE_STRIDE_E
                    + up_rows[:, None] * SCALE_STRIDE_M
                    + scale_cols[None, :] * SCALE_STRIDE_G
                )

                gate_packed = tl.load(
                    gate_up_packed_ptr + gate_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                up_packed = tl.load(
                    gate_up_packed_ptr + up_packed_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0,
                )
                gate_nibble = tl.where((cols[None, :] & 1) == 0, gate_packed & 0x0F, (gate_packed >> 4) & 0x0F)
                up_nibble = tl.where((cols[None, :] & 1) == 0, up_packed & 0x0F, (up_packed >> 4) & 0x0F)
                gate_w = _e2m1_from_nibble(gate_nibble)
                up_w = _e2m1_from_nibble(up_nibble)
                gate_scale = tl.load(
                    gate_up_scale_ptr + gate_scale_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                up_scale = tl.load(
                    gate_up_scale_ptr + up_scale_offsets,
                    mask=inter_mask[:, None] & col_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                gate_acc += tl.sum(gate_w * (gate_scale / global_scale) * x[None, :], axis=1)
                up_acc += tl.sum(up_w * (up_scale / global_scale) * x[None, :], axis=1)

            gate_silu = gate_acc * tl.sigmoid(gate_acc)
            inter = gate_silu * up_acc
            tl.store(
                inter_ptr + slot * INTERMEDIATE + inter_offsets * INTER_STRIDE_I,
                inter.to(tl.bfloat16),
                mask=inter_mask,
            )


def nvfp4_grouped_gate_up_silu(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 64,
    block_hidden: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute top-k `silu(gate(x))*up(x)` from grouped packed NVFP4 weights."""
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_effective_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast gate/up path that consumes precomputed `scale / global_scale`.

    This is an opt-in MoE repack probe.  It keeps the same packed weight layout
    and output contract as `nvfp4_grouped_gate_up_silu_fast_decode`, but removes
    the per-element division from the decode kernel when the runner has attached
    effective scale tensors.
    """
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_effective_scale.ndim != 3:
        raise ValueError(
            "expected grouped 3D tensors, got "
            f"packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_effective_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    if out is None:
        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    else:
        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
            raise ValueError(
                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
                f"for top_k={expert_ids.numel()}"
            )
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as x")
        inter = out[: expert_ids.numel()]
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_effective_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_effective_scale.stride(0),
        gate_up_effective_scale.stride(1),
        gate_up_effective_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,

exec
/bin/zsh -lc "sed -n '260,620p' engine/full_forward.py" in /Users/lynn/Downloads/Lynn/lynn-engine
exec
/bin/zsh -lc "sed -n '260,620p' engine/incremental_decode.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
                w_up_scale,
                expert_id=e,
            )
            # FP8 down via per-token cast + _scaled_mm.
            inter_max = inter_bf16.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
            inter_scale = (inter_max / 448.0).to(torch.float32)
            inter_fp8 = (inter_bf16.to(torch.float32) / inter_scale).to(torch.float8_e4m3fn)
            ffn_e = torch._scaled_mm(
                inter_fp8,
                w_down_fp8[e].t(),  # column-major view: stride(0)==1 satisfies torch._scaled_mm B-arg requirement (no copy)
                scale_a=inter_scale,
                scale_b=w_down_scale[e].view(1, -1).to(torch.float32),
                out_dtype=torch.bfloat16,
            )
            weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
            moe_out.index_add_(0, token_idx, ffn_e * weight_e)

        # Shared expert FP8 path
        if "mlp.shared_expert.gate_proj.weight_fp8" in w:
            from triton_kernels.spark_fp8_gate_up_fused import fp8_gate_up_silu_fused
            shared_inter = fp8_gate_up_silu_fused(
                h_flat,
                w["mlp.shared_expert.gate_proj.weight_fp8"],
                w["mlp.shared_expert.up_proj.weight_fp8"],
                w["mlp.shared_expert.gate_proj.weight_fp8_scale"].to(torch.float32),
                w["mlp.shared_expert.up_proj.weight_fp8_scale"].to(torch.float32),
                auto_block=True,
            )
            # FP8 shared down
            inter_max_s = shared_inter.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
            inter_scale_s = (inter_max_s / 448.0).to(torch.float32)
            inter_fp8_s = (shared_inter.to(torch.float32) / inter_scale_s).to(torch.float8_e4m3fn)
            shared_ffn = torch._scaled_mm(
                inter_fp8_s,
                w["mlp.shared_expert.down_proj.weight_fp8"].t(),  # column-major view: stride(0)==1 satisfies torch._scaled_mm B-arg requirement (no copy)
                scale_a=inter_scale_s,
                scale_b=w["mlp.shared_expert.down_proj.weight_fp8_scale"].view(1, -1).to(torch.float32),
                out_dtype=torch.bfloat16,
            )
            if "mlp.shared_expert_gate.weight" in w:
                shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
                shared_ffn = shared_ffn * shared_gate
            moe_out = moe_out + shared_ffn
        return moe_out.view(B, M, D)

    fused_experts = (
        "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w
    )
    for e in range(E):
        mask = (expert_indices == e)
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        if fused_experts:
            gate_up = F.linear(x_e, w["mlp.experts.gate_up_proj"][e])
            gate_e, up_e = gate_up.chunk(2, dim=-1)
            ffn_e = F.linear(F.silu(gate_e) * up_e, w["mlp.experts.down_proj"][e])
        else:
            gate_e = F.linear(x_e, w[f"mlp.experts.{e}.gate_proj.weight"])
            up_e = F.linear(x_e, w[f"mlp.experts.{e}.up_proj.weight"])
            ffn_e = F.linear(F.silu(gate_e) * up_e, w[f"mlp.experts.{e}.down_proj.weight"])
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out.view(B, M, D)


def _w4a8_fake_quant_mode() -> str:
    """Opt-in FP8 activation round-trip used by W4A8 quality gates.

    The default stays off.  This is intentionally a runtime gate rather than a
    model-format gate so W4A16 artifacts can be compared against W4A8-active
    semantics before native FP8 kernels are promoted.
    """
    mode = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE", "off").lower()
    if mode not in {"off", "gateup", "full"}:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_ACTIVE must be off, gateup, or full")
    return mode


def _fake_quant_fp8_activation(x: torch.Tensor) -> torch.Tensor:
    fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3").lower()
    granularity = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16").lower()
    if fmt == "e4m3":
        fp8_dtype = torch.float8_e4m3fn
        qmax = 448.0
    elif fmt == "e5m2":
        fp8_dtype = torch.float8_e5m2
        qmax = 57344.0
    else:
        raise ValueError("LYNN_W4A8_FAKE_QUANT_FORMAT must be e4m3 or e5m2")

    x_float = x.float()
    if granularity == "tensor":
        scale = x_float.abs().amax().clamp_min(1.0e-6) / qmax
        return (x_float / scale).to(fp8_dtype).to(torch.float32).mul(scale).to(x.dtype)
    if granularity == "row":
        scale = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / qmax
        return (x_float / scale).to(fp8_dtype).to(torch.float32).mul(scale).to(x.dtype)
    if granularity == "per16":
        if x.shape[-1] % 16 != 0:
            raise ValueError(f"W4A8 per16 fake quant requires last dim divisible by 16, got {tuple(x.shape)}")
        groups = x_float.reshape(*x_float.shape[:-1], x_float.shape[-1] // 16, 16)
        scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / qmax
        rounded = (groups / scale).to(fp8_dtype).to(torch.float32).mul(scale)
        return rounded.reshape_as(x_float).to(x.dtype)
    raise ValueError("LYNN_W4A8_FAKE_QUANT_GRANULARITY must be tensor, row, or per16")


def _dense_fp4xfp8_project_scalar(x: torch.Tensor, w: dict, proj: str) -> torch.Tensor:
    """Run one dense FFN projection through the SCALAR REFERENCE kernel (P191 GREEN)."""
    prefix = f"mlp._fp4xfp8.{proj}."
    packed = w[prefix + "weight_packed"]
    scale = w[prefix + "weight_scale"].float()
    global_scale = w[prefix + "weight_global_scale"].float().view(-1)
    n, k_half = packed.shape
    k = k_half * 2
    act_fp8, act_scale = _quantize_to_fp8_e4m3_per16(x.reshape(-1)[:k])
    out = _dense_fp4xfp8_extension().dense_fp4xfp8_scalar_reference(
        act_fp8,
        act_scale,
        packed.contiguous(),
        scale.contiguous(),
        global_scale.contiguous(),
    )
    return out.reshape(*x.shape[:-1], int(n))


def _dense_ffn_forward(h: torch.Tensor, w: dict) -> torch.Tensor:
    """Dense Qwen FFN: down_proj(silu(gate_proj(x)) * up_proj(x)).

    Env knobs for isolation:
      LYNN_DENSE_FFN_TRUE_FP8=1          Enable R6000 sm_120a true FP4xFP8 path
                                         (requires mlp._fp4xfp8.* sidecar)
      LYNN_DENSE_FFN_TRUE_FP8_KERNEL=    mma (default) | scalar
      LYNN_DENSE_FFN_TRUE_FP8_SCOPE=     full (default) | gateup | down

    Phase 2 Spark sm_121 FP8 path (auto-selected when the model dir is
    the lynn-variable-w4a8-fp8-v1 artifact — loader populates the weight
    dict with ``.weight_fp8`` + ``.weight_fp8_scale`` keys instead of the
    plain ``.weight``):

      - When ``mlp.gate_proj.weight_fp8`` and ``mlp.up_proj.weight_fp8``
        are present, the gate+up matmul + SwiGLU goes through the fused
        Triton FP8 kernel (``triton_kernels/spark_fp8_gate_up_fused.py``).
        ``mlp.down_proj.weight_fp8`` similarly drives the down matmul
        through ``torch._scaled_mm`` (no fused SwiGLU there).
      - Opt-out via ``LYNN_DISABLE_W4A8_FP8_PATH=1`` for diagnostic
        comparison against a BF16 fallback (requires the loader to keep
        BF16 sidecars, which it currently does NOT for FP8 artifacts).
    """
    # Phase 2 Spark FP8 path: keyed on weight dict contents (loader-driven).
    fp8_disabled = os.environ.get("LYNN_DISABLE_W4A8_FP8_PATH", "0").lower() in {"1", "true", "yes"}
    if not fp8_disabled and "mlp.gate_proj.weight_fp8" in w and "mlp.up_proj.weight_fp8" in w:
        from triton_kernels.spark_fp8_gate_up_fused import fp8_gate_up_silu_fused
        B, M, D = h.shape
        h_2d = h.reshape(B * M, D)
        inter = fp8_gate_up_silu_fused(
            h_2d,
            w["mlp.gate_proj.weight_fp8"],
            w["mlp.up_proj.weight_fp8"],
            w["mlp.gate_proj.weight_fp8_scale"],
            w["mlp.up_proj.weight_fp8_scale"],
            auto_block=True,
        )
        # down_proj: BF16 reduction tail for now (V2 will add fused FP8 down).
        if "mlp.down_proj.weight_fp8" in w:
            # FP8 down via per-token activation cast + torch._scaled_mm.
            inter_max = inter.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
            inter_scale = (inter_max / 448.0).to(torch.float32)
            inter_fp8 = (inter.to(torch.float32) / inter_scale).to(torch.float8_e4m3fn)
            w_down = w["mlp.down_proj.weight_fp8"]
            w_down_scale = w["mlp.down_proj.weight_fp8_scale"]
            out = torch._scaled_mm(
                inter_fp8,
                w_down.t(),  # column-major view: stride(0)==1 satisfies torch._scaled_mm B-arg requirement (no copy)
                scale_a=inter_scale,
                scale_b=w_down_scale.view(1, -1),
                out_dtype=torch.bfloat16,
            )
            return out.reshape(B, M, -1)
        # BF16 down fallback if down_proj fp8 key absent.
        return torch.nn.functional.linear(
            inter, w.get("mlp.down_proj.weight"),
        ).reshape(B, M, -1) if "mlp.down_proj.weight" in w else inter.reshape(B, M, -1)

    true_fp8_mode = os.environ.get("LYNN_DENSE_FFN_TRUE_FP8", "0").strip().lower()
    is_single_decode_token = h.reshape(-1, h.shape[-1]).shape[0] == 1
    if (
        true_fp8_mode not in {"0", "off", "false", "no"}
        and is_single_decode_token
        and _has_dense_fp4xfp8_sidecar(w)
    ):
        kernel = os.environ.get("LYNN_DENSE_FFN_TRUE_FP8_KERNEL", "mma").strip().lower()
        scope = os.environ.get("LYNN_DENSE_FFN_TRUE_FP8_SCOPE", "full").strip().lower()

        # Select projection function based on kernel type
        if kernel == "scalar":
            proj_fn = _dense_fp4xfp8_project_scalar
        else:
            proj_fn = _dense_fp4xfp8_project

        if scope == "gateup":
            # Only gate/up through FP8; down stays BF16
            gate = proj_fn(h, w, "gate_proj")
            up = proj_fn(h, w, "up_proj")
            inter = F.silu(gate.float()) * up.float()
            out = F.linear(inter.to(h.dtype), w["mlp.down_proj.weight"])
            return out
        elif scope == "down":
            # gate/up stay BF16; only down through FP8
            fused = w.get("mlp._gate_up_proj.weight")
            if fused is not None:
                gate_up = F.linear(h, fused)
                gate, up = gate_up.chunk(2, dim=-1)
            else:
                gate = F.linear(h, w["mlp.gate_proj.weight"])
                up = F.linear(h, w["mlp.up_proj.weight"])
            inter = F.silu(gate) * up
            out = proj_fn(inter, w, "down_proj")
            return out.to(h.dtype)
        else:
            # full: all three projections through FP8
            gate = proj_fn(h, w, "gate_proj")
            up = proj_fn(h, w, "up_proj")
            inter = F.silu(gate.float()) * up.float()
            out = proj_fn(inter, w, "down_proj")
            return out.to(h.dtype)

    w4a8_mode = _w4a8_fake_quant_mode()
    active_h = _fake_quant_fp8_activation(h) if w4a8_mode in {"gateup", "full"} else h
    fused = w.get("mlp._gate_up_proj.weight")
    if fused is not None:
        gate_up = F.linear(active_h, fused)
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = F.linear(active_h, w["mlp.gate_proj.weight"])
        up = F.linear(active_h, w["mlp.up_proj.weight"])
    inter = F.silu(gate) * up
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    return F.linear(inter, w["mlp.down_proj.weight"])


def _ffn_forward(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    if cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        return _moe_forward(h, w, cfg)
    return _dense_ffn_forward(h, w)


def _layer_forward(h: torch.Tensor, position_ids: torch.Tensor, layer_type: str,
                   w: dict, cfg: dict) -> torch.Tensor:
    """One transformer block."""
    # Pre-norm
    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])

    # Attention path
    if layer_type == "linear_attention":
        from engine.qwen36_linear_attn_block import lynn_linear_attn_forward
        attn_out = lynn_linear_attn_forward(h_norm, w)
    elif layer_type == "full_attention":
        attn_out = _full_attn_forward(h_norm, position_ids, w, cfg)
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")
    h = residual + attn_out

    # Post-norm + FFN
    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    ffn_out = _ffn_forward(h_norm, w, cfg)
    return residual + ffn_out


def _with_inferred_layer_config(base_cfg: dict, inferred: dict | None, layer_idx: int | None = None) -> dict:
    """Return a per-layer runtime cfg updated from loaded tensor shapes.

    Vanilla Qwen3.6 has a scalar `num_experts=256`, but Lynn's 27B variable
    skeleton physically slices a different number of experts per layer. The
    loader infers those dimensions from tensors; all forward paths must use the
    per-layer value instead of blindly trusting config.json.
    """
    layer_cfg = dict(base_cfg)
    if inferred:
        layer_cfg.update(inferred)
    if layer_idx is not None:
        layer_cfg["layer_idx"] = int(layer_idx)
    return layer_cfg


# ----------------- outside-weights loader -----------------

def load_outside_weights(model_dir: str, device: str, dtype=torch.bfloat16):
    """Load embeddings + lm_head + final norm.

    Lynn's early internal checkpoints store these tensors in `outside.safetensors`.
    Public HF-style BF16 and NVFP4 artifacts keep them in regular model shards
    (`model.safetensors.index.json`) or a single `model.safetensors`. Support
    both layouts so the same forward code can run on released checkpoints.
    """
    from safetensors import safe_open

    model_path = Path(model_dir)
    keys = [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]

    outside_path = model_path / "outside.safetensors"
    if outside_path.exists():
        weight_map = {k: outside_path.name for k in keys}
    else:
        index_path = model_path / "model.safetensors.index.json"
        single_path = model_path / "model.safetensors"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
        elif single_path.exists():
            weight_map = {k: single_path.name for k in keys}
        else:
            raise FileNotFoundError(
                f"No outside.safetensors, model.safetensors.index.json, or "
                f"model.safetensors found under {model_path}"
            )

    file_to_keys: dict[str, list[str]] = {}
    for k in keys:
        if k not in weight_map:
            raise KeyError(f"Outside tensor {k!r} not found in {model_path}")
        file_to_keys.setdefault(weight_map[k], []).append(k)

    out = {}
    for file_name, file_keys in file_to_keys.items():
        with safe_open(model_path / file_name, framework="pt", device=device) as f:
            for k in file_keys:
                out[k] = f.get_tensor(k).to(dtype)
    return out


# ----------------- incremental greedy decode (Phase 3.1) -----------------

def _prefill_layer(h, position_ids, layer_type, w, cfg, state, layer_idx):
    """Forward one DecoderLayer in prefill mode + populate cache."""
    from engine.incremental_decode import prefill_full_attn, prefill_linear_attn

    residual = h
    h_norm = _rms_norm(h, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)

 succeeded in 0ms:
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
    with profile_section("full_attn_t1.rope"):
        cos, sin = rope_builder(pos_tensor, rotary_dim, rope_theta, h_new.device, h_new.dtype)
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
        if fixed_shape:
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

    # 8. attn_output_gate
    with profile_section("full_attn_t1.gate"):
        attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)

    # 9. o_proj
    with profile_section("full_attn_t1.o_proj_total"):
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
        return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))


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

exec
/bin/zsh -lc "sed -n '260,620p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:

    selected: set[int] = set()
    for raw in spec.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"full", "full_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "full_attention")
        elif item in {"linear", "linear_attention"}:
            selected.update(i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention")
        else:
            selected.add(int(item))
    return layer_idx in selected


def _active_moe_native_cuda_scalar(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Opt-in native CUDA scalar contract path.

    This is intentionally slower than the Triton default today. It exists so
    the real grouped native-FP4 kernel can replace the scalar inner loops behind
    the same runtime contract.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    inter = ext.gate_up_silu_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
    )
    return ext.down_weighted_sum_scalar(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _gate_up_native_cuda_tile_inter(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P55 opt-in tile-inter CUDA scalar gate/up projection."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.gate_up_silu_tile_inter_scalar(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
    )


def _gate_up_native_split16_fp4(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P98 experimental SM120a split16 FP4 MMA gate/up projection.

    This backend quantizes the current decode activation to Lynn-native E2M1
    per-16 format, then consumes the existing packed NVFP4 expert weights
    directly. P98 proved this is not production-promotable yet: the activation
    quantization is not CUDA-graph-capture-safe and changes greedy generations.
    """
    from engine.native_cuda import load_lynn_native_extension

    if (
        _env_bool("LYNN_LINEAR_BLOCK_GRAPH", False)
        or _env_bool("LYNN_LINEAR_BLOCK_GRAPH_REUSE", False)
        or _env_bool("LYNN_LINEAR_BLOCK_GRAPH_PREWARM", False)
    ):
        raise RuntimeError(
            "LYNN_NATIVE_GATEUP_BACKEND=split16_fp4 is an experimental P98 research backend and is not "
            "CUDA-graph-capture-safe. Set LYNN_LINEAR_BLOCK_GRAPH=0, LYNN_LINEAR_BLOCK_GRAPH_REUSE=0, "
            "and LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0 for isolated probes; do not use it in production."
        )

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    act_packed, act_scale = _quantize_activation_to_fp4(hidden.view(1, -1))
    return ext.gate_up_silu_split16_topk_fp4(
        act_packed[0].contiguous(),
        act_scale[0].float().contiguous(),
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"].float(),
        _env_int("LYNN_NATIVE_FP4_MMA_SCALE_BYTE", 127),
    )


def _down_weighted_sum_native_cuda_tile(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """P48 opt-in tile-hidden non-atomic down projection."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.down_weighted_sum_tile_scalar(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _active_moe_native_cuda_scalar_contract(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """One-call native active-MoE contract for the future grouped FP4 kernel.

    The P45 implementation still delegates to the scalar reference kernels
    inside the extension.  Its purpose is to freeze the Python/CUDA ABI that the
    true grouped/block-diagonal FP4 kernel will replace.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_scalar_contract(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _active_moe_native_grouped_per16(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Reserved production ABI for the real grouped per-16 FP4 kernel.

    P56/P58 closed the tempting scalar tile-inter bridge: it has local speed
    signal, but full-generate greedy drift.  Keep a named backend for the real
    implementation so experiments fail loudly instead of accidentally falling
    back to a rejected scalar path.
    """
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_contract(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _active_moe_native_grouped_per16_fused(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Fail-loud ABI for the true one-boundary grouped per-16 active kernel."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_fused_contract(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
    )


def _active_moe_native_grouped_per16_nonatomic(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Native-owned scratch, non-atomic grouped per-16 reference backend."""
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_nonatomic_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _active_moe_packed_pretransposed_graphsafe_v31(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """V3.1 graph-safe pretransposed MoE: true zero-allocation hot path.

    Load-time prep (lazy, first call only):
      - Dequant packed NVFP4 gate_up/down weights to BF16
      - Pretranspose: W_fused_T [2048, 8192], W_down_T [8, 512, 2048]
      - Preallocate scratch: gate_up[1,8192], inter[8,1,512], down[8,1,2048], out[2048]

    Hot path: mm_out + silu_out + mul_ + bmm_out + custom CUDA reduce kernel.
    No torch::empty/zeros/to/sum/new tensor in hot path.
    """
    from engine.native_cuda import load_lynn_native_extension

    # ── Lazy one-time weight prep (first call per layer) ──
    if "_graphsafe_v31_W_fused_T" not in w:
        _prepare_graphsafe_v31_weights(w, hidden.device)

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))

    # Gather experts into pretransposed scratch (pre-graph region)
    _gather_and_pretranspose_v31(w, expert_ids)

    x_2d = hidden.view(1, 2048)
    return ext.moe_packed_pretransposed_graphsafe_v3(
        x_2d,
        routing_weights,
        w["_graphsafe_v31_W_fused_T"],
        w["_graphsafe_v31_W_down_T"],
        w["_graphsafe_v31_gate_up_scratch"],
        w["_graphsafe_v31_inter_scratch"],
        w["_graphsafe_v31_down_scratch"],
        w["_graphsafe_v31_out_scratch"],
    )


def _prepare_graphsafe_v32_weights(w: dict, device) -> None:
    """One-time scratch allocation for V3.2 graph-safe exact scalar path.

    Does NOT pre-dequant weights. Keeps original packed NVFP4 weights.
    Only allocates caller-owned scratch buffers for intermediate and output.
    """
    top_k = 8
    w["_graphsafe_v32_inter_scratch"] = torch.empty(top_k, 512, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v32_out_scratch"] = torch.empty(2048, device=device, dtype=torch.bfloat16)


def _active_moe_packed_pretransposed_graphsafe_v32_ordered(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """V3.2 graph-safe exact scalar MoE: caller-owned scratch, native FP4 dequant.

    Load-time prep (lazy, first call only):
      - Preallocate scratch: inter[8,512], out[2048]
      - Weights stay in original packed NVFP4 format (no pre-dequant)

    Hot path: native FP4→FP32 scalar gate/up + native FP4→FP32 scalar down.
    No torch::empty/zeros/to/sum/new tensor in hot path.
    Route weights kept in FP32 (no BF16 truncation).
    Slot order strictly follows expert_ids.

    Designed to match Triton reference exactly (P37 exact 3/3 target).
    Slower than V3.1 cuBLAS path.
    """
    from engine.native_cuda import load_lynn_native_extension

    if "_graphsafe_v32_inter_scratch" not in w:
        _prepare_graphsafe_v32_weights(w, hidden.device)

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))

    ext.active_moe_scalar_out_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        w["_graphsafe_v32_inter_scratch"],
        w["_graphsafe_v32_out_scratch"],
    )
    return w["_graphsafe_v32_out_scratch"]


def _prepare_graphsafe_v31_weights(w: dict, device) -> None:
    """One-time scratch allocation for V3.1 graph-safe path.

    Does NOT pre-dequant all 256 experts (too much memory).
    Instead, preallocates scratch buffers. Per-call dequant of 8 selected
    experts happens in _gather_and_pretranspose_v31.
    """
    top_k = 8
    w["_graphsafe_v31_W_fused_T"] = torch.empty(2048, top_k * 1024, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_W_down_T"] = torch.empty(top_k, 512, 2048, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_gate_up_scratch"] = torch.empty(1, top_k * 1024, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_inter_scratch"] = torch.empty(top_k, 1, 512, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_down_scratch"] = torch.empty(top_k, 1, 2048, device=device, dtype=torch.bfloat16)
    w["_graphsafe_v31_out_scratch"] = torch.empty(2048, device=device, dtype=torch.bfloat16)


def _dequant_nvfp4_slot(packed, scale, global_scale, device):
    """Dequant a single slot [rows, cols/2] → [rows, cols] BF16."""
    E2M1_TABLE = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    low = (packed & 0x0F).int()
    high = ((packed >> 4) & 0x0F).int()
    low_val = E2M1_TABLE[low & 7] * (1 - 2 * ((low >> 3) & 1).float())
    high_val = E2M1_TABLE[high & 7] * (1 - 2 * ((high >> 3) & 1).float())
    K = packed.shape[-1] * 2
    result = torch.zeros(*packed.shape[:-1], K, device=device, dtype=torch.float32)
    result[..., 0::2] = low_val
    result[..., 1::2] = high_val
    inv_g = 1.0 / global_scale.float().item()
    se = scale.float().unsqueeze(-1).expand(*scale.shape, 16).reshape(*packed.shape[:-1], K)
    return (result * se * inv_g).to(torch.bfloat16)


def _gather_and_pretranspose_v31(w: dict, expert_ids: torch.Tensor) -> None:
    """Dequant + gather 8 selected experts into pretransposed scratch."""
    device = expert_ids.device
    gu_packed = w["mlp.experts._gate_up_packed"]       # [256, 1024, 1024]

codex
`rg` is not installed here, so I used direct reads. The decode path is clearer now: the big levers are around MoE active top-k, shared expert, router, and attention micro-ops rather than raw weight bytes.
exec
/bin/zsh -lc "sed -n '620,1040p' engine/incremental_decode.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
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

exec
/bin/zsh -lc "grep -n \"def decode_full_attn\\|def decode_linear_attn\\|def _decode_layer\\|def _ffn_forward\\|moe_forward_decode_packed_nvfp4\\|_moe_forward_decode_packed_nvfp4_fixed_triton\\|nvfp4_grouped\" engine/incremental_decode.py engine/full_forward.py engine/moe_packed_nvfp4.py triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
engine/incremental_decode.py:382:def decode_full_attn(h_new, new_position_id, w, cfg, K_cache_full, V_cache_full,
engine/incremental_decode.py:516:def decode_full_attn_k2(
engine/incremental_decode.py:780:def decode_full_attn_block(
engine/incremental_decode.py:1028:def decode_linear_attn_k2(
engine/incremental_decode.py:1062:def decode_linear_attn_block(
engine/incremental_decode.py:1090:def decode_linear_attn(h_new, w, recurrent_state, conv_state, *, recurrent_backend: str = "torch"):
engine/full_forward.py:515:def _ffn_forward(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
engine/full_forward.py:652:        from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4 as _moe
engine/full_forward.py:658:def _decode_layer_k2(
engine/full_forward.py:835:def _decode_layer_block(
engine/full_forward.py:964:def _decode_layer(
engine/moe_packed_nvfp4.py:11:    nvfp4_grouped_down_weighted_sum,
engine/moe_packed_nvfp4.py:12:    nvfp4_grouped_down_weighted_sum_effective_scale,
engine/moe_packed_nvfp4.py:13:    nvfp4_grouped_down_weighted_sum_prepared,
engine/moe_packed_nvfp4.py:14:    nvfp4_grouped_gate_up_silu,
engine/moe_packed_nvfp4.py:15:    nvfp4_grouped_gate_up_silu_fast_decode,
engine/moe_packed_nvfp4.py:16:    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale,
engine/moe_packed_nvfp4.py:17:    nvfp4_grouped_gate_up_silu_fast_decode_prepared,
engine/moe_packed_nvfp4.py:680:def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
engine/moe_packed_nvfp4.py:711:        inter = nvfp4_grouped_gate_up_silu_fast_decode_prepared(
engine/moe_packed_nvfp4.py:726:            nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
engine/moe_packed_nvfp4.py:728:            else nvfp4_grouped_gate_up_silu_fast_decode
engine/moe_packed_nvfp4.py:743:        inter = nvfp4_grouped_gate_up_silu(
engine/moe_packed_nvfp4.py:754:        inter = nvfp4_grouped_gate_up_silu(
engine/moe_packed_nvfp4.py:781:        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
engine/moe_packed_nvfp4.py:793:            nvfp4_grouped_down_weighted_sum_effective_scale
engine/moe_packed_nvfp4.py:795:            else nvfp4_grouped_down_weighted_sum
engine/moe_packed_nvfp4.py:809:        moe_out = nvfp4_grouped_down_weighted_sum(
engine/moe_packed_nvfp4.py:899:def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
engine/moe_packed_nvfp4.py:958:        return _moe_forward_decode_packed_nvfp4_fixed_triton(h, w, cfg)
engine/moe_packed_nvfp4.py:1026:                    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
engine/moe_packed_nvfp4.py:1028:                    else nvfp4_grouped_gate_up_silu_fast_decode
engine/moe_packed_nvfp4.py:1044:                inter = nvfp4_grouped_gate_up_silu(
engine/moe_packed_nvfp4.py:1055:                inter = nvfp4_grouped_gate_up_silu(
engine/moe_packed_nvfp4.py:1082:                    nvfp4_grouped_down_weighted_sum_effective_scale
engine/moe_packed_nvfp4.py:1084:                    else nvfp4_grouped_down_weighted_sum
triton_kernels/nvfp4_moe.py:596:def nvfp4_grouped_gate_up_silu(
triton_kernels/nvfp4_moe.py:642:def nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
triton_kernels/nvfp4_moe.py:657:    and output contract as `nvfp4_grouped_gate_up_silu_fast_decode`, but removes
triton_kernels/nvfp4_moe.py:707:def nvfp4_grouped_gate_up_silu_scale_hoist(
triton_kernels/nvfp4_moe.py:760:def nvfp4_grouped_gate_up_silu_fast_decode(
triton_kernels/nvfp4_moe.py:818:def nvfp4_grouped_gate_up_silu_fast_decode_prepared(
triton_kernels/nvfp4_moe.py:884:def nvfp4_grouped_gate_up_silu_merged_topk(
triton_kernels/nvfp4_moe.py:936:def nvfp4_grouped_down_weighted_sum(
triton_kernels/nvfp4_moe.py:996:def nvfp4_grouped_down_weighted_sum_effective_scale(
triton_kernels/nvfp4_moe.py:1057:def nvfp4_grouped_down_weighted_sum_prepared(
triton_kernels/nvfp4_moe.py:1118:def nvfp4_grouped_down_weighted_sum_scale_hoist(
triton_kernels/nvfp4_moe.py:1174:    "nvfp4_grouped_down_weighted_sum",
triton_kernels/nvfp4_moe.py:1175:    "nvfp4_grouped_down_weighted_sum_effective_scale",
triton_kernels/nvfp4_moe.py:1176:    "nvfp4_grouped_down_weighted_sum_prepared",
triton_kernels/nvfp4_moe.py:1177:    "nvfp4_grouped_down_weighted_sum_scale_hoist",
triton_kernels/nvfp4_moe.py:1178:    "nvfp4_grouped_gate_up_silu",
triton_kernels/nvfp4_moe.py:1179:    "nvfp4_grouped_gate_up_silu_fast_decode",
triton_kernels/nvfp4_moe.py:1180:    "nvfp4_grouped_gate_up_silu_fast_decode_effective_scale",
triton_kernels/nvfp4_moe.py:1181:    "nvfp4_grouped_gate_up_silu_fast_decode_prepared",
triton_kernels/nvfp4_moe.py:1182:    "nvfp4_grouped_gate_up_silu_merged_topk",
triton_kernels/nvfp4_moe.py:1183:    "nvfp4_grouped_gate_up_silu_scale_hoist",

exec
/bin/zsh -lc "sed -n '700,1180p' triton_kernels/nvfp4_moe.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
        BLOCK_HIDDEN=block_hidden,
        SCALE_EFFECTIVE=True,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_scale_hoist(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """P53 scale-hoisted gate/up probe.

    This variant loads each per-16 scale once per group instead of repeating it
    across the 16 columns. It is intentionally opt-in until full-generate gates
    prove both speed and numerical stability.
    """
    _require_triton()
    if block_hidden % 16 != 0:
        raise ValueError(f"block_hidden must be divisible by 16, got {block_hidden}")
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_scale_hoist_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """P53 lightweight probe: same kernel shape, faster E2M1 expression only."""
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    if out is None:
        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    else:
        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
            raise ValueError(
                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
                f"for top_k={expert_ids.numel()}"
            )
        if out.device != x.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as x")
        inter = out[: expert_ids.numel()]
    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        SCALE_EFFECTIVE=False,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_gate_up_silu_fast_decode_prepared(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    scale_effective: bool = False,
) -> torch.Tensor:
    """Prepared-shape gate/up entrypoint for the resident decode hot path.

    The public wrappers defensively cast and contiguous-copy inputs. The fixed
    resident W4A16 path already prepares these tensors. This exact wrapper keeps
    the same Triton kernel and constants, but avoids redundant boundary ops.
    """
    _require_triton()
    top_k = expert_ids.numel()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != x.device:
        raise ValueError("expert_ids must be a contiguous int32 tensor on x.device")
    if not expert_ids.is_contiguous():
        raise ValueError("expert_ids must be contiguous")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    if not (
        x.is_contiguous()
        and gate_up_packed.is_contiguous()
        and gate_up_scale.is_contiguous()
        and gate_up_global_scale.is_contiguous()
    ):
        raise ValueError("prepared gate/up tensors must be contiguous")
    if out.ndim != 2 or out.shape[0] < top_k or out.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)}")
    if out.device != x.device or out.dtype != torch.bfloat16:
        raise ValueError("out must be a bfloat16 tensor on x.device")
    inter = out[:top_k]
    grid = (top_k, triton.cdiv(INTERMEDIATE_SIZE, 8))
    _grouped_gate_up_silu_fast_decode_kernel[grid](
        x,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global_scale,
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_INTER=8,
        BLOCK_HIDDEN=256,
        SCALE_EFFECTIVE=scale_effective,
        num_warps=4,
    )
    return inter


def nvfp4_grouped_gate_up_silu_merged_topk(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    *,
    block_inter: int = 8,
    block_hidden: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Gate/up variant with one program per inter block and a top-k inner loop.

    This is an opt-in P26 probe, not the production default. It tests whether
    reducing kernel program count from `[top_k, inter_blocks]` to
    `[inter_blocks]` helps launch/scheduling overhead while preserving the
    current per-16 scalar contract.
    """
    _require_triton()
    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(INTERMEDIATE_SIZE, block_inter),)
    _grouped_gate_up_silu_merged_topk_kernel[grid](
        x.contiguous(),
        expert_ids,
        gate_up_packed.contiguous(),
        gate_up_scale.contiguous(),
        gate_up_global_scale.to(device=x.device).contiguous(),
        inter,
        gate_up_packed.stride(0),
        gate_up_packed.stride(1),
        gate_up_packed.stride(2),
        gate_up_scale.stride(0),
        gate_up_scale.stride(1),
        gate_up_scale.stride(2),
        inter.stride(1),
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        TOP_K=expert_ids.numel(),
        BLOCK_INTER=block_inter,
        BLOCK_HIDDEN=block_hidden,
        num_warps=num_warps,
    )
    return inter


def nvfp4_grouped_down_weighted_sum(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 16,
    block_inter: int = 128,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted top-k down projection from grouped packed NVFP4 weights."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if down_packed.ndim != 3 or down_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    if out is None:
        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    else:
        if out.shape != (HIDDEN_SIZE,):
            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
        if out.device != inter.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
    _grouped_down_weighted_sum_kernel[grid](
        inter.contiguous(),
        expert_ids,
        routing_weights,
        down_packed.contiguous(),
        down_scale.contiguous(),
        down_global_scale.to(device=inter.device).contiguous(),
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=block_hidden,
        BLOCK_INTER=block_inter,
        SCALE_EFFECTIVE=False,
        num_warps=num_warps,
    )
    return out


def nvfp4_grouped_down_weighted_sum_effective_scale(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_effective_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 16,
    block_inter: int = 128,
    num_warps: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Down projection consuming precomputed `scale / global_scale`."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if down_packed.ndim != 3 or down_effective_scale.ndim != 3:
        raise ValueError(
            "expected grouped 3D tensors, got "
            f"packed={tuple(down_packed.shape)} scale={tuple(down_effective_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    if out is None:
        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    else:
        if out.shape != (HIDDEN_SIZE,):
            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
        if out.device != inter.device or out.dtype != torch.bfloat16:
            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
    _grouped_down_weighted_sum_kernel[grid](
        inter.contiguous(),
        expert_ids,
        routing_weights,
        down_packed.contiguous(),
        down_effective_scale.contiguous(),
        down_global_scale.to(device=inter.device).contiguous(),
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_effective_scale.stride(0),
        down_effective_scale.stride(1),
        down_effective_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=block_hidden,
        BLOCK_INTER=block_inter,
        SCALE_EFFECTIVE=True,
        num_warps=num_warps,
    )
    return out


def nvfp4_grouped_down_weighted_sum_prepared(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    scale_effective: bool = False,
) -> torch.Tensor:
    """Prepared-shape down entrypoint for the resident decode hot path."""
    _require_triton()
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != inter.device:
        raise ValueError("expert_ids must be a contiguous int32 tensor on inter.device")
    if routing_weights.ndim != 1 or routing_weights.dtype != torch.float32 or routing_weights.device != inter.device:
        raise ValueError("routing_weights must be a contiguous float32 tensor on inter.device")
    if not expert_ids.is_contiguous() or not routing_weights.is_contiguous():
        raise ValueError("expert_ids and routing_weights must be contiguous")
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    if down_packed.ndim != 3 or down_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
        )
    if not (inter.is_contiguous() and down_packed.is_contiguous() and down_scale.is_contiguous() and down_global_scale.is_contiguous()):
        raise ValueError("prepared down tensors must be contiguous")
    if out.shape != (HIDDEN_SIZE,):
        raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
    if out.device != inter.device or out.dtype != torch.bfloat16:
        raise ValueError("out must be a bfloat16 tensor on inter.device")
    grid = (triton.cdiv(HIDDEN_SIZE, 8),)
    _grouped_down_weighted_sum_kernel[grid](
        inter,
        expert_ids,
        routing_weights,
        down_packed,
        down_scale,
        down_global_scale,
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=8,
        BLOCK_INTER=512,
        SCALE_EFFECTIVE=scale_effective,
        num_warps=8,
    )
    return out


def nvfp4_grouped_down_weighted_sum_scale_hoist(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
    *,
    block_hidden: int = 8,
    block_inter: int = 512,
    num_warps: int = 8,
) -> torch.Tensor:
    """P53 scale-hoisted down weighted-sum probe."""
    _require_triton()
    if block_inter != INTERMEDIATE_SIZE:
        raise ValueError("scale-hoist down currently expects block_inter=512")
    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
    if down_packed.ndim != 3 or down_scale.ndim != 3:
        raise ValueError(
            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
        )
    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
        raise ValueError("expert_ids/routing_weights must match inter top_k")
    out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
    _grouped_down_weighted_sum_scale_hoist_kernel[grid](
        inter.contiguous(),
        expert_ids,
        routing_weights,
        down_packed.contiguous(),
        down_scale.contiguous(),
        down_global_scale.to(device=inter.device).contiguous(),
        out,
        down_packed.stride(0),
        down_packed.stride(1),
        down_packed.stride(2),
        down_scale.stride(0),
        down_scale.stride(1),
        down_scale.stride(2),
        inter.stride(0),
        inter.stride(1),
        TOP_K=inter.shape[0],
        HIDDEN=HIDDEN_SIZE,
        INTERMEDIATE=INTERMEDIATE_SIZE,
        BLOCK_HIDDEN=block_hidden,
        BLOCK_INTER=block_inter,
        num_warps=num_warps,
    )
    return out


__all__ = [
    "HAS_TRITON",
    "nvfp4_grouped_down_weighted_sum",
    "nvfp4_grouped_down_weighted_sum_effective_scale",
    "nvfp4_grouped_down_weighted_sum_prepared",
    "nvfp4_grouped_down_weighted_sum_scale_hoist",
    "nvfp4_grouped_gate_up_silu",
    "nvfp4_grouped_gate_up_silu_fast_decode",
    "nvfp4_grouped_gate_up_silu_fast_decode_effective_scale",

exec
/bin/zsh -lc "sed -n '620,1100p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
    gu_packed = w["mlp.experts._gate_up_packed"]       # [256, 1024, 1024]
    gu_scale = w["mlp.experts._gate_up_scale"]         # [256, 1024, 128]
    gu_global = w["mlp.experts._gate_up_global_scale"] # scalar
    d_packed = w["mlp.experts._down_packed"]            # [256, 2048, 256]
    d_scale = w["mlp.experts._down_scale"]              # [256, 2048, 32]
    d_global = w["mlp.experts._down_global_scale"]      # scalar

    W_fused_T = w["_graphsafe_v31_W_fused_T"]
    W_down_T = w["_graphsafe_v31_W_down_T"]

    ids = expert_ids.long()
    # Dequant only the 8 selected experts
    slot_gu_packed = gu_packed[ids]   # [8, 1024, 1024]
    slot_gu_scale = gu_scale[ids]     # [8, 1024, 128]
    slot_d_packed = d_packed[ids]     # [8, 2048, 256]
    slot_d_scale = d_scale[ids]       # [8, 2048, 32]

    slot_gu_bf16 = _dequant_nvfp4_slot(slot_gu_packed, slot_gu_scale, gu_global, device)  # [8, 1024, 2048]
    slot_d_bf16 = _dequant_nvfp4_slot(slot_d_packed, slot_d_scale, d_global, device)      # [8, 2048, 512]

    # Pretranspose into scratch
    W_fused_T.copy_(slot_gu_bf16.reshape(8 * 1024, 2048).t())
    W_down_T.copy_(slot_d_bf16.transpose(1, 2))


def _active_moe_native_grouped_per16_nonatomic_out(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict,
) -> torch.Tensor:
    """Caller-owned scratch variant for CUDA graph capture probes."""
    from engine.native_cuda import load_lynn_native_extension

    inter_scratch = w.get("mlp.experts._active_inter_scratch")
    out_scratch = w.get("mlp.experts._active_out_scratch")
    if inter_scratch is None or out_scratch is None:
        raise RuntimeError(
            "LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic_out requires "
            "LYNN_MOE_ACTIVE_SCRATCH=1 so resident_runner preallocates MoE scratch."
        )

    ext = load_lynn_native_extension(verbose=_env_bool("LYNN_NATIVE_CUDA_VERBOSE", False))
    return ext.active_moe_grouped_per16_nonatomic_out_reference(
        hidden,
        expert_ids,
        routing_weights,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        inter_scratch,
        out_scratch,
        _env_int("LYNN_NATIVE_GATEUP_TILE_INTER", 2),
        _env_int("LYNN_NATIVE_DOWN_TILE_HIDDEN", 2),
    )


def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Fixed-config production fast path for the current R6000 best profile."""
    h_flat = h.reshape(-1, h.shape[-1])
    router_logits = _router_linear(h_flat, w)
    top_k = int(cfg["num_experts_per_tok"])
    routing_weights, expert_indices = _router_topk(
        router_logits,
        top_k,
        sorted=False,
        scratch_owner=w,
    )
    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    limit = _topk_limit_from_env(top_k)
    if limit != top_k:
        expert_ids = expert_ids[:limit].contiguous()
        routing_weights = routing_weights[:limit].contiguous()
        if _env_bool("LYNN_MOE_TOPK_RENORMALIZE", True):
            routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
    hidden = h_flat[0]
    w4a8_mode = _w4a8_fake_quant_mode()
    if w4a8_mode in {"gateup", "full"}:
        hidden = _fake_quant_fp8_activation(hidden)
    gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
    prepared_triton = _env_bool("LYNN_MOE_TRITON_PREPARED", False)
    if prepared_triton:
        if gateup_backend != "triton_fast_decode":
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
        inter_scratch = w.get("mlp.experts._active_inter_scratch")
        if inter_scratch is None:
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        inter = nvfp4_grouped_gate_up_silu_fast_decode_prepared(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            inter_scratch,
            scale_effective=_use_moe_effective_scale(w),
        )
    elif gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
        inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
    elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
        inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
    elif gateup_backend == "triton_fast_decode":
        gateup_fn = (
            nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
            if _use_moe_effective_scale(w)
            else nvfp4_grouped_gate_up_silu_fast_decode
        )
        inter = gateup_fn(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_effective_scale"]
            if _use_moe_effective_scale(w)
            else w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    elif gateup_backend == "triton":
        inter = nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
        inter = nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
    else:
        raise ValueError(
            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
            "or 'split16_fp4', got "
            f"{gateup_backend!r}"
        )
    if w4a8_mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
    elif prepared_triton:
        if down_backend != "triton":
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
        out_scratch = w.get("mlp.experts._active_out_scratch")
        if out_scratch is None:
            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            out_scratch,
            scale_effective=_use_moe_effective_scale(w),
        ).reshape_as(h_flat)
    elif down_backend == "triton":
        down_fn = (
            nvfp4_grouped_down_weighted_sum_effective_scale
            if _use_moe_effective_scale(w)
            else nvfp4_grouped_down_weighted_sum
        )
        moe_out = down_fn(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    elif down_backend == "cuda_tile":
        moe_out = nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        ).reshape_as(h_flat)
    else:
        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")

    if _skip_shared_from_env():
        return moe_out.to(h.dtype).reshape_as(h)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if "mlp.shared_expert._gate_up_proj.weight" in w:
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
    return moe_out.to(h.dtype).reshape_as(h)


def moe_forward_verify_smallm_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Small-M grouped MoE for the MTP spec-verify block (M = K_draft+1).

    Replaces the per-position T=1 MoE loop: routes each of the M positions, then
    for each UNIQUE active expert runs its gate_up+down ONCE over its gathered
    rows (a small-M tile) instead of once per position; the dense shared expert
    is one batched pass over all M rows. Routed dequant uses the canonical
    ``_dequant_nvfp4_slot`` (W4A16 BF16 numerics) so each row matches the T=1
    packed path -> token-exact intent. Eager only (host syncs in grouping).
    Gated by LYNN_MTP_VERIFY_SMALLM=1.
    """
    h_flat = h.reshape(-1, h.shape[-1])
    M, hidden = h_flat.shape
    top_k = int(cfg["num_experts_per_tok"])
    dev = h_flat.device
    out_dtype = h_flat.dtype

    eids = torch.empty(M, top_k, dtype=torch.long, device=dev)
    wts = torch.empty(M, top_k, dtype=torch.float32, device=dev)
    sorted_topk = _env_bool("LYNN_ROUTER_TOPK_SORTED", False)
    for i in range(M):
        rl = _router_linear(h_flat[i : i + 1], w)
        rw, ei = _router_topk(rl, top_k, sorted=sorted_topk, scratch_owner=w)
        rw = _router_softmax(rw, scratch_owner=w)
        eids[i] = ei[0].to(torch.long)
        wts[i] = rw.reshape(-1).to(torch.float32)

    gu_packed = w["mlp.experts._gate_up_packed"]
    gu_scale = w["mlp.experts._gate_up_scale"]
    gu_g = w["mlp.experts._gate_up_global_scale"]
    d_packed = w["mlp.experts._down_packed"]
    d_scale = w["mlp.experts._down_scale"]
    d_g = w["mlp.experts._down_global_scale"]

    x32 = h_flat.to(torch.float32)
    moe_out = torch.zeros(M, hidden, device=dev, dtype=torch.float32)
    for e in torch.unique(eids).tolist():
        sel = eids == e  # [M, top_k]
        rows = sel.any(dim=1).nonzero(as_tuple=False).flatten()
        rw_e = (wts * sel.to(torch.float32)).sum(dim=1)[rows]  # [m_e]
        xr = x32[rows]  # [m_e, hidden]
        gu = _dequant_nvfp4_slot(gu_packed[e], gu_scale[e], gu_g, dev).to(torch.float32)  # [2*inter, hidden]
        inter = xr @ gu.t()  # [m_e, 2*inter]
        half = inter.shape[-1] // 2
        act = F.silu(inter[:, :half]) * inter[:, half:]  # [m_e, inter]
        dn = _dequant_nvfp4_slot(d_packed[e], d_scale[e], d_g, dev).to(torch.float32)  # [hidden, inter]
        moe_out[rows] += (act @ dn.t()) * rw_e[:, None]

    moe_out = moe_out.to(out_dtype)  # bf16, matching the packed kernel's output
    if not _skip_shared_from_env() and "mlp.shared_expert.gate_proj.weight" in w:
        if "mlp.shared_expert._gate_up_proj.weight" in w:
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)
    return moe_out.reshape_as(h)


def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Decode-only MoE using packed NVFP4 expert weights.

    Router and shared expert stay on the existing BF16 resident path for now;
    active routed experts consume packed NVFP4 weights directly. This is the
    first production-shaped bridge from P10-H into the resident runner.
    """
    if h.shape[1] != 1:
        raise NotImplementedError("packed NVFP4 MoE path is decode-only")
    required = (
        "mlp.experts._gate_up_packed",
        "mlp.experts._gate_up_scale",
        "mlp.experts._gate_up_global_scale",
        "mlp.experts._down_packed",
        "mlp.experts._down_scale",
        "mlp.experts._down_global_scale",
    )
    missing = [key for key in required if key not in w]
    if missing:
        raise KeyError(f"packed NVFP4 MoE aliases missing: {missing}")

    h_flat = h.reshape(-1, h.shape[-1])
    if h_flat.shape[0] != 1:
        raise NotImplementedError("packed NVFP4 MoE path currently supports batch=1")
    if _env_bool("LYNN_MOE_FAST_FIXED", True):
        if _env_bool("LYNN_ROUTER_TOPK_SORTED", False):
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_ROUTER_TOPK_SORTED=0")
        if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
            raise RuntimeError("LYNN_MOE_FAST_FIXED does not support LYNN_MOE_PROFILE_SKIP_ACTIVE")
        if os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton") != "triton":
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton")
        if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") not in {
            "triton",
            "triton_fast_decode",
            "cuda_tile_inter",
            "split16_fp4",
        }:
            raise RuntimeError(
                "LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_GATEUP_BACKEND=triton, "
                "triton_fast_decode, cuda_tile_inter, or split16_fp4"
            )
        if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") not in {"triton", "cuda_tile"}:
            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_DOWN_BACKEND=triton or cuda_tile")
        if _env_bool("LYNN_MOE_TRITON_PREPARED", False):
            if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") != "triton_fast_decode":
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
            if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") != "triton":
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
            if not _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False):
                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
        if (
            _env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
            _env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
            _env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            _env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            _env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
            _env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
        ) not in {(8, 256, 8, 512, 4, 8), (8, 256, 4, 512, 4, 8)}:
            raise RuntimeError("LYNN_MOE_FAST_FIXED only supports the R6000-best or Spark-best MoE kernel config")
        return _moe_forward_decode_packed_nvfp4_fixed_triton(h, w, cfg)

    router_logits = _router_linear(h_flat, w)
    routing_weights, expert_indices = _router_topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
        scratch_owner=w,
    )
    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
    # Triton kernels consume int32 expert ids. Keep this as int32 once here so
    # gate/up and down do not each pay a tiny per-layer dtype conversion.
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    topk_limit = os.environ.get("LYNN_MOE_PROFILE_TOPK_LIMIT")
    if topk_limit:
        limit = int(topk_limit)
        if not (1 <= limit <= expert_ids.numel()):
            raise ValueError(f"LYNN_MOE_PROFILE_TOPK_LIMIT must be in [1, {expert_ids.numel()}], got {limit}")
        expert_ids = expert_ids[:limit].contiguous()
        routing_weights = routing_weights[:limit].contiguous()
        routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
    hidden = h_flat[0]
    w4a8_mode = _w4a8_fake_quant_mode()
    if w4a8_mode in {"gateup", "full"}:
        hidden = _fake_quant_fp8_activation(hidden)

    if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
        moe_out = torch.zeros_like(h_flat)
    else:
        backend = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton")
        down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
        if backend == "packed_pretransposed_graphsafe_v31":
            moe_out = _active_moe_packed_pretransposed_graphsafe_v31(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "packed_pretransposed_graphsafe_v32_ordered":
            moe_out = _active_moe_packed_pretransposed_graphsafe_v32_ordered(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_nonatomic_out" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_nonatomic_out(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_nonatomic" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_nonatomic(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16_fused" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16_fused(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "grouped_per16" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_grouped_per16(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar_contract" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar_contract(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend == "cuda_scalar" and _layer_selected_for_native_cuda(cfg):
            moe_out = _active_moe_native_cuda_scalar(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
        elif backend in {
            "triton",
            "cuda_scalar",
            "cuda_scalar_contract",
            "grouped_per16",
            "grouped_per16_fused",
            "grouped_per16_nonatomic",
            "grouped_per16_nonatomic_out",
        }:
            gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
            if gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
                inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
            elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
                inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
            elif gateup_backend == "triton_fast_decode":
                inter_scratch = (
                    w.get("mlp.experts._active_inter_scratch")
                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
                    else None
                )
                gateup_fn = (
                    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
                    if _use_moe_effective_scale(w)
                    else nvfp4_grouped_gate_up_silu_fast_decode
                )
                inter = gateup_fn(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_effective_scale"]
                    if _use_moe_effective_scale(w)
                    else w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
                    out=inter_scratch,
                )
            elif gateup_backend == "triton":
                inter = nvfp4_grouped_gate_up_silu(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
                )
            elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
                inter = nvfp4_grouped_gate_up_silu(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
                )
            else:
                raise ValueError(
                    "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
                    "or 'split16_fp4', got "
                    f"{gateup_backend!r}"
                )
            if w4a8_mode == "full":
                inter = _fake_quant_fp8_activation(inter)
            if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
                moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
            elif down_backend == "triton":
                out_scratch = (
                    w.get("mlp.experts._active_out_scratch")
                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
                    else None
                )
                down_fn = (
                    nvfp4_grouped_down_weighted_sum_effective_scale
                    if _use_moe_effective_scale(w)
                    else nvfp4_grouped_down_weighted_sum
                )
                moe_out = down_fn(
                    inter,
                    expert_ids,
                    routing_weights,
                    w["mlp.experts._down_packed"],
                    w["mlp.experts._down_effective_scale"]
                    if _use_moe_effective_scale(w)
                    else w["mlp.experts._down_scale"],
                    w["mlp.experts._down_global_scale"],
                    block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
                    block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
                    num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
                    out=out_scratch,
                ).reshape_as(h_flat)
            else:

exec
/bin/zsh -lc "sed -n '620,940p' engine/full_forward.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
        attn_out, last_state, last_conv = prefill_linear_attn(h_norm, w)
        state.update_linear_attn_state(layer_idx, last_state, last_conv)
    else:  # full_attention
        attn_out, K, V = prefill_full_attn(h_norm, position_ids, w, cfg)
        state.update_full_attn_kv(layer_idx, K, V, position_start=0)
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    ffn_out = _ffn_forward(h_norm, w, cfg)
    return residual + ffn_out


@lru_cache(maxsize=None)
def _resolve_decode_moe_impl(impl: str):
    """Resolve the decode MoE implementation once per process.

    `_decode_layer` sits in the per-layer/per-token hot path. Re-importing and
    re-branching on `LYNN_MOE_IMPL` is tiny next to GPU kernels, but after P25
    the remaining gap between graph replay and serving is mostly orchestration.
    Keep the default env-driven behavior for legacy callers while allowing the
    resident runner to pass a fixed function pointer.
    """
    if impl == "optimized":
        from engine.moe_optimized import moe_forward_decode_optimized as _moe
    elif impl == "bmm":
        from engine.moe_optimized import moe_forward_decode_bmm as _moe
    elif impl == "indexed_bmm":
        from triton_kernels.moe_expert_ffn import moe_forward_decode_indexed_bmm as _moe
    elif impl == "triton":
        from triton_kernels.moe_expert_ffn import moe_forward_decode_triton as _moe
    elif impl == "packed_nvfp4":
        from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4 as _moe
    else:
        raise ValueError(f"Unknown LYNN_MOE_IMPL: {impl}")
    return _moe


def _decode_layer_k2(
    h_new_k2,
    position_ids_k2,
    layer_type,
    w,
    cfg,
    state,
    layer_idx,
    *,
    moe_fn=None,
    recurrent_backend: str | None = None,
    linear_state_update: str | None = None,
):
    """Forward one DecoderLayer over K=2 new tokens at once.

    Mirrors :func:`_decode_layer` (T=1) but uses the K=2 specialized decode
    primitives:

    * ``decode_full_attn_k2`` does a single SDPA over [B, 2, ...] with a
      prefix-causal mask (causal between the two new positions, full attention
      to all cached prefix positions).
    * ``decode_linear_attn_k2`` rolls out the GDN SSM two steps sequentially —
      there is no parallelism within an SSM layer, but the cost adds only
      ~16% to a full K=2 forward across the 30/40 layer split.
    * The MoE forward (``moe_forward_decode_optimized`` and variants) already
      accepts arbitrary T; we pass T=2 directly. The per-token expert routing
      naturally activates 1-2× more unique experts than T=1.

    Returns: ``h_new_k2 [B, 2, HIDDEN]`` after one decoder block.
    """
    from engine.incremental_decode import decode_full_attn, decode_full_attn_k2, decode_linear_attn_k2

    residual = h_new_k2
    h_norm = _rms_norm(h_new_k2, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        if recurrent_backend is None:
            recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
        if linear_state_update is None:
            linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign")

        if os.environ.get("LYNN_MTP_K2_LINEAR_ATTN_MODE", "k2") == "t1_loop":
            # Strict verifier fallback: two T=1 decode_linear_attn calls with
            # state.update_linear_attn_state interleaved between them — exactly
            # mirrors the sequential T=1 verifier (two decode_one_to_logits_and_hidden
            # invocations, each calling _decode_layer that issues
            # update_linear_attn_state). The default K=2 path
            # (decode_linear_attn_k2) does call T=1 twice internally but
            # threads (recurrent_state, conv_state) via local variables only;
            # state.recurrent_state[layer_idx] is updated once at the end of
            # the layer rather than between the two positions. M16 / M17 left
            # batched verify divergence at linear-attention layers (zero
            # advance: layer 5; event-5 K2-first: layer 32 pos1, conv-state
            # drift peaks layer 38). M18 (2026-05-20) bisects whether the
            # remaining drift originates here by toggling this knob
            # independently from LYNN_FULL_ATTN_K2_BACKEND.
            from engine.incremental_decode import decode_linear_attn
            out0, intermediate_state, intermediate_conv = decode_linear_attn(
                h_norm[:, 0:1, :].contiguous(), w,
                state.recurrent_state[layer_idx], state.conv_state[layer_idx],
                recurrent_backend=recurrent_backend,
            )
            if linear_state_update == "inplace":
                rec_target = state.recurrent_state[layer_idx]
                if rec_target.data_ptr() != intermediate_state.data_ptr():
                    rec_target.copy_(intermediate_state)
                conv_target = state.conv_state[layer_idx]
                if conv_target.data_ptr() != intermediate_conv.data_ptr():
                    conv_target.copy_(intermediate_conv)
            else:
                state.update_linear_attn_state(layer_idx, intermediate_state, intermediate_conv)
            out1, new_state, new_conv = decode_linear_attn(
                h_norm[:, 1:2, :].contiguous(), w,
                state.recurrent_state[layer_idx], state.conv_state[layer_idx],
                recurrent_backend=recurrent_backend,
            )
            attn_out = torch.cat([out0, out1], dim=1)
        else:
            with profile_section("k2_layer.linear_attention"):
                attn_out, new_state, new_conv = decode_linear_attn_k2(
                    h_norm, w,
                    state.recurrent_state[layer_idx],
                    state.conv_state[layer_idx],
                    recurrent_backend=recurrent_backend,
                )
        if linear_state_update == "inplace":
            recurrent_target = state.recurrent_state[layer_idx]
            if recurrent_target.data_ptr() != new_state.data_ptr():
                recurrent_target.copy_(new_state)
            conv_target = state.conv_state[layer_idx]
            if conv_target.data_ptr() != new_conv.data_ptr():
                conv_target.copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    else:
        K, V = state.kv_cache[layer_idx]
        full_attn_k2_backend = os.environ.get("LYNN_FULL_ATTN_K2_BACKEND", "t1_loop")
        if full_attn_k2_backend == "t1_loop":
            # Strict verifier fallback: reuse the exact T=1 full-attention
            # primitive twice so Q/K/V/O projection, RoPE, SDPA, and cache
            # writes match the sequential speculative verifier. Keep this as
            # the default until the true batched full-attention path is
            # bit-stable; opt into the fast path with LYNN_FULL_ATTN_K2_BACKEND=k2.
            attn0 = decode_full_attn(
                h_norm[:, 0:1, :].contiguous(),
                position_ids_k2[:, 0:1].contiguous(),
                w,
                cfg,
                K,
                V,
                cached_seq_len=state.seq_len,
            )
            attn1 = decode_full_attn(
                h_norm[:, 1:2, :].contiguous(),
                position_ids_k2[:, 1:2].contiguous(),
                w,
                cfg,
                K,
                V,
                cached_seq_len=state.seq_len + 1,
            )
            attn_out = torch.cat([attn0, attn1], dim=1)
        elif full_attn_k2_backend in {"k2", "rowwise_bridge", "rowwise_gate_bridge", "rowwise_kernel_bridge"}:
            with profile_section("k2_layer.full_attention"):
                attn_out = decode_full_attn_k2(
                    h_norm, position_ids_k2, w, cfg, K, V,
                    cached_seq_len=state.seq_len,
                )
        else:
            raise ValueError(f"Unknown LYNN_FULL_ATTN_K2_BACKEND: {full_attn_k2_backend}")
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    if not cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        # Dense FFN (e.g. Qwen3.5-9B): _dense_ffn_forward is M-general (it carries
        # the Spark FP8 fused gate/up + down path), so the whole K=2 block runs in
        # one call — no per-position MoE loop, no mlp.gate.weight router.
        with profile_section("k2_layer.dense_ffn"):
            return residual + _dense_ffn_forward(h_norm, w)
    # MoE for K=2: the packed_nvfp4 backend (Spark Config D default) is
    # T=1-only — its fused Triton kernel hard-codes h.shape[1] == 1. The
    # cheap fallback ``moe_fn=optimized`` (BF16 active-expert loop) works
    # at T>=2 but uses different kernels from the T=1 production path —
    # the resulting numerical drift broke MTP speculative accept (was 11%
    # with BF16 fallback vs 77% with sequential T=1 packed_nvfp4).
    #
    # The fix is to keep the same backend the runner is configured for
    # but call it ONCE PER TOKEN (T=1 invocations), then concat the
    # outputs. This trades a tiny per-position Python launch for backend
    # consistency, which is critical for the K=1 batched MTP verify path
    # (M9 — confirmed on Spark 2026-05-19).
    base_moe_fn = moe_fn if moe_fn is not None else _resolve_decode_moe_impl(
        os.environ.get("LYNN_MOE_IMPL", "optimized")
    )
    with profile_section("k2_layer.moe"):
        if h_norm.shape[1] == 1:
            moe_out = base_moe_fn(h_norm, w, cfg)
        elif os.environ.get("LYNN_MTP_VERIFY_SMALLM", "") == "1":
            from engine.moe_packed_nvfp4 import moe_forward_verify_smallm_nvfp4
            moe_out = moe_forward_verify_smallm_nvfp4(h_norm, w, cfg)
        elif os.environ.get("LYNN_MTP_K2_MOE_MODE", "") == "batched_optimized":
            # Diagnostic only. The production packed_nvfp4 decode MoE is T=1
            # exact; this one-call BF16 optimized path tests whether batching
            # MoE is even numerically acceptable before writing a real packed
            # T=2 kernel.
            with profile_section("k2_layer.moe_batched_optimized"):
                moe_out = _resolve_decode_moe_impl("optimized")(h_norm, w, cfg)
        else:
            # Per-position T=1 MoE for backend consistency.
            moe_per_token = [
                base_moe_fn(h_norm[:, t : t + 1, :].contiguous(), w, cfg)
                for t in range(h_norm.shape[1])
            ]
            moe_out = torch.cat(moe_per_token, dim=1)
    return residual + moe_out


def _decode_layer_block(
    h_new_block,
    position_ids_block,
    layer_type,
    w,
    cfg,
    state,
    layer_idx,
    *,
    moe_fn=None,
    recurrent_backend: str | None = None,
    linear_state_update: str | None = None,
):
    """Forward one DecoderLayer over an arbitrary verify block.

    This is an opt-in K=N counterpart to ``_decode_layer_k2`` for
    MTP/APEX-style speculative verification. ``K=1`` and ``K=2`` keep using the
    existing hot paths; larger blocks use prefix-causal full attention and a
    sequential recurrent rollout for linear-attention layers.
    """
    block_len = int(h_new_block.shape[1])
    if block_len == 1:
        return _decode_layer(
            h_new_block,
            position_ids_block[:, 0:1].contiguous(),
            layer_type,
            w,
            cfg,
            state,
            layer_idx,
            moe_fn=moe_fn,
            recurrent_backend=recurrent_backend,
            linear_state_update=linear_state_update,
        )
    if block_len == 2:
        return _decode_layer_k2(
            h_new_block,
            position_ids_block,
            layer_type,
            w,
            cfg,
            state,
            layer_idx,
            moe_fn=moe_fn,
            recurrent_backend=recurrent_backend,
            linear_state_update=linear_state_update,
        )

    from engine.incremental_decode import decode_full_attn, decode_full_attn_block, decode_linear_attn_block

    residual = h_new_block
    h_norm = _rms_norm(h_new_block, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        if recurrent_backend is None:
            recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
        if linear_state_update is None:
            linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign")
        attn_out, new_state, new_conv = decode_linear_attn_block(
            h_norm,
            w,
            state.recurrent_state[layer_idx],
            state.conv_state[layer_idx],
            recurrent_backend=recurrent_backend,
        )
        if linear_state_update == "inplace":
            recurrent_target = state.recurrent_state[layer_idx]
            if recurrent_target.data_ptr() != new_state.data_ptr():
                recurrent_target.copy_(new_state)
            conv_target = state.conv_state[layer_idx]
            if conv_target.data_ptr() != new_conv.data_ptr():
                conv_target.copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    else:
        K, V = state.kv_cache[layer_idx]
        if os.environ.get("LYNN_FULL_ATTN_BLOCK_BACKEND", "t1_loop") == "t1_loop":
            # Same correctness-first policy as K=2: block verify reuses the
            # canonical T=1 full-attention primitive by default. The true block
            # SDPA verifier remains opt-in with LYNN_FULL_ATTN_BLOCK_BACKEND=block.
            pieces = []
            for idx in range(block_len):
                pieces.append(
                    decode_full_attn(
                        h_norm[:, idx:idx + 1, :].contiguous(),
                        position_ids_block[:, idx:idx + 1].contiguous(),
                        w,
                        cfg,
                        K,
                        V,
                        cached_seq_len=state.seq_len + idx,
                    )
                )
            attn_out = torch.cat(pieces, dim=1)
        else:
            attn_out = decode_full_attn_block(
                h_norm,
                position_ids_block,
                w,
                cfg,
                K,
                V,
                cached_seq_len=state.seq_len,
            )
    h = residual + attn_out

    residual = h

codex
The fixed NVFP4 path already compresses active experts into two Triton launches per MoE layer: one gate/up/SwiGLU and one down/weighted-sum. The remaining per-layer launch overhead comes from router/top-k/softmax, shared expert dense FFN, residual/norm glue, and attention state updates.
exec
/bin/zsh -lc "sed -n '940,1060p' engine/full_forward.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    if not cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        # Dense FFN (e.g. Qwen3.5-9B): one M-general call for the whole block.
        return residual + _dense_ffn_forward(h_norm, w)
    base_moe_fn = moe_fn if moe_fn is not None else _resolve_decode_moe_impl(
        os.environ.get("LYNN_MOE_IMPL", "optimized")
    )
    if h_norm.shape[1] == 1:
        moe_out = base_moe_fn(h_norm, w, cfg)
    elif os.environ.get("LYNN_MTP_VERIFY_SMALLM", "") == "1":
        from engine.moe_packed_nvfp4 import moe_forward_verify_smallm_nvfp4
        moe_out = moe_forward_verify_smallm_nvfp4(h_norm, w, cfg)
    else:
        moe_out = torch.cat(
            [
                base_moe_fn(h_norm[:, t:t + 1, :].contiguous(), w, cfg)
                for t in range(h_norm.shape[1])
            ],
            dim=1,
        )
    return residual + moe_out


def _decode_layer(
    h_new,
    position_id,
    layer_type,
    w,
    cfg,
    state,
    layer_idx,
    *,
    moe_fn=None,
    recurrent_backend: str | None = None,
    linear_state_update: str | None = None,
):
    """Forward one DecoderLayer in decode mode (T=1) using cached state.

    LYNN_MOE_IMPL env var selects MoE implementation:
      optimized    Phase 3.2.1 active-experts loop (default)
      bmm          Phase 3.2.2 batched matmul
      indexed_bmm  Phase 3.2.2.5 pre-stacked grouped indexed_bmm
    """
    from engine.incremental_decode import decode_full_attn, decode_linear_attn

    residual = h_new
    h_norm = _rms_norm(h_new, w["input_layernorm.weight"])
    if layer_type == "linear_attention":
        if recurrent_backend is None:
            recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
        attn_out, new_state, new_conv = decode_linear_attn(
            h_norm, w,
            state.recurrent_state[layer_idx],
            state.conv_state[layer_idx],
            recurrent_backend=recurrent_backend,
        )
        if linear_state_update is None:
            linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign")
        if linear_state_update == "inplace":
            recurrent_target = state.recurrent_state[layer_idx]
            if recurrent_target.data_ptr() != new_state.data_ptr():
                recurrent_target.copy_(new_state)
            conv_target = state.conv_state[layer_idx]
            if conv_target.data_ptr() != new_conv.data_ptr():
                conv_target.copy_(new_conv)
        else:
            state.update_linear_attn_state(layer_idx, new_state, new_conv)
    else:
        K, V = state.kv_cache[layer_idx]
        attn_out = decode_full_attn(
            h_norm, position_id, w, cfg, K, V,
            cached_seq_len=state.seq_len,
        )
    h = residual + attn_out

    residual = h
    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
    if cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
        if moe_fn is None:
            moe_fn = _resolve_decode_moe_impl(os.environ.get("LYNN_MOE_IMPL", "optimized"))
        ffn_out = moe_fn(h_norm, w, cfg)
    else:
        ffn_out = _dense_ffn_forward(h_norm, w)
    return residual + ffn_out


def generate_incremental(model_dir, prompt, max_new=5, device="cuda",
                         dtype=torch.bfloat16, verbose=True, max_seq_len=2048):
    """Phase 3.1 incremental decode: prefill once, then 1-token-per-step decode.

    Compared to generate_greedy (brute-force), this should be ~10x faster
    on Spark for short generations and scale O(1) per token (vs O(T) brute).
    """
    from engine.loader import load_qwen36_layer
    from engine.inference_state import LynnInferenceState, infer_layer_types

    with open(Path(model_dir) / "config.json") as f:
        full_cfg = json.load(f)
    tc = full_cfg["text_config"]
    rope_p = tc.get("rope_parameters", {})
    num_experts = int(tc.get("num_experts", 0) or 0)
    layer_types = infer_layer_types(tc)
    cfg = {
        "hidden_size": tc["hidden_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["head_dim"],
        "num_experts": num_experts,
        "num_experts_per_tok": int(tc.get("num_experts_per_tok", 0) or 0),
        "is_moe": num_experts > 0,
        "rope_theta": rope_p.get("rope_theta", tc.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope_p.get("partial_rotary_factor", 1.0),
    }
    n_layers = tc["num_hidden_layers"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]


exec
/bin/zsh -lc "sed -n '1090,1270p' engine/incremental_decode.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
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
    beta = b.sigmoid()
    dt_bias = W("linear_attn.dt_bias")
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -W("linear_attn.A_log").float().exp()
    g = neg_exp_A_log * F.softplus(a.float() + dt_bias.float())

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
    if use_outconv_recurrent:
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

exec
/bin/zsh -lc "sed -n '1100,1180p' engine/moe_packed_nvfp4.py" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
            else:
                raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
        else:
            raise ValueError(
                "LYNN_NATIVE_ACTIVE_MOE_BACKEND must be 'triton', 'cuda_scalar', "
                "'cuda_scalar_contract', 'grouped_per16', 'grouped_per16_fused', "
                "'grouped_per16_nonatomic', 'grouped_per16_nonatomic_out', "
                "'packed_pretransposed_graphsafe_v31', "
                "'packed_pretransposed_graphsafe_v32_ordered', "
                f"got {backend!r}"
            )

    if os.environ.get("LYNN_MOE_PROFILE_SKIP_SHARED", "0") == "1":
        return moe_out.to(h.dtype).reshape_as(h)

    if "mlp.shared_expert.gate_proj.weight" in w:
        if (
            "mlp.shared_expert.gate_proj.weight.packed" in w
            and "mlp.shared_expert.up_proj.weight.packed" in w
            and "mlp.shared_expert.down_proj.weight.packed" in w
        ):
            gate_s, up_s = dual_scalar_bridge(
                h_flat[0],
                w["mlp.shared_expert.gate_proj.weight.packed"],
                w["mlp.shared_expert.up_proj.weight.packed"],
            )
            shared = w["mlp.shared_expert.down_proj.weight.packed"](
                (F.silu(gate_s) * up_s).to(h.dtype)
            ).reshape_as(h_flat)
        elif (
            _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True)
            and "mlp.shared_expert._gate_up_proj.weight" in w
        ):
            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        else:
            gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
            up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        moe_out = _finalize_shared_expert_output(h_flat, moe_out, shared, w)

    return moe_out.to(h.dtype).reshape_as(h)

exec
/bin/zsh -lc "grep -R \"def recurrent_gated_delta_fused_prepare\\|def linear_conv1d_update_triton\\|def qk_norm_rope_pair_triton\\|def rms_norm_gated_triton\\|def rowwise_prefix_attention\" -n triton_kernels engine | head -80" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
triton_kernels/rmsnorm_gated.py:46:def rms_norm_gated_triton(x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
triton_kernels/rowwise_attention.py:127:def rowwise_prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, block_n: int = 64) -> torch.Tensor:
triton_kernels/qk_norm_rope.py:155:def qk_norm_rope_pair_triton(
triton_kernels/gated_delta.py:214:def recurrent_gated_delta_fused_prepare(
triton_kernels/gated_delta.py:260:def recurrent_gated_delta_fused_prepare_gqa(
triton_kernels/gated_delta.py:304:def recurrent_gated_delta_fused_prepare_from_outconv_gqa(
triton_kernels/gated_delta.py:348:def recurrent_gated_delta_fused_prepare_from_outconv_ab_gqa(
triton_kernels/linear_conv.py:56:def linear_conv1d_update_triton(

codex
I’m pulling line-numbered snippets for the final citations. After that I’ll give the ranked fusion list with estimated launch savings per generated token, using the repository’s current fixed-path assumptions.
exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '899,1135p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   899	def moe_forward_decode_packed_nvfp4(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   900	    """Decode-only MoE using packed NVFP4 expert weights.
   901	
   902	    Router and shared expert stay on the existing BF16 resident path for now;
   903	    active routed experts consume packed NVFP4 weights directly. This is the
   904	    first production-shaped bridge from P10-H into the resident runner.
   905	    """
   906	    if h.shape[1] != 1:
   907	        raise NotImplementedError("packed NVFP4 MoE path is decode-only")
   908	    required = (
   909	        "mlp.experts._gate_up_packed",
   910	        "mlp.experts._gate_up_scale",
   911	        "mlp.experts._gate_up_global_scale",
   912	        "mlp.experts._down_packed",
   913	        "mlp.experts._down_scale",
   914	        "mlp.experts._down_global_scale",
   915	    )
   916	    missing = [key for key in required if key not in w]
   917	    if missing:
   918	        raise KeyError(f"packed NVFP4 MoE aliases missing: {missing}")
   919	
   920	    h_flat = h.reshape(-1, h.shape[-1])
   921	    if h_flat.shape[0] != 1:
   922	        raise NotImplementedError("packed NVFP4 MoE path currently supports batch=1")
   923	    if _env_bool("LYNN_MOE_FAST_FIXED", True):
   924	        if _env_bool("LYNN_ROUTER_TOPK_SORTED", False):
   925	            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_ROUTER_TOPK_SORTED=0")
   926	        if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
   927	            raise RuntimeError("LYNN_MOE_FAST_FIXED does not support LYNN_MOE_PROFILE_SKIP_ACTIVE")
   928	        if os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton") != "triton":
   929	            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton")
   930	        if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") not in {
   931	            "triton",
   932	            "triton_fast_decode",
   933	            "cuda_tile_inter",
   934	            "split16_fp4",
   935	        }:
   936	            raise RuntimeError(
   937	                "LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_GATEUP_BACKEND=triton, "
   938	                "triton_fast_decode, cuda_tile_inter, or split16_fp4"
   939	            )
   940	        if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") not in {"triton", "cuda_tile"}:
   941	            raise RuntimeError("LYNN_MOE_FAST_FIXED requires LYNN_NATIVE_DOWN_BACKEND=triton or cuda_tile")
   942	        if _env_bool("LYNN_MOE_TRITON_PREPARED", False):
   943	            if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") != "triton_fast_decode":
   944	                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
   945	            if os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton") != "triton":
   946	                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
   947	            if not _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False):
   948	                raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   949	        if (
   950	            _env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
   951	            _env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
   952	            _env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
   953	            _env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
   954	            _env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
   955	            _env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
   956	        ) not in {(8, 256, 8, 512, 4, 8), (8, 256, 4, 512, 4, 8)}:
   957	            raise RuntimeError("LYNN_MOE_FAST_FIXED only supports the R6000-best or Spark-best MoE kernel config")
   958	        return _moe_forward_decode_packed_nvfp4_fixed_triton(h, w, cfg)
   959	
   960	    router_logits = _router_linear(h_flat, w)
   961	    routing_weights, expert_indices = _router_topk(
   962	        router_logits,
   963	        int(cfg["num_experts_per_tok"]),
   964	        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
   965	        scratch_owner=w,
   966	    )
   967	    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
   968	    # Triton kernels consume int32 expert ids. Keep this as int32 once here so
   969	    # gate/up and down do not each pay a tiny per-layer dtype conversion.
   970	    expert_ids = expert_indices[0].to(torch.int32).contiguous()
   971	    topk_limit = os.environ.get("LYNN_MOE_PROFILE_TOPK_LIMIT")
   972	    if topk_limit:
   973	        limit = int(topk_limit)
   974	        if not (1 <= limit <= expert_ids.numel()):
   975	            raise ValueError(f"LYNN_MOE_PROFILE_TOPK_LIMIT must be in [1, {expert_ids.numel()}], got {limit}")
   976	        expert_ids = expert_ids[:limit].contiguous()
   977	        routing_weights = routing_weights[:limit].contiguous()
   978	        routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
   979	    hidden = h_flat[0]
   980	    w4a8_mode = _w4a8_fake_quant_mode()
   981	    if w4a8_mode in {"gateup", "full"}:
   982	        hidden = _fake_quant_fp8_activation(hidden)
   983	
   984	    if os.environ.get("LYNN_MOE_PROFILE_SKIP_ACTIVE", "0") == "1":
   985	        moe_out = torch.zeros_like(h_flat)
   986	    else:
   987	        backend = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND", "triton")
   988	        down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
   989	        if backend == "packed_pretransposed_graphsafe_v31":
   990	            moe_out = _active_moe_packed_pretransposed_graphsafe_v31(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
   991	        elif backend == "packed_pretransposed_graphsafe_v32_ordered":
   992	            moe_out = _active_moe_packed_pretransposed_graphsafe_v32_ordered(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
   993	        elif backend == "grouped_per16_nonatomic_out" and _layer_selected_for_native_cuda(cfg):
   994	            moe_out = _active_moe_native_grouped_per16_nonatomic_out(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
   995	        elif backend == "grouped_per16_nonatomic" and _layer_selected_for_native_cuda(cfg):
   996	            moe_out = _active_moe_native_grouped_per16_nonatomic(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
   997	        elif backend == "grouped_per16_fused" and _layer_selected_for_native_cuda(cfg):
   998	            moe_out = _active_moe_native_grouped_per16_fused(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
   999	        elif backend == "grouped_per16" and _layer_selected_for_native_cuda(cfg):
  1000	            moe_out = _active_moe_native_grouped_per16(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
  1001	        elif backend == "cuda_scalar_contract" and _layer_selected_for_native_cuda(cfg):
  1002	            moe_out = _active_moe_native_cuda_scalar_contract(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
  1003	        elif backend == "cuda_scalar" and _layer_selected_for_native_cuda(cfg):
  1004	            moe_out = _active_moe_native_cuda_scalar(hidden, expert_ids, routing_weights, w).reshape_as(h_flat)
  1005	        elif backend in {
  1006	            "triton",
  1007	            "cuda_scalar",
  1008	            "cuda_scalar_contract",
  1009	            "grouped_per16",
  1010	            "grouped_per16_fused",
  1011	            "grouped_per16_nonatomic",
  1012	            "grouped_per16_nonatomic_out",
  1013	        }:
  1014	            gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
  1015	            if gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
  1016	                inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
  1017	            elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
  1018	                inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
  1019	            elif gateup_backend == "triton_fast_decode":
  1020	                inter_scratch = (
  1021	                    w.get("mlp.experts._active_inter_scratch")
  1022	                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
  1023	                    else None
  1024	                )
  1025	                gateup_fn = (
  1026	                    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
  1027	                    if _use_moe_effective_scale(w)
  1028	                    else nvfp4_grouped_gate_up_silu_fast_decode
  1029	                )
  1030	                inter = gateup_fn(
  1031	                    hidden,
  1032	                    expert_ids,
  1033	                    w["mlp.experts._gate_up_packed"],
  1034	                    w["mlp.experts._gate_up_effective_scale"]
  1035	                    if _use_moe_effective_scale(w)
  1036	                    else w["mlp.experts._gate_up_scale"],
  1037	                    w["mlp.experts._gate_up_global_scale"],
  1038	                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
  1039	                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
  1040	                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
  1041	                    out=inter_scratch,
  1042	                )
  1043	            elif gateup_backend == "triton":
  1044	                inter = nvfp4_grouped_gate_up_silu(
  1045	                    hidden,
  1046	                    expert_ids,
  1047	                    w["mlp.experts._gate_up_packed"],
  1048	                    w["mlp.experts._gate_up_scale"],
  1049	                    w["mlp.experts._gate_up_global_scale"],
  1050	                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
  1051	                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
  1052	                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
  1053	                )
  1054	            elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
  1055	                inter = nvfp4_grouped_gate_up_silu(
  1056	                    hidden,
  1057	                    expert_ids,
  1058	                    w["mlp.experts._gate_up_packed"],
  1059	                    w["mlp.experts._gate_up_scale"],
  1060	                    w["mlp.experts._gate_up_global_scale"],
  1061	                    block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
  1062	                    block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
  1063	                    num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
  1064	                )
  1065	            else:
  1066	                raise ValueError(
  1067	                    "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
  1068	                    "or 'split16_fp4', got "
  1069	                    f"{gateup_backend!r}"
  1070	                )
  1071	            if w4a8_mode == "full":
  1072	                inter = _fake_quant_fp8_activation(inter)
  1073	            if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
  1074	                moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
  1075	            elif down_backend == "triton":
  1076	                out_scratch = (
  1077	                    w.get("mlp.experts._active_out_scratch")
  1078	                    if _env_bool("LYNN_MOE_ACTIVE_SCRATCH", False)
  1079	                    else None
  1080	                )
  1081	                down_fn = (
  1082	                    nvfp4_grouped_down_weighted_sum_effective_scale
  1083	                    if _use_moe_effective_scale(w)
  1084	                    else nvfp4_grouped_down_weighted_sum
  1085	                )
  1086	                moe_out = down_fn(
  1087	                    inter,
  1088	                    expert_ids,
  1089	                    routing_weights,
  1090	                    w["mlp.experts._down_packed"],
  1091	                    w["mlp.experts._down_effective_scale"]
  1092	                    if _use_moe_effective_scale(w)
  1093	                    else w["mlp.experts._down_scale"],
  1094	                    w["mlp.experts._down_global_scale"],
  1095	                    block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
  1096	                    block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
  1097	                    num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
  1098	                    out=out_scratch,
  1099	                ).reshape_as(h_flat)
  1100	            else:
  1101	                raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
  1102	        else:
  1103	            raise ValueError(
  1104	                "LYNN_NATIVE_ACTIVE_MOE_BACKEND must be 'triton', 'cuda_scalar', "
  1105	                "'cuda_scalar_contract', 'grouped_per16', 'grouped_per16_fused', "
  1106	                "'grouped_per16_nonatomic', 'grouped_per16_nonatomic_out', "
  1107	                "'packed_pretransposed_graphsafe_v31', "
  1108	                "'packed_pretransposed_graphsafe_v32_ordered', "
  1109	                f"got {backend!r}"
  1110	            )
  1111	
  1112	    if os.environ.get("LYNN_MOE_PROFILE_SKIP_SHARED", "0") == "1":
  1113	        return moe_out.to(h.dtype).reshape_as(h)
  1114	
  1115	    if "mlp.shared_expert.gate_proj.weight" in w:
  1116	        if (
  1117	            "mlp.shared_expert.gate_proj.weight.packed" in w
  1118	            and "mlp.shared_expert.up_proj.weight.packed" in w
  1119	            and "mlp.shared_expert.down_proj.weight.packed" in w
  1120	        ):
  1121	            gate_s, up_s = dual_scalar_bridge(
  1122	                h_flat[0],
  1123	                w["mlp.shared_expert.gate_proj.weight.packed"],
  1124	                w["mlp.shared_expert.up_proj.weight.packed"],
  1125	            )
  1126	            shared = w["mlp.shared_expert.down_proj.weight.packed"](
  1127	                (F.silu(gate_s) * up_s).to(h.dtype)
  1128	            ).reshape_as(h_flat)
  1129	        elif (
  1130	            _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True)
  1131	            and "mlp.shared_expert._gate_up_proj.weight" in w
  1132	        ):
  1133	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
  1134	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
  1135	            shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])

exec
/bin/zsh -lc "nl -ba engine/moe_packed_nvfp4.py | sed -n '680,830p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   680	def _moe_forward_decode_packed_nvfp4_fixed_triton(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
   681	    """Fixed-config production fast path for the current R6000 best profile."""
   682	    h_flat = h.reshape(-1, h.shape[-1])
   683	    router_logits = _router_linear(h_flat, w)
   684	    top_k = int(cfg["num_experts_per_tok"])
   685	    routing_weights, expert_indices = _router_topk(
   686	        router_logits,
   687	        top_k,
   688	        sorted=False,
   689	        scratch_owner=w,
   690	    )
   691	    routing_weights = _router_softmax(routing_weights, scratch_owner=w)
   692	    expert_ids = expert_indices[0].to(torch.int32).contiguous()
   693	    limit = _topk_limit_from_env(top_k)
   694	    if limit != top_k:
   695	        expert_ids = expert_ids[:limit].contiguous()
   696	        routing_weights = routing_weights[:limit].contiguous()
   697	        if _env_bool("LYNN_MOE_TOPK_RENORMALIZE", True):
   698	            routing_weights = routing_weights / routing_weights.sum().clamp_min(1e-20)
   699	    hidden = h_flat[0]
   700	    w4a8_mode = _w4a8_fake_quant_mode()
   701	    if w4a8_mode in {"gateup", "full"}:
   702	        hidden = _fake_quant_fp8_activation(hidden)
   703	    gateup_backend = os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton")
   704	    prepared_triton = _env_bool("LYNN_MOE_TRITON_PREPARED", False)
   705	    if prepared_triton:
   706	        if gateup_backend != "triton_fast_decode":
   707	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode")
   708	        inter_scratch = w.get("mlp.experts._active_inter_scratch")
   709	        if inter_scratch is None:
   710	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   711	        inter = nvfp4_grouped_gate_up_silu_fast_decode_prepared(
   712	            hidden,
   713	            expert_ids,
   714	            w["mlp.experts._gate_up_packed"],
   715	            w["mlp.experts._gate_up_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._gate_up_scale"],
   716	            w["mlp.experts._gate_up_global_scale"],
   717	            inter_scratch,
   718	            scale_effective=_use_moe_effective_scale(w),
   719	        )
   720	    elif gateup_backend == "split16_fp4" and _layer_selected_for_native_cuda(cfg):
   721	        inter = _gate_up_native_split16_fp4(hidden, expert_ids, w)
   722	    elif gateup_backend == "cuda_tile_inter" and _layer_selected_for_native_cuda(cfg):
   723	        inter = _gate_up_native_cuda_tile_inter(hidden, expert_ids, w)
   724	    elif gateup_backend == "triton_fast_decode":
   725	        gateup_fn = (
   726	            nvfp4_grouped_gate_up_silu_fast_decode_effective_scale
   727	            if _use_moe_effective_scale(w)
   728	            else nvfp4_grouped_gate_up_silu_fast_decode
   729	        )
   730	        inter = gateup_fn(
   731	            hidden,
   732	            expert_ids,
   733	            w["mlp.experts._gate_up_packed"],
   734	            w["mlp.experts._gate_up_effective_scale"]
   735	            if _use_moe_effective_scale(w)
   736	            else w["mlp.experts._gate_up_scale"],
   737	            w["mlp.experts._gate_up_global_scale"],
   738	            block_inter=8,
   739	            block_hidden=256,
   740	            num_warps=4,
   741	        )
   742	    elif gateup_backend == "triton":
   743	        inter = nvfp4_grouped_gate_up_silu(
   744	            hidden,
   745	            expert_ids,
   746	            w["mlp.experts._gate_up_packed"],
   747	            w["mlp.experts._gate_up_scale"],
   748	            w["mlp.experts._gate_up_global_scale"],
   749	            block_inter=8,
   750	            block_hidden=256,
   751	            num_warps=4,
   752	        )
   753	    elif gateup_backend in {"cuda_tile_inter", "split16_fp4"}:
   754	        inter = nvfp4_grouped_gate_up_silu(
   755	            hidden,
   756	            expert_ids,
   757	            w["mlp.experts._gate_up_packed"],
   758	            w["mlp.experts._gate_up_scale"],
   759	            w["mlp.experts._gate_up_global_scale"],
   760	            block_inter=8,
   761	            block_hidden=256,
   762	            num_warps=4,
   763	        )
   764	    else:
   765	        raise ValueError(
   766	            "LYNN_NATIVE_GATEUP_BACKEND must be 'triton', 'triton_fast_decode', 'cuda_tile_inter', "
   767	            "or 'split16_fp4', got "
   768	            f"{gateup_backend!r}"
   769	        )
   770	    if w4a8_mode == "full":
   771	        inter = _fake_quant_fp8_activation(inter)
   772	    down_backend = os.environ.get("LYNN_NATIVE_DOWN_BACKEND", "triton")
   773	    if down_backend == "cuda_tile" and _layer_selected_for_native_cuda(cfg):
   774	        moe_out = _down_weighted_sum_native_cuda_tile(inter, expert_ids, routing_weights, w).reshape_as(h_flat)
   775	    elif prepared_triton:
   776	        if down_backend != "triton":
   777	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_NATIVE_DOWN_BACKEND=triton")
   778	        out_scratch = w.get("mlp.experts._active_out_scratch")
   779	        if out_scratch is None:
   780	            raise RuntimeError("LYNN_MOE_TRITON_PREPARED requires LYNN_MOE_ACTIVE_SCRATCH=1")
   781	        moe_out = nvfp4_grouped_down_weighted_sum_prepared(
   782	            inter,
   783	            expert_ids,
   784	            routing_weights,
   785	            w["mlp.experts._down_packed"],
   786	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   787	            w["mlp.experts._down_global_scale"],
   788	            out_scratch,
   789	            scale_effective=_use_moe_effective_scale(w),
   790	        ).reshape_as(h_flat)
   791	    elif down_backend == "triton":
   792	        down_fn = (
   793	            nvfp4_grouped_down_weighted_sum_effective_scale
   794	            if _use_moe_effective_scale(w)
   795	            else nvfp4_grouped_down_weighted_sum
   796	        )
   797	        moe_out = down_fn(
   798	            inter,
   799	            expert_ids,
   800	            routing_weights,
   801	            w["mlp.experts._down_packed"],
   802	            w["mlp.experts._down_effective_scale"] if _use_moe_effective_scale(w) else w["mlp.experts._down_scale"],
   803	            w["mlp.experts._down_global_scale"],
   804	            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
   805	            block_inter=512,
   806	            num_warps=8,
   807	        ).reshape_as(h_flat)
   808	    elif down_backend == "cuda_tile":
   809	        moe_out = nvfp4_grouped_down_weighted_sum(
   810	            inter,
   811	            expert_ids,
   812	            routing_weights,
   813	            w["mlp.experts._down_packed"],
   814	            w["mlp.experts._down_scale"],
   815	            w["mlp.experts._down_global_scale"],
   816	            block_hidden=8,
   817	            block_inter=512,
   818	            num_warps=8,
   819	        ).reshape_as(h_flat)
   820	    else:
   821	        raise ValueError("LYNN_NATIVE_DOWN_BACKEND must be 'triton' or 'cuda_tile', got " f"{down_backend!r}")
   822	
   823	    if _skip_shared_from_env():
   824	        return moe_out.to(h.dtype).reshape_as(h)
   825	
   826	    if "mlp.shared_expert.gate_proj.weight" in w:
   827	        if "mlp.shared_expert._gate_up_proj.weight" in w:
   828	            gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
   829	            gate_s, up_s = gate_up_s.chunk(2, dim=-1)
   830	        else:

exec
/bin/zsh -lc "nl -ba triton_kernels/nvfp4_moe.py | sed -n '760,1095p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   760	def nvfp4_grouped_gate_up_silu_fast_decode(
   761	    x: torch.Tensor,
   762	    expert_ids: torch.Tensor,
   763	    gate_up_packed: torch.Tensor,
   764	    gate_up_scale: torch.Tensor,
   765	    gate_up_global_scale: torch.Tensor,
   766	    *,
   767	    block_inter: int = 8,
   768	    block_hidden: int = 256,
   769	    num_warps: int = 4,
   770	    out: torch.Tensor | None = None,
   771	) -> torch.Tensor:
   772	    """P53 lightweight probe: same kernel shape, faster E2M1 expression only."""
   773	    _require_triton()
   774	    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
   775	        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
   776	    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
   777	        raise ValueError(
   778	            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
   779	        )
   780	    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
   781	    if out is None:
   782	        inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
   783	    else:
   784	        if out.ndim != 2 or out.shape[0] < expert_ids.numel() or out.shape[1] != INTERMEDIATE_SIZE:
   785	            raise ValueError(
   786	                f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)} "
   787	                f"for top_k={expert_ids.numel()}"
   788	            )
   789	        if out.device != x.device or out.dtype != torch.bfloat16:
   790	            raise ValueError("out must be a bfloat16 tensor on the same device as x")
   791	        inter = out[: expert_ids.numel()]
   792	    grid = (expert_ids.numel(), triton.cdiv(INTERMEDIATE_SIZE, block_inter))
   793	    _grouped_gate_up_silu_fast_decode_kernel[grid](
   794	        x.contiguous(),
   795	        expert_ids,
   796	        gate_up_packed.contiguous(),
   797	        gate_up_scale.contiguous(),
   798	        gate_up_global_scale.to(device=x.device).contiguous(),
   799	        inter,
   800	        gate_up_packed.stride(0),
   801	        gate_up_packed.stride(1),
   802	        gate_up_packed.stride(2),
   803	        gate_up_scale.stride(0),
   804	        gate_up_scale.stride(1),
   805	        gate_up_scale.stride(2),
   806	        inter.stride(0),
   807	        inter.stride(1),
   808	        HIDDEN=HIDDEN_SIZE,
   809	        INTERMEDIATE=INTERMEDIATE_SIZE,
   810	        BLOCK_INTER=block_inter,
   811	        BLOCK_HIDDEN=block_hidden,
   812	        SCALE_EFFECTIVE=False,
   813	        num_warps=num_warps,
   814	    )
   815	    return inter
   816	
   817	
   818	def nvfp4_grouped_gate_up_silu_fast_decode_prepared(
   819	    x: torch.Tensor,
   820	    expert_ids: torch.Tensor,
   821	    gate_up_packed: torch.Tensor,
   822	    gate_up_scale: torch.Tensor,
   823	    gate_up_global_scale: torch.Tensor,
   824	    out: torch.Tensor,
   825	    *,
   826	    scale_effective: bool = False,
   827	) -> torch.Tensor:
   828	    """Prepared-shape gate/up entrypoint for the resident decode hot path.
   829	
   830	    The public wrappers defensively cast and contiguous-copy inputs. The fixed
   831	    resident W4A16 path already prepares these tensors. This exact wrapper keeps
   832	    the same Triton kernel and constants, but avoids redundant boundary ops.
   833	    """
   834	    _require_triton()
   835	    top_k = expert_ids.numel()
   836	    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
   837	        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
   838	    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != x.device:
   839	        raise ValueError("expert_ids must be a contiguous int32 tensor on x.device")
   840	    if not expert_ids.is_contiguous():
   841	        raise ValueError("expert_ids must be contiguous")
   842	    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
   843	        raise ValueError(
   844	            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
   845	        )
   846	    if not (
   847	        x.is_contiguous()
   848	        and gate_up_packed.is_contiguous()
   849	        and gate_up_scale.is_contiguous()
   850	        and gate_up_global_scale.is_contiguous()
   851	    ):
   852	        raise ValueError("prepared gate/up tensors must be contiguous")
   853	    if out.ndim != 2 or out.shape[0] < top_k or out.shape[1] != INTERMEDIATE_SIZE:
   854	        raise ValueError(f"out must be at least [top_k, {INTERMEDIATE_SIZE}], got {tuple(out.shape)}")
   855	    if out.device != x.device or out.dtype != torch.bfloat16:
   856	        raise ValueError("out must be a bfloat16 tensor on x.device")
   857	    inter = out[:top_k]
   858	    grid = (top_k, triton.cdiv(INTERMEDIATE_SIZE, 8))
   859	    _grouped_gate_up_silu_fast_decode_kernel[grid](
   860	        x,
   861	        expert_ids,
   862	        gate_up_packed,
   863	        gate_up_scale,
   864	        gate_up_global_scale,
   865	        inter,
   866	        gate_up_packed.stride(0),
   867	        gate_up_packed.stride(1),
   868	        gate_up_packed.stride(2),
   869	        gate_up_scale.stride(0),
   870	        gate_up_scale.stride(1),
   871	        gate_up_scale.stride(2),
   872	        inter.stride(0),
   873	        inter.stride(1),
   874	        HIDDEN=HIDDEN_SIZE,
   875	        INTERMEDIATE=INTERMEDIATE_SIZE,
   876	        BLOCK_INTER=8,
   877	        BLOCK_HIDDEN=256,
   878	        SCALE_EFFECTIVE=scale_effective,
   879	        num_warps=4,
   880	    )
   881	    return inter
   882	
   883	
   884	def nvfp4_grouped_gate_up_silu_merged_topk(
   885	    x: torch.Tensor,
   886	    expert_ids: torch.Tensor,
   887	    gate_up_packed: torch.Tensor,
   888	    gate_up_scale: torch.Tensor,
   889	    gate_up_global_scale: torch.Tensor,
   890	    *,
   891	    block_inter: int = 8,
   892	    block_hidden: int = 256,
   893	    num_warps: int = 4,
   894	) -> torch.Tensor:
   895	    """Gate/up variant with one program per inter block and a top-k inner loop.
   896	
   897	    This is an opt-in P26 probe, not the production default. It tests whether
   898	    reducing kernel program count from `[top_k, inter_blocks]` to
   899	    `[inter_blocks]` helps launch/scheduling overhead while preserving the
   900	    current per-16 scalar contract.
   901	    """
   902	    _require_triton()
   903	    if x.ndim != 1 or x.numel() != HIDDEN_SIZE:
   904	        raise ValueError(f"x must be [2048], got {tuple(x.shape)}")
   905	    if gate_up_packed.ndim != 3 or gate_up_scale.ndim != 3:
   906	        raise ValueError(
   907	            f"expected grouped 3D tensors, got packed={tuple(gate_up_packed.shape)} scale={tuple(gate_up_scale.shape)}"
   908	        )
   909	    expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
   910	    inter = torch.empty((expert_ids.numel(), INTERMEDIATE_SIZE), device=x.device, dtype=torch.bfloat16)
   911	    grid = (triton.cdiv(INTERMEDIATE_SIZE, block_inter),)
   912	    _grouped_gate_up_silu_merged_topk_kernel[grid](
   913	        x.contiguous(),
   914	        expert_ids,
   915	        gate_up_packed.contiguous(),
   916	        gate_up_scale.contiguous(),
   917	        gate_up_global_scale.to(device=x.device).contiguous(),
   918	        inter,
   919	        gate_up_packed.stride(0),
   920	        gate_up_packed.stride(1),
   921	        gate_up_packed.stride(2),
   922	        gate_up_scale.stride(0),
   923	        gate_up_scale.stride(1),
   924	        gate_up_scale.stride(2),
   925	        inter.stride(1),
   926	        HIDDEN=HIDDEN_SIZE,
   927	        INTERMEDIATE=INTERMEDIATE_SIZE,
   928	        TOP_K=expert_ids.numel(),
   929	        BLOCK_INTER=block_inter,
   930	        BLOCK_HIDDEN=block_hidden,
   931	        num_warps=num_warps,
   932	    )
   933	    return inter
   934	
   935	
   936	def nvfp4_grouped_down_weighted_sum(
   937	    inter: torch.Tensor,
   938	    expert_ids: torch.Tensor,
   939	    routing_weights: torch.Tensor,
   940	    down_packed: torch.Tensor,
   941	    down_scale: torch.Tensor,
   942	    down_global_scale: torch.Tensor,
   943	    *,
   944	    block_hidden: int = 16,
   945	    block_inter: int = 128,
   946	    num_warps: int = 4,
   947	    out: torch.Tensor | None = None,
   948	) -> torch.Tensor:
   949	    """Compute weighted top-k down projection from grouped packed NVFP4 weights."""
   950	    _require_triton()
   951	    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
   952	        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
   953	    if down_packed.ndim != 3 or down_scale.ndim != 3:
   954	        raise ValueError(
   955	            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
   956	        )
   957	    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
   958	    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
   959	    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
   960	        raise ValueError("expert_ids/routing_weights must match inter top_k")
   961	    if out is None:
   962	        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
   963	    else:
   964	        if out.shape != (HIDDEN_SIZE,):
   965	            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
   966	        if out.device != inter.device or out.dtype != torch.bfloat16:
   967	            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
   968	    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
   969	    _grouped_down_weighted_sum_kernel[grid](
   970	        inter.contiguous(),
   971	        expert_ids,
   972	        routing_weights,
   973	        down_packed.contiguous(),
   974	        down_scale.contiguous(),
   975	        down_global_scale.to(device=inter.device).contiguous(),
   976	        out,
   977	        down_packed.stride(0),
   978	        down_packed.stride(1),
   979	        down_packed.stride(2),
   980	        down_scale.stride(0),
   981	        down_scale.stride(1),
   982	        down_scale.stride(2),
   983	        inter.stride(0),
   984	        inter.stride(1),
   985	        TOP_K=inter.shape[0],
   986	        HIDDEN=HIDDEN_SIZE,
   987	        INTERMEDIATE=INTERMEDIATE_SIZE,
   988	        BLOCK_HIDDEN=block_hidden,
   989	        BLOCK_INTER=block_inter,
   990	        SCALE_EFFECTIVE=False,
   991	        num_warps=num_warps,
   992	    )
   993	    return out
   994	
   995	
   996	def nvfp4_grouped_down_weighted_sum_effective_scale(
   997	    inter: torch.Tensor,
   998	    expert_ids: torch.Tensor,
   999	    routing_weights: torch.Tensor,
  1000	    down_packed: torch.Tensor,
  1001	    down_effective_scale: torch.Tensor,
  1002	    down_global_scale: torch.Tensor,
  1003	    *,
  1004	    block_hidden: int = 16,
  1005	    block_inter: int = 128,
  1006	    num_warps: int = 4,
  1007	    out: torch.Tensor | None = None,
  1008	) -> torch.Tensor:
  1009	    """Down projection consuming precomputed `scale / global_scale`."""
  1010	    _require_triton()
  1011	    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
  1012	        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
  1013	    if down_packed.ndim != 3 or down_effective_scale.ndim != 3:
  1014	        raise ValueError(
  1015	            "expected grouped 3D tensors, got "
  1016	            f"packed={tuple(down_packed.shape)} scale={tuple(down_effective_scale.shape)}"
  1017	        )
  1018	    expert_ids = expert_ids.to(device=inter.device, dtype=torch.int32).contiguous()
  1019	    routing_weights = routing_weights.to(device=inter.device, dtype=torch.float32).contiguous()
  1020	    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
  1021	        raise ValueError("expert_ids/routing_weights must match inter top_k")
  1022	    if out is None:
  1023	        out = torch.empty((HIDDEN_SIZE,), device=inter.device, dtype=torch.bfloat16)
  1024	    else:
  1025	        if out.shape != (HIDDEN_SIZE,):
  1026	            raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
  1027	        if out.device != inter.device or out.dtype != torch.bfloat16:
  1028	            raise ValueError("out must be a bfloat16 tensor on the same device as inter")
  1029	    grid = (triton.cdiv(HIDDEN_SIZE, block_hidden),)
  1030	    _grouped_down_weighted_sum_kernel[grid](
  1031	        inter.contiguous(),
  1032	        expert_ids,
  1033	        routing_weights,
  1034	        down_packed.contiguous(),
  1035	        down_effective_scale.contiguous(),
  1036	        down_global_scale.to(device=inter.device).contiguous(),
  1037	        out,
  1038	        down_packed.stride(0),
  1039	        down_packed.stride(1),
  1040	        down_packed.stride(2),
  1041	        down_effective_scale.stride(0),
  1042	        down_effective_scale.stride(1),
  1043	        down_effective_scale.stride(2),
  1044	        inter.stride(0),
  1045	        inter.stride(1),
  1046	        TOP_K=inter.shape[0],
  1047	        HIDDEN=HIDDEN_SIZE,
  1048	        INTERMEDIATE=INTERMEDIATE_SIZE,
  1049	        BLOCK_HIDDEN=block_hidden,
  1050	        BLOCK_INTER=block_inter,
  1051	        SCALE_EFFECTIVE=True,
  1052	        num_warps=num_warps,
  1053	    )
  1054	    return out
  1055	
  1056	
  1057	def nvfp4_grouped_down_weighted_sum_prepared(
  1058	    inter: torch.Tensor,
  1059	    expert_ids: torch.Tensor,
  1060	    routing_weights: torch.Tensor,
  1061	    down_packed: torch.Tensor,
  1062	    down_scale: torch.Tensor,
  1063	    down_global_scale: torch.Tensor,
  1064	    out: torch.Tensor,
  1065	    *,
  1066	    scale_effective: bool = False,
  1067	) -> torch.Tensor:
  1068	    """Prepared-shape down entrypoint for the resident decode hot path."""
  1069	    _require_triton()
  1070	    if inter.ndim != 2 or inter.shape[1] != INTERMEDIATE_SIZE:
  1071	        raise ValueError(f"inter must be [top_k, 512], got {tuple(inter.shape)}")
  1072	    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32 or expert_ids.device != inter.device:
  1073	        raise ValueError("expert_ids must be a contiguous int32 tensor on inter.device")
  1074	    if routing_weights.ndim != 1 or routing_weights.dtype != torch.float32 or routing_weights.device != inter.device:
  1075	        raise ValueError("routing_weights must be a contiguous float32 tensor on inter.device")
  1076	    if not expert_ids.is_contiguous() or not routing_weights.is_contiguous():
  1077	        raise ValueError("expert_ids and routing_weights must be contiguous")
  1078	    if expert_ids.numel() != inter.shape[0] or routing_weights.numel() != inter.shape[0]:
  1079	        raise ValueError("expert_ids/routing_weights must match inter top_k")
  1080	    if down_packed.ndim != 3 or down_scale.ndim != 3:
  1081	        raise ValueError(
  1082	            f"expected grouped 3D tensors, got packed={tuple(down_packed.shape)} scale={tuple(down_scale.shape)}"
  1083	        )
  1084	    if not (inter.is_contiguous() and down_packed.is_contiguous() and down_scale.is_contiguous() and down_global_scale.is_contiguous()):
  1085	        raise ValueError("prepared down tensors must be contiguous")
  1086	    if out.shape != (HIDDEN_SIZE,):
  1087	        raise ValueError(f"out must be [{HIDDEN_SIZE}], got {tuple(out.shape)}")
  1088	    if out.device != inter.device or out.dtype != torch.bfloat16:
  1089	        raise ValueError("out must be a bfloat16 tensor on inter.device")
  1090	    grid = (triton.cdiv(HIDDEN_SIZE, 8),)
  1091	    _grouped_down_weighted_sum_kernel[grid](
  1092	        inter,
  1093	        expert_ids,
  1094	        routing_weights,
  1095	        down_packed,

exec
/bin/zsh -lc "nl -ba engine/incremental_decode.py | sed -n '1090,1215p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
  1090	def decode_linear_attn(h_new, w, recurrent_state, conv_state, *, recurrent_backend: str = "torch"):
  1091	    """Decode 1 new token using cached recurrent_state + conv_state.
  1092	
  1093	    h_new: [B, 1, HIDDEN] (post input_layernorm)
  1094	    recurrent_state: [B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM] FP32
  1095	    conv_state: [B, CONV_DIM, CONV_KERNEL-1]
  1096	
  1097	    Returns:
  1098	        out: [B, 1, HIDDEN]
  1099	        new_recurrent_state: [B, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM] FP32
  1100	        new_conv_state: [B, CONV_DIM, CONV_KERNEL-1]
  1101	    """
  1102	    B = h_new.shape[0]
  1103	
  1104	    def W(k):
  1105	        return _decode_weight(w, k)
  1106	
  1107	    # 1. Input projections on h_new.
  1108	    #
  1109	    # P6-O opt-in path: fuse qkv/z/b/a input projections into one GEMM.
  1110	    # Decode is dominated by many tiny launches across 30 linear-attn layers;
  1111	    # one larger matmul is usually better than four independent launches.
  1112	    use_packed_linear = (
  1113	        os.environ.get("LYNN_PACKED_DECODE", "0") == "1"
  1114	        or os.environ.get("LYNN_PACKED_DECODE_LINEAR_ATTN", "0") == "1"
  1115	    )
  1116	    if not use_packed_linear and "linear_attn._in_proj_qkv_z_b_a.weight" in w:
  1117	        proj_all = _linear(h_new, W("linear_attn._in_proj_qkv_z_b_a.weight"))
  1118	        mixed_new, z, b, a = torch.split(
  1119	            proj_all,
  1120	            [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
  1121	            dim=-1,
  1122	        )
  1123	    else:
  1124	        mixed_new = _linear(h_new, W("linear_attn.in_proj_qkv.weight"))    # [B, 1, conv_dim]
  1125	        z = _linear(h_new, W("linear_attn.in_proj_z.weight")).reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
  1126	        b = _linear(h_new, W("linear_attn.in_proj_b.weight"))
  1127	        a = _linear(h_new, W("linear_attn.in_proj_a.weight"))
  1128	    mixed_new = mixed_new.transpose(1, 2)                              # [B, conv_dim, 1]
  1129	
  1130	    # 2. Causal conv1d update: prepend conv_state to mixed_new, run conv, take last 1 output.
  1131	    # Triton opt-in can fuse cat + depthwise conv + silu + state shift.
  1132	    out_conv, new_conv_state = _linear_conv_update_decode(
  1133	        mixed_new,
  1134	        conv_state,
  1135	        W("linear_attn.conv1d.weight"),
  1136	    )
  1137	
  1138	    # 3. z, beta, g (using h_new)
  1139	    z = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
  1140	    beta = b.sigmoid()
  1141	    dt_bias = W("linear_attn.dt_bias")
  1142	    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
  1143	    if neg_exp_A_log is None:
  1144	        neg_exp_A_log = -W("linear_attn.A_log").float().exp()
  1145	    g = neg_exp_A_log * F.softplus(a.float() + dt_bias.float())
  1146	
  1147	    # 4. q, k repeat by V_PER_K. P10-D can avoid this materialization for the
  1148	    # Triton recurrent path by reading q/k as grouped-query heads directly.
  1149	    use_gqa_recurrent = (
  1150	        recurrent_backend == "triton_fused_prepare"
  1151	        and V_PER_K > 1
  1152	        and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"
  1153	    )
  1154	
  1155	    use_outconv_recurrent = (
  1156	        use_gqa_recurrent
  1157	        and os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV", "0") == "1"
  1158	    )
  1159	    if use_outconv_recurrent:
  1160	        if recurrent_gated_delta_fused_prepare_from_outconv_gqa is None:
  1161	            raise RuntimeError("outconv recurrent requested but Triton kernel is unavailable")
  1162	        core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare_from_outconv_gqa(
  1163	            out_conv, g, beta, recurrent_state
  1164	        )
  1165	    else:
  1166	        q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
  1167	        q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
  1168	        k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
  1169	        v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
  1170	        if V_PER_K > 1 and not use_gqa_recurrent:
  1171	            q = q.repeat_interleave(V_PER_K, dim=2)
  1172	            k = k.repeat_interleave(V_PER_K, dim=2)
  1173	
  1174	    # 5. recurrent gated delta rule (single-step)
  1175	    if use_outconv_recurrent:
  1176	        pass
  1177	    elif recurrent_backend == "torch":
  1178	        core_attn_out, new_recurrent_state = _recurrent_gated_delta_rule(
  1179	            q, k, v, g, beta, recurrent_state
  1180	        )
  1181	    elif recurrent_backend == "triton_fused_prepare":
  1182	        if recurrent_gated_delta_fused_prepare is None:
  1183	            raise RuntimeError("triton_fused_prepare requested but triton kernel is unavailable")
  1184	        if use_gqa_recurrent:
  1185	            if recurrent_gated_delta_fused_prepare_gqa is None:
  1186	                raise RuntimeError("GQA recurrent requested but triton kernel is unavailable")
  1187	            core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare_gqa(
  1188	                q, k, v, g, beta, recurrent_state
  1189	            )
  1190	        else:
  1191	            core_attn_out, new_recurrent_state = recurrent_gated_delta_fused_prepare(
  1192	                q, k, v, g, beta, recurrent_state
  1193	            )
  1194	    else:
  1195	        raise ValueError(f"unknown recurrent_backend={recurrent_backend!r}")
  1196	
  1197	    # 7. RMSNormGated
  1198	    norm_w = W("linear_attn.norm.weight")
  1199	    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
  1200	    flat_z = z.reshape(-1, HEAD_V_DIM)
  1201	    flat_y = _rms_norm_gated_decode(flat_x, norm_w, flat_z)
  1202	    core_attn_out = flat_y.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
  1203	
  1204	    # 8. out_proj
  1205	    out = _linear(core_attn_out, W("linear_attn.out_proj.weight"))
  1206	    return out, new_recurrent_state, new_conv_state
  1207	
  1208	
  1209	# ============================================================================
  1210	# Helper: chunk_gated_delta_rule with output_final_state=True
  1211	# (the version in qwen36_linear_attn_block.py discards last_state — we need it here)
  1212	# ============================================================================
  1213	
  1214	def _chunk_gated_delta_with_state(query, key, value, g, beta, chunk_size=64,
  1215	                                   use_qk_l2norm=True, initial_state=None,

exec
/bin/zsh -lc "nl -ba engine/incremental_decode.py | sed -n '382,515p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   382	def decode_full_attn(h_new, new_position_id, w, cfg, K_cache_full, V_cache_full,
   383	                    cached_seq_len: int):
   384	    """Decode 1 new token using cached K/V.
   385	
   386	    h_new: [B, 1, HIDDEN]  (post input_layernorm)
   387	    new_position_id: scalar int (position of the new token, 0-indexed)
   388	    K_cache_full, V_cache_full: [B, NUM_KV_HEADS, max_T, HEAD_DIM] (pre-allocated full size)
   389	    cached_seq_len: how many positions are already populated in cache (= new_position_id for fresh decode)
   390	
   391	    Side effect: writes new K/V into K_cache_full[:, :, cached_seq_len:cached_seq_len+1, :] etc.
   392	
   393	    Returns: attn_out [B, 1, HIDDEN]
   394	    """
   395	    B, _, _ = h_new.shape
   396	    H_Q = cfg["num_attention_heads"]
   397	    H_KV = cfg["num_key_value_heads"]
   398	    head_dim = cfg["head_dim"]
   399	    rope_theta = cfg["rope_theta"]
   400	    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
   401	
   402	    # 1. Q/K/V projection on the single new token
   403	    with profile_section("full_attn_t1.qkv"):
   404	        if os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1" and "self_attn._qkv_proj.weight" in w:
   405	            q_out = int(w["self_attn.q_proj.weight"].shape[0])
   406	            k_out = int(w["self_attn.k_proj.weight"].shape[0])
   407	            v_out = int(w["self_attn.v_proj.weight"].shape[0])
   408	            qkv = _linear(h_new, w["self_attn._qkv_proj.weight"])
   409	            q_full, k_new, v_new = qkv.split((q_out, k_out, v_out), dim=-1)
   410	        else:
   411	            q_full = _linear(h_new, _decode_weight(w, "self_attn.q_proj.weight"))
   412	            k_new = _linear(h_new, _decode_weight(w, "self_attn.k_proj.weight"))
   413	            v_new = _linear(h_new, _decode_weight(w, "self_attn.v_proj.weight"))
   414	
   415	        q_full_view = q_full.view(B, 1, H_Q, head_dim * 2)
   416	        q, gate = q_full_view.chunk(2, dim=-1)
   417	        q = q.transpose(1, 2)
   418	        gate = gate.transpose(1, 2)
   419	        k_new = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
   420	        v_new = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
   421	
   422	    # 2+3. q/k norm + RoPE on the new position.
   423	    # CUDA graph capture cannot include fresh torch.tensor allocations, so
   424	    # benchmark/serving paths may pass a preallocated [[position]] tensor.
   425	    if torch.is_tensor(new_position_id):
   426	        pos_tensor = new_position_id
   427	    else:
   428	        pos_tensor = torch.tensor([[new_position_id]], device=h_new.device, dtype=torch.long)
   429	    rope_builder = (
   430	        _build_rope_cos_sin_cached
   431	        if os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE", "0") == "1"
   432	        else _build_rope_cos_sin
   433	    )
   434	    with profile_section("full_attn_t1.rope"):
   435	        cos, sin = rope_builder(pos_tensor, rotary_dim, rope_theta, h_new.device, h_new.dtype)
   436	        q, k_new = _qk_norm_rope_pair_decode(
   437	            q,
   438	            k_new,
   439	            w["self_attn.q_norm.weight"],
   440	            w["self_attn.k_norm.weight"],
   441	            cos,
   442	            sin,
   443	            rotary_dim,
   444	        )
   445	
   446	    # 4. Append to cache at position cached_seq_len
   447	    #    M3 fixed-shape (graph-replayable) path: write K/V via index_copy_ (the
   448	    #    write position is read from pos_tensor, not baked as a Python-int slice)
   449	    #    and attend over the FULL fixed-length cache with a position mask. No
   450	    #    tensor shape then depends on cached_seq_len, so the whole decode can be
   451	    #    captured once and replayed. Bit-exact with the variable-slice path:
   452	    #    masked (future/stale) key positions get zero attention weight.
   453	    fixed_shape = os.environ.get("LYNN_FULL_ATTN_FIXED_SHAPE", "0") == "1"
   454	    attn_valid_mask = None
   455	    with profile_section("full_attn_t1.cache_write"):
   456	        if fixed_shape:
   457	            pos_idx = pos_tensor.reshape(-1)[:1]
   458	            K_cache_full.index_copy_(2, pos_idx, k_new)
   459	            V_cache_full.index_copy_(2, pos_idx, v_new)
   460	        else:
   461	            K_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = k_new
   462	            V_cache_full[:, :, cached_seq_len:cached_seq_len + 1, :] = v_new
   463	
   464	    # 5. Select cache window: fixed full + position mask, or variable slice.
   465	    if fixed_shape:
   466	        K_used = K_cache_full
   467	        V_used = V_cache_full
   468	        max_T = K_cache_full.shape[2]
   469	        positions = torch.arange(max_T, device=h_new.device).view(1, 1, 1, max_T)
   470	        attn_valid_mask = positions <= pos_tensor.reshape(1, 1, 1, 1)
   471	    else:
   472	        new_total = cached_seq_len + 1
   473	        K_used = K_cache_full[:, :, :new_total, :]
   474	        V_used = V_cache_full[:, :, :new_total, :]
   475	
   476	    full_attn_backend = os.environ.get("LYNN_FULL_ATTN_DECODE_BACKEND", "sdpa")
   477	    if fixed_shape:
   478	        # Direct masked SDPA (bool mask: True = attend). Bypasses
   479	        # _full_attn_attention's q.shape[2]==2 K2 special-casing.
   480	        with profile_section("full_attn_t1.attention.sdpa_fixed"):
   481	            attn_out = F.scaled_dot_product_attention(
   482	                q, K_used, V_used,
   483	                attn_mask=attn_valid_mask,
   484	                is_causal=False,
   485	                enable_gqa=(H_KV != H_Q),
   486	            )
   487	    elif full_attn_backend == "manual_gqa":
   488	        # Decode uses a single query token. Avoid SDPA launch/dispatch overhead
   489	        # by doing grouped-query attention explicitly without materializing
   490	        # repeated KV heads. This is opt-in until parity + latency are proven.
   491	        group = H_Q // H_KV
   492	        q_grouped = q.view(B, H_KV, group, 1, head_dim)
   493	        scale = 1.0 / math.sqrt(head_dim)
   494	        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped.float(), K_used.float()) * scale
   495	        probs = torch.softmax(scores, dim=-1).to(V_used.dtype)
   496	        attn_out = torch.einsum("bhgqk,bhkd->bhgqd", probs, V_used)
   497	        attn_out = attn_out.reshape(B, H_Q, 1, head_dim)
   498	    elif full_attn_backend == "sdpa":
   499	        # SDPA with enable_gqa=True (PyTorch 2.5+) — internal broadcast,
   500	        # no memory expansion. Math equivalent to explicit repeat_interleave+SDPA.
   501	        # Replaces 2× repeat_interleave (8x mem copy on H_Q/H_KV=8) with view-only.
   502	        attn_out = _full_attn_attention(q, K_used, V_used, enable_gqa=(H_KV != H_Q))
   503	    else:
   504	        raise ValueError(f"Unknown LYNN_FULL_ATTN_DECODE_BACKEND: {full_attn_backend}")
   505	
   506	    # 8. attn_output_gate
   507	    with profile_section("full_attn_t1.gate"):
   508	        attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
   509	
   510	    # 9. o_proj
   511	    with profile_section("full_attn_t1.o_proj_total"):
   512	        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
   513	        return _full_attn_o_proj(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))
   514	
   515	

exec
/bin/zsh -lc "nl -ba engine/full_forward.py | sed -n '964,1035p'" in /Users/lynn/Downloads/Lynn/lynn-engine
 succeeded in 0ms:
   964	def _decode_layer(
   965	    h_new,
   966	    position_id,
   967	    layer_type,
   968	    w,
   969	    cfg,
   970	    state,
   971	    layer_idx,
   972	    *,
   973	    moe_fn=None,
   974	    recurrent_backend: str | None = None,
   975	    linear_state_update: str | None = None,
   976	):
   977	    """Forward one DecoderLayer in decode mode (T=1) using cached state.
   978	
   979	    LYNN_MOE_IMPL env var selects MoE implementation:
   980	      optimized    Phase 3.2.1 active-experts loop (default)
   981	      bmm          Phase 3.2.2 batched matmul
   982	      indexed_bmm  Phase 3.2.2.5 pre-stacked grouped indexed_bmm
   983	    """
   984	    from engine.incremental_decode import decode_full_attn, decode_linear_attn
   985	
   986	    residual = h_new
   987	    h_norm = _rms_norm(h_new, w["input_layernorm.weight"])
   988	    if layer_type == "linear_attention":
   989	        if recurrent_backend is None:
   990	            recurrent_backend = os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch")
   991	        attn_out, new_state, new_conv = decode_linear_attn(
   992	            h_norm, w,
   993	            state.recurrent_state[layer_idx],
   994	            state.conv_state[layer_idx],
   995	            recurrent_backend=recurrent_backend,
   996	        )
   997	        if linear_state_update is None:
   998	            linear_state_update = os.environ.get("LYNN_LINEAR_STATE_UPDATE", "assign")
   999	        if linear_state_update == "inplace":
  1000	            recurrent_target = state.recurrent_state[layer_idx]
  1001	            if recurrent_target.data_ptr() != new_state.data_ptr():
  1002	                recurrent_target.copy_(new_state)
  1003	            conv_target = state.conv_state[layer_idx]
  1004	            if conv_target.data_ptr() != new_conv.data_ptr():
  1005	                conv_target.copy_(new_conv)
  1006	        else:
  1007	            state.update_linear_attn_state(layer_idx, new_state, new_conv)
  1008	    else:
  1009	        K, V = state.kv_cache[layer_idx]
  1010	        attn_out = decode_full_attn(
  1011	            h_norm, position_id, w, cfg, K, V,
  1012	            cached_seq_len=state.seq_len,
  1013	        )
  1014	    h = residual + attn_out
  1015	
  1016	    residual = h
  1017	    h_norm = _rms_norm(h, w["post_attention_layernorm.weight"])
  1018	    if cfg.get("is_moe", int(cfg.get("num_experts", 0) or 0) > 0):
  1019	        if moe_fn is None:
  1020	            moe_fn = _resolve_decode_moe_impl(os.environ.get("LYNN_MOE_IMPL", "optimized"))
  1021	        ffn_out = moe_fn(h_norm, w, cfg)
  1022	    else:
  1023	        ffn_out = _dense_ffn_forward(h_norm, w)
  1024	    return residual + ffn_out
  1025	
  1026	
  1027	def generate_incremental(model_dir, prompt, max_new=5, device="cuda",
  1028	                         dtype=torch.bfloat16, verbose=True, max_seq_len=2048):
  1029	    """Phase 3.1 incremental decode: prefill once, then 1-token-per-step decode.
  1030	
  1031	    Compared to generate_greedy (brute-force), this should be ~10x faster
  1032	    on Spark for short generations and scale O(1) per token (vs O(T) brute).
  1033	    """
  1034	    from engine.loader import load_qwen36_layer
  1035	    from engine.inference_state import LynnInferenceState, infer_layer_types

codex
Assumption for launch-saved estimates: 40 decode layers, MoE on each layer, with the repo’s comments implying ~30 linear-attention / ~10 full-attention split. Scale linearly if the actual layer mix differs.

1. **Fuse layer RMSNorm + residual glue**
   - **Merge:** `_decode_layer` input RMSNorm, attention residual add, post-attention RMSNorm, and final FFN residual add around [full_forward.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/full_forward.py:986).
   - **Approach:** Add Triton kernels for:
     - `rmsnorm_kernel(x, weight) -> h_norm`
     - `residual_add_rmsnorm_kernel(residual, attn_out, post_norm_weight) -> h_after_attn, h_norm_for_moe`
     - fold final `residual + ffn_out` into the MoE/shared final output kernel where possible.
     Current `_rms_norm` is PyTorch scalar op composition, so this cuts multiple tiny elementwise launches per norm.
   - **Est. saved:** ~2-4 launches/layer, ~80-160/token depending current eager op decomposition.
   - **Exactness risk:** Low-medium. RMSNorm order and BF16 cast must match Qwen’s `(1.0 + weight)` behavior. Mitigate with per-layer tensor parity and full greedy token parity.
   - **Effort:** M.

2. **Fuse MoE router matvec + top-k + softmax**
   - **Merge:** `_router_linear`, `_router_topk`, `_router_softmax`, and `expert_indices[0].to(torch.int32).contiguous()` in [_moe_forward_decode_packed_nvfp4_fixed_triton](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:680), especially lines 683-692.
   - **Approach:** One Triton router kernel: one program computes 256 router logits from hidden `[2048]`, maintains top-8 in registers, applies FP32 softmax over top-k, stores `expert_ids:int32` and `routing_weights:fp32` into prepared scratch. Do not materialize full `[1, E]` logits unless debug mode asks for it.
   - **Est. saved:** ~2-3 launches/MoE layer, ~80-120/token.
   - **Exactness risk:** High. Router top-k is token-critical; tiny logit drift can change experts. Mitigate by matching current accumulation/cast policy as closely as possible, requiring exact `expert_ids` parity on a large prompt suite, and keeping PyTorch router fallback.
   - **Effort:** M-L.

3. **Fuse shared expert with active MoE finalization**
   - **Merge:** Shared expert gate/up, SwiGLU, down, shared gate, and add path at [moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:826) and [moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:1115), plus `_finalize_shared_expert_output`.
   - **Approach:** Keep active routed MoE’s existing grouped gate/up and down kernels, but add a prepared shared-expert path:
     - kernel A: shared `gate_up` matvec + SwiGLU, optionally compute shared gate scalar.
     - kernel B: extend `nvfp4_grouped_down_weighted_sum_prepared` / `_grouped_down_weighted_sum_kernel` to also do shared down projection and final `active + shared * sigmoid(shared_gate)`.
   - **Est. saved:** ~2-4 launches/MoE layer, ~80-160/token if shared expert is enabled on all layers.
   - **Exactness risk:** Medium-high. BF16 `F.linear` shared expert accumulation will differ if rewritten in Triton. Mitigate behind env gate; compare shared-only output, final MoE output, and greedy tokens.
   - **Effort:** L.

4. **Fuse active NVFP4 gate/up + down into one active-MoE boundary**
   - **Merge:** `nvfp4_grouped_gate_up_silu_fast_decode(_prepared)` at [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:760) / [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:818) with `nvfp4_grouped_down_weighted_sum(_prepared)` at [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:936) / [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:1057).
   - **Approach:** New native/Triton active-MoE kernel consuming hidden, top-k ids, routing weights, packed gate/up and down weights, producing final hidden. Avoid writing/reading `inter [8,512]`. This is not a naive per-output-block recompute; it needs a careful tiled design or native CUDA cooperative staging.
   - **Est. saved:** 1 launch/MoE layer, ~40/token.
   - **Exactness risk:** Medium. Same scalar NVFP4 decode table and same FP32 accumulation order per projection are required. Performance risk is high if intermediate recomputation creeps in.
   - **Effort:** L.

5. **Fuse linear-attention decode micro-ops**
   - **Merge:** In [decode_linear_attn](/Users/lynn/Downloads/Lynn/lynn-engine/engine/incremental_decode.py:1090), combine projection-adjacent scalar ops, conv update, recurrent prep, RMSNormGated, and out projection around lines 1116-1205.
   - **Approach:** Extend existing Triton pieces:
     - `linear_conv1d_update_triton`
     - `recurrent_gated_delta_fused_prepare_from_outconv_gqa`
     - `rms_norm_gated_triton`
     into a two-kernel path: `conv + beta/g + recurrent` and `RMSNormGated + out_proj`. The existing `from_outconv_gqa` path is the right base because it already avoids q/k materialization.
   - **Est. saved:** ~1-3 launches/linear-attn layer, ~30-90/token.
   - **Exactness risk:** Medium. Stateful recurrent update is token-critical; keep FP32 state math and l2norm order identical.
   - **Effort:** L.

6. **Fuse full-attention decode side effects and projection epilogue**
   - **Merge:** In [decode_full_attn](/Users/lynn/Downloads/Lynn/lynn-engine/engine/incremental_decode.py:382), combine q/k norm+RoPE with K/V cache write at lines 434-462, and combine attention gate with `o_proj` at lines 506-513.
   - **Approach:** Extend `qk_norm_rope_pair_triton` to write post-RoPE K directly into `K_cache_full` and V into `V_cache_full`, returning only Q for SDPA. Add rowwise `o_proj` variant that applies `sigmoid(gate)` inside the projection load path.
   - **Est. saved:** ~1-3 launches/full-attn layer, ~10-30/token.
   - **Exactness risk:** Low-medium for cache write fusion; medium for gated `o_proj` if replacing cuBLAS with Triton rowwise accumulation.
   - **Effort:** M.

**Highest-ROI pick:** fused MoE router matvec + top-k + softmax. It attacks a repeated per-layer launch cluster before every active expert call, can save ~80-120 launches/token by itself, and does not require redesigning the packed NVFP4 expert kernels. The only serious issue is router exactness, so I’d implement it behind an env gate with strict `expert_ids` parity before measuring generation speed.
tokens used
90,140
Assumption for launch-saved estimates: 40 decode layers, MoE on each layer, with the repo’s comments implying ~30 linear-attention / ~10 full-attention split. Scale linearly if the actual layer mix differs.

1. **Fuse layer RMSNorm + residual glue**
   - **Merge:** `_decode_layer` input RMSNorm, attention residual add, post-attention RMSNorm, and final FFN residual add around [full_forward.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/full_forward.py:986).
   - **Approach:** Add Triton kernels for:
     - `rmsnorm_kernel(x, weight) -> h_norm`
     - `residual_add_rmsnorm_kernel(residual, attn_out, post_norm_weight) -> h_after_attn, h_norm_for_moe`
     - fold final `residual + ffn_out` into the MoE/shared final output kernel where possible.
     Current `_rms_norm` is PyTorch scalar op composition, so this cuts multiple tiny elementwise launches per norm.
   - **Est. saved:** ~2-4 launches/layer, ~80-160/token depending current eager op decomposition.
   - **Exactness risk:** Low-medium. RMSNorm order and BF16 cast must match Qwen’s `(1.0 + weight)` behavior. Mitigate with per-layer tensor parity and full greedy token parity.
   - **Effort:** M.

2. **Fuse MoE router matvec + top-k + softmax**
   - **Merge:** `_router_linear`, `_router_topk`, `_router_softmax`, and `expert_indices[0].to(torch.int32).contiguous()` in [_moe_forward_decode_packed_nvfp4_fixed_triton](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:680), especially lines 683-692.
   - **Approach:** One Triton router kernel: one program computes 256 router logits from hidden `[2048]`, maintains top-8 in registers, applies FP32 softmax over top-k, stores `expert_ids:int32` and `routing_weights:fp32` into prepared scratch. Do not materialize full `[1, E]` logits unless debug mode asks for it.
   - **Est. saved:** ~2-3 launches/MoE layer, ~80-120/token.
   - **Exactness risk:** High. Router top-k is token-critical; tiny logit drift can change experts. Mitigate by matching current accumulation/cast policy as closely as possible, requiring exact `expert_ids` parity on a large prompt suite, and keeping PyTorch router fallback.
   - **Effort:** M-L.

3. **Fuse shared expert with active MoE finalization**
   - **Merge:** Shared expert gate/up, SwiGLU, down, shared gate, and add path at [moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:826) and [moe_packed_nvfp4.py](/Users/lynn/Downloads/Lynn/lynn-engine/engine/moe_packed_nvfp4.py:1115), plus `_finalize_shared_expert_output`.
   - **Approach:** Keep active routed MoE’s existing grouped gate/up and down kernels, but add a prepared shared-expert path:
     - kernel A: shared `gate_up` matvec + SwiGLU, optionally compute shared gate scalar.
     - kernel B: extend `nvfp4_grouped_down_weighted_sum_prepared` / `_grouped_down_weighted_sum_kernel` to also do shared down projection and final `active + shared * sigmoid(shared_gate)`.
   - **Est. saved:** ~2-4 launches/MoE layer, ~80-160/token if shared expert is enabled on all layers.
   - **Exactness risk:** Medium-high. BF16 `F.linear` shared expert accumulation will differ if rewritten in Triton. Mitigate behind env gate; compare shared-only output, final MoE output, and greedy tokens.
   - **Effort:** L.

4. **Fuse active NVFP4 gate/up + down into one active-MoE boundary**
   - **Merge:** `nvfp4_grouped_gate_up_silu_fast_decode(_prepared)` at [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:760) / [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:818) with `nvfp4_grouped_down_weighted_sum(_prepared)` at [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:936) / [triton_kernels/nvfp4_moe.py](/Users/lynn/Downloads/Lynn/lynn-engine/triton_kernels/nvfp4_moe.py:1057).
   - **Approach:** New native/Triton active-MoE kernel consuming hidden, top-k ids, routing weights, packed gate/up and down weights, producing final hidden. Avoid writing/reading `inter [8,512]`. This is not a naive per-output-block recompute; it needs a careful tiled design or native CUDA cooperative staging.
   - **Est. saved:** 1 launch/MoE layer, ~40/token.
   - **Exactness risk:** Medium. Same scalar NVFP4 decode table and same FP32 accumulation order per projection are required. Performance risk is high if intermediate recomputation creeps in.
   - **Effort:** L.

5. **Fuse linear-attention decode micro-ops**
   - **Merge:** In [decode_linear_attn](/Users/lynn/Downloads/Lynn/lynn-engine/engine/incremental_decode.py:1090), combine projection-adjacent scalar ops, conv update, recurrent prep, RMSNormGated, and out projection around lines 1116-1205.
   - **Approach:** Extend existing Triton pieces:
     - `linear_conv1d_update_triton`
     - `recurrent_gated_delta_fused_prepare_from_outconv_gqa`
     - `rms_norm_gated_triton`
     into a two-kernel path: `conv + beta/g + recurrent` and `RMSNormGated + out_proj`. The existing `from_outconv_gqa` path is the right base because it already avoids q/k materialization.
   - **Est. saved:** ~1-3 launches/linear-attn layer, ~30-90/token.
   - **Exactness risk:** Medium. Stateful recurrent update is token-critical; keep FP32 state math and l2norm order identical.
   - **Effort:** L.

6. **Fuse full-attention decode side effects and projection epilogue**
   - **Merge:** In [decode_full_attn](/Users/lynn/Downloads/Lynn/lynn-engine/engine/incremental_decode.py:382), combine q/k norm+RoPE with K/V cache write at lines 434-462, and combine attention gate with `o_proj` at lines 506-513.
   - **Approach:** Extend `qk_norm_rope_pair_triton` to write post-RoPE K directly into `K_cache_full` and V into `V_cache_full`, returning only Q for SDPA. Add rowwise `o_proj` variant that applies `sigmoid(gate)` inside the projection load path.
   - **Est. saved:** ~1-3 launches/full-attn layer, ~10-30/token.
   - **Exactness risk:** Low-medium for cache write fusion; medium for gated `o_proj` if replacing cuBLAS with Triton rowwise accumulation.
   - **Effort:** M.

**Highest-ROI pick:** fused MoE router matvec + top-k + softmax. It attacks a repeated per-layer launch cluster before every active expert call, can save ~80-120 launches/token by itself, and does not require redesigning the packed NVFP4 expert kernels. The only serious issue is router exactness, so I’d implement it behind an env gate with strict `expert_ids` parity before measuring generation speed.
