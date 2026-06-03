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


_TRITON_RMSNORM = None
_RMSNORM_OFFSET_CACHE: dict = {}


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen3_5MoeRMSNorm — note the `(1.0 + weight)` factor, not plain `weight`.

    From HF transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:806 ::
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # the +1 offset
        return output.type_as(x)

    See https://github.com/huggingface/transformers/pull/29402 — Qwen-family
    diverges from Llama-style RMSNorm (which is `weight * x` only).
    """
    if os.environ.get("LYNN_RMSNORM_FUSED", "0") == "1" and x.is_cuda:
        # Fused Triton RMSNorm: 1 launch vs ~6-8 eager (pow/mean/add/rsqrt/mul/...).
        # rmsnorm_triton does x/rms*scale; pass the cached (1.0+weight) so it matches
        # Qwen's +1 offset exactly. Offset computed once per weight (data_ptr cache).
        global _TRITON_RMSNORM
        if _TRITON_RMSNORM is None:
            from triton_kernels.rmsnorm import make_triton_rmsnorm
            _TRITON_RMSNORM = make_triton_rmsnorm()
        ck = (weight.data_ptr(), int(weight.shape[0]))
        offset = _RMSNORM_OFFSET_CACHE.get(ck)
        if offset is None:
            offset = (1.0 + weight.float()).contiguous()
            _RMSNORM_OFFSET_CACHE[ck] = offset
        return _TRITON_RMSNORM(x, offset, eps).to(x.dtype)
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


def _packed_prefill_slow_enabled() -> bool:
    return os.environ.get("LYNN_PACKED_PREFILL_SLOW", "0").lower() in {"1", "true", "yes", "on"}


def _moe_forward_packed_prefill_decode_kernel_slow(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """Functional packed-NVFP4 MoE prefill by replaying the T=1 decode MoE.

    This path is useful as a fast diagnostic, but Spark P0.1 showed it is not
    token-exact against the BF16 prefill state across decode steps. Keep it
    available behind LYNN_PACKED_PREFILL_SLOW_MODE=decode_kernel only.
    """
    if h.shape[0] != 1:
        raise NotImplementedError("LYNN_PACKED_PREFILL_SLOW currently supports batch=1")
    from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4

    flat = h.reshape(-1, 1, h.shape[-1])
    outs = [
        moe_forward_decode_packed_nvfp4(flat[i : i + 1], w, cfg).reshape(-1)
        for i in range(flat.shape[0])
    ]
    return torch.stack(outs, dim=0).reshape_as(h).to(h.dtype)


def _moe_forward_packed_prefill_stream_bf16(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    """No-reload packed prefill proof by streaming one layer's BF16 shadow.

    P0.1 is a correctness/memory gate, not the final performance path. The
    persistent 60 GiB MoE BF16 shadow has been released; this function reads the
    resident packed NVFP4 tensors, dequants only the current layer into temporary
    BF16 tensors, runs the same math as the original BF16 prefill MoE, then lets
    those temporaries die before the next layer.

    That proves zero-resident-shadow serving can be token-exact. P2 replaces this
    with real grouped M>1 packed kernels so we stop paying the per-layer temporary
    dequant cost.
    """
    from engine.moe_packed_nvfp4 import _dequant_nvfp4_slot

    B, M, D = h.shape
    E = int(cfg.get("num_experts", 0) or 0)
    if E <= 0:
        raise RuntimeError("MoE packed-prefill stream called for a dense FFN layer")
    K = cfg["num_experts_per_tok"]
    h_flat = h.view(B * M, D)

    gate_up_shadow = _dequant_nvfp4_slot(
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        h.device,
    ).to(h.dtype)
    down_shadow = _dequant_nvfp4_slot(
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        h.device,
    ).to(h.dtype)

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, K, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h.dtype)
    moe_out = torch.zeros_like(h_flat)

    for e in range(E):
        mask = expert_indices == e
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        gate_up = F.linear(x_e, gate_up_shadow[e])
        gate_e, up_e = gate_up.chunk(2, dim=-1)
        ffn_e = F.linear(F.silu(gate_e) * up_e, down_shadow[e])
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert is not part of the 60 GiB grouped MoE shadow release in P0.1.
    # Keep the existing BF16 shared path so this proof isolates the banked shadow.
    if "mlp.shared_expert.gate_proj.weight" in w:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    del gate_up_shadow, down_shadow
    return moe_out.view(B, M, D)


def _moe_forward_packed_prefill_slow(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    mode = os.environ.get("LYNN_PACKED_PREFILL_SLOW_MODE", "stream_bf16").strip().lower()
    if mode in {"stream_bf16", "stream", "bf16"}:
        return _moe_forward_packed_prefill_stream_bf16(h, w, cfg)
    if mode in {"decode_kernel", "decode", "packed_decode"}:
        return _moe_forward_packed_prefill_decode_kernel_slow(h, w, cfg)
    raise ValueError(
        "LYNN_PACKED_PREFILL_SLOW_MODE must be stream_bf16 or decode_kernel, "
        f"got {mode!r}"
    )


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

    if (
        _packed_prefill_slow_enabled()
        and "mlp.experts._gate_up_packed" in w
        and "mlp.experts._down_packed" in w
    ):
        return _moe_forward_packed_prefill_slow(h, w, cfg)

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

    if verbose:
        print(f"prompt: {prompt!r}, T={T}, max_new={max_new}", flush=True)

    # Outside weights (resident)
    if verbose:
        print("Loading outside weights ...", flush=True)
    outside = load_outside_weights(model_dir, device, dtype)

    # All 40 layers resident
    if verbose:
        print(f"Loading all {n_layers} layers (resident) ...", flush=True)
    layer_weights = []
    layer_cfgs = []
    t_start = time.time()
    for i in range(n_layers):
        w, inferred = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                        device=device, dequant_dtype=dtype)
        layer_weights.append(w)
        layer_cfgs.append(_with_inferred_layer_config(cfg, inferred, i))
        if verbose and (i % 5 == 4 or i == n_layers - 1):
            print(f"  L{i:2}: cumulative {time.time()-t_start:.1f}s", flush=True)
    if verbose:
        print(f"All weights resident in {time.time()-t_start:.1f}s\n", flush=True)

    # Phase 3.2 implementation selector. Indexed decode needs original expert
    # weights for prefill, so stacking happens after prefill completes.
    import os as _os
    _impl = _os.environ.get("LYNN_MOE_IMPL", "optimized")

    # Allocate inference state
    state = LynnInferenceState.from_config(tc, batch=1, max_seq_len=max_seq_len, device=device, dtype=dtype)

    # === PREFILL ===
    if verbose:
        print(f"Prefill T={T} ...", flush=True)
    t0 = time.time()
    h = F.embedding(ids, outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
    for i in range(n_layers):
        h = _prefill_layer(h, pos, layer_types[i], layer_weights[i], layer_cfgs[i], state, i)
    state.seq_len = T

    # First token from prefill last position
    h_final = _rms_norm(h, outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], outside["lm_head.weight"])
    next_id = int(logits[0].argmax().item())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if verbose:
        print(f"  prefill done in {time.time()-t0:.2f}s, "
              f"next={next_id} {tok.decode([next_id])!r}", flush=True)
    new_ids = [next_id]

    # Phase 3.2 indexed decode preparation. Prefill above uses the baseline MoE
    # path, so only now can we replace per-expert weights with stacked tensors.
    if cfg["is_moe"] and _impl in ("indexed_bmm", "triton"):
        from triton_kernels.moe_expert_ffn import stack_expert_weights
        if verbose:
            print(f"Pre-stacking expert weights for {_impl} after prefill ...", flush=True)
        _t0 = time.time()
        for _i in range(n_layers):
            layer_e = layer_cfgs[_i]["num_experts"]
            stack_expert_weights(layer_weights[_i], num_experts=layer_e)
            for _e in range(layer_e):
                for _proj in ("gate_proj", "up_proj", "down_proj"):
                    _key = f"mlp.experts.{_e}.{_proj}.weight"
                    layer_weights[_i].pop(_key, None)
            if verbose and (_i % 10 == 9 or _i == n_layers - 1):
                torch.cuda.empty_cache()
                _free = torch.cuda.mem_get_info()[0] / 1e9
                print(f"  pre-stack L{_i:2}: {time.time()-_t0:.1f}s GPU free {_free:.1f} GB", flush=True)
        torch.cuda.empty_cache()
        if verbose:
            _free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"Pre-stack done {time.time()-_t0:.1f}s, GPU free {_free:.1f} GB\n", flush=True)

    # === DECODE ===
    for step in range(1, max_new):
        t0 = time.time()
        new_token_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        h = F.embedding(new_token_tensor, outside["model.language_model.embed_tokens.weight"])
        pos = state.seq_len   # new token position

        for i in range(n_layers):
            h = _decode_layer(h, pos, layer_types[i], layer_weights[i], layer_cfgs[i], state, i)

        state.seq_len += 1
        h_final = _rms_norm(h, outside["model.language_model.norm.weight"])
        logits = F.linear(h_final[:, -1, :], outside["lm_head.weight"])
        next_id = int(logits[0].argmax().item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if verbose:
            print(f"  decode step {step+1}/{max_new}  pos={state.seq_len}  "
                  f"{(time.time()-t0)*1000:.0f}ms  next={next_id} "
                  f"{tok.decode([next_id])!r}", flush=True)
        new_ids.append(next_id)

    full_ids = ids[0].tolist() + new_ids
    full_text = tok.decode(full_ids)
    return full_text, new_ids


# ----------------- multi-token greedy decode (brute-force, Phase 2) ---

def generate_greedy(model_dir: str, prompt: str, max_new: int = 5,
                    device: str = "cuda", dtype=torch.bfloat16, verbose: bool = True):
    """Brute-force greedy generation: re-prefill at every step.

    No KV cache — slower than a proper engine, but unambiguously correct.
    Loads all 40 layers resident once, then runs N successive full forwards.
    Per-step cost grows ~quadratic in (prompt_len + new_tokens) since the
    full forward sees every position each time.

    Returns:
        text: prompt + generated continuation
        new_token_ids: list of generated token ids
    """
    from engine.loader import load_qwen36_layer
    from engine.inference_state import infer_layer_types

    with open(Path(model_dir) / "config.json") as f:
        full_cfg = json.load(f)
    tc = full_cfg["text_config"]
    rope_p = tc.get("rope_parameters", {})
    num_experts = int(tc.get("num_experts", 0) or 0)
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
    layer_types = infer_layer_types(tc)
    n_layers = tc["num_hidden_layers"]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    if verbose:
        print(f"prompt: {prompt!r}  ids[:10]={ids[0].tolist()[:10]}", flush=True)

    # Outside + all 40 layers resident
    if verbose:
        print("Loading outside weights ...", flush=True)
    outside = load_outside_weights(model_dir, device, dtype)
    embed = outside["model.language_model.embed_tokens.weight"]
    lm_head_w = outside["lm_head.weight"]
    final_norm_w = outside["model.language_model.norm.weight"]

    if verbose:
        print(f"Loading all {n_layers} layers (resident) ...", flush=True)
    weights_per_layer = []
    layer_cfgs = []
    t_start = time.time()
    for i in range(n_layers):
        w, inferred = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                        device=device, dequant_dtype=dtype)
        weights_per_layer.append(w)
        layer_cfgs.append(_with_inferred_layer_config(cfg, inferred, i))
        if verbose and (i % 5 == 4 or i == n_layers - 1):
            print(f"  L{i:2}: cumulative {time.time()-t_start:.1f}s", flush=True)
    if verbose:
        print(f"All weights resident in {time.time()-t_start:.1f}s\n", flush=True)

    new_ids = []
    for step in range(max_new):
        T = ids.shape[1]
        pos = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)

        t0 = time.time()
        h = F.embedding(ids, embed)
        for i in range(n_layers):
            h = _layer_forward(h, pos, layer_types[i], weights_per_layer[i], layer_cfgs[i])
        h = _rms_norm(h, final_norm_w)
        last_h = h[:, -1, :]
        logits = F.linear(last_h, lm_head_w)
        next_id = int(logits[0].argmax().item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        new_ids.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if verbose:
            text = tok.decode([next_id]).replace("\n", "\\n")
            print(f"  step {step+1}/{max_new}  T={T}->{T+1}  "
                  f"{elapsed:.2f}s  next={next_id} {text!r}", flush=True)

    full_text = tok.decode(ids[0].tolist())
    return full_text, new_ids


# ----------------- end-to-end forward (single token, original) ---------

def run_forward(model_dir: str, prompt: str, max_new: int = 1, device: str = "cuda",
                dtype=torch.bfloat16, verbose: bool = True):
    """End-to-end Lynn-engine forward on `prompt`. Returns logits + top-1 token."""
    from engine.loader import load_qwen36_layer
    from engine.inference_state import infer_layer_types

    # Config
    with open(Path(model_dir) / "config.json") as f:
        full_config = json.load(f)
    tc = full_config["text_config"]
    rope_p = tc.get("rope_parameters", {})
    num_experts = int(tc.get("num_experts", 0) or 0)
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
    layer_types = infer_layer_types(tc)
    n_layers = tc["num_hidden_layers"]

    # Tokenize
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    B, T = input_ids.shape
    position_ids = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

    if verbose:
        print(f"prompt: {prompt!r}")
        print(f"input_ids shape: {input_ids.shape}, tokens: {input_ids[0].tolist()[:20]}...")

    # Load embeddings + final norm + lm_head (kept resident)
    if verbose:
        print(f"\nLoading outside weights ...")
    t0 = time.time()
    outside = load_outside_weights(model_dir, device, dtype)
    if verbose:
        print(f"  done in {time.time()-t0:.1f}s")

    # Embed
    h = F.embedding(input_ids, outside["model.language_model.embed_tokens.weight"])
    if verbose:
        print(f"\nh0 shape: {tuple(h.shape)}, mag: {h.float().abs().mean().item():.3f}")

    # Forward through layers
    t_total = time.time()
    for i in range(n_layers):
        layer_type = layer_types[i]
        t0 = time.time()
        weights, inferred = load_qwen36_layer(model_dir, i, num_experts=cfg["num_experts"],
                                              device=device, dequant_dtype=dtype)
        layer_cfg = _with_inferred_layer_config(cfg, inferred, i)
        t_load = time.time() - t0

        t0 = time.time()
        h = _layer_forward(h, position_ids, layer_type, weights, layer_cfg)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd = time.time() - t0

        if verbose:
            ne = int(layer_cfg.get("num_experts", 0) or 0)
            ffn_tag = f"E {ne:3d}" if ne > 0 else "dense"
            print(f"  L{i:2} ({layer_type[:6]}) load {t_load:5.1f}s  fwd {t_fwd*1000:5.0f}ms  "
                  f"{ffn_tag}  h_mag {h.float().abs().mean().item():.3f}")

        del weights
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if verbose:
        print(f"\nTotal forward: {time.time()-t_total:.1f}s")

    # Final norm
    h = _rms_norm(h, outside["model.language_model.norm.weight"])

    # lm_head
    last_h = h[:, -1, :]
    logits = F.linear(last_h, outside["lm_head.weight"])

    # Top-K
    top_k = 10
    topv, topi = torch.topk(logits[0], top_k)
    if verbose:
        print(f"\nTop-{top_k} next tokens (logit | id | text):")
        for v, i in zip(topv.tolist(), topi.tolist()):
            text = tok.decode([i]).replace("\n", "\\n")
            print(f"  {v:8.3f}  {i:8d}  {text!r}")

    return {
        "logits": logits.detach().cpu(),
        "top_token_id": topi[0].item(),
        "top_token_text": tok.decode([topi[0].item()]),
        "input_ids": input_ids,
        "tokenizer": tok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=1,
                    help="number of tokens to greedy-decode (1 = next-token only)")
    ap.add_argument("--mode", choices=["prefill", "brute", "incremental"],
                    default="incremental",
                    help="prefill = single-token next-logit only; "
                         "brute = brute-force re-prefill per token; "
                         "incremental = KV cache + recurrent state cache (Phase 3.1)")
    args = ap.parse_args()

    sys.path.insert(0, "/work")

    if args.mode == "prefill" or args.max_new <= 1:
        out = run_forward(args.model, args.prompt, device=args.device)
        print(f"\n=== Lynn Engine top-1 next token: "
              f"{out['top_token_id']} ({out['top_token_text']!r}) ===")
    elif args.mode == "brute":
        text, new_ids = generate_greedy(args.model, args.prompt,
                                        max_new=args.max_new, device=args.device)
        print(f"\n=== Lynn Engine brute-force ({args.max_new} new tokens) ===")
        print(text)
        print(f"\nnew_ids: {new_ids}")
    else:  # incremental
        text, new_ids = generate_incremental(args.model, args.prompt,
                                             max_new=args.max_new, device=args.device)
        print(f"\n=== Lynn Engine incremental ({args.max_new} new tokens) ===")
        print(text)
        print(f"\nnew_ids: {new_ids}")


if __name__ == "__main__":
    main()
