#!/usr/bin/env python3
"""P6-G: segment profile for a full-attention decode layer.

This profiles the hot full-attention layers that dominate the P6-F full-token
profile after the linear-attention recurrent path was fused.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import (  # noqa: E402
    _apply_partial_rope,
    _build_rope_cos_sin,
    _linear,
    decode_full_attn,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _bench(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _prefill(runner: LynnIncrementalRunner, prompt: str, *, use_chat_template: bool):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    seq_len = ids.shape[1]
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(seq_len, device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = seq_len
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    next_id = int(logits[0].argmax().item())
    return next_id, state


def _full_attn_intermediates(h_norm: torch.Tensor, pos_id: int, w: dict[str, Any], cfg: dict[str, Any], state: LynnInferenceState, layer: int):
    K_cache, V_cache = state.kv_cache[layer]
    B, _, _ = h_norm.shape
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])

    q_full = _linear(h_norm, w["self_attn.q_proj.weight"])
    k_new = _linear(h_norm, w["self_attn.k_proj.weight"])
    v_new = _linear(h_norm, w["self_attn.v_proj.weight"])
    q_full_view = q_full.view(B, 1, H_Q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)
    gate = gate.transpose(1, 2)
    k_new = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    v_new = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
    q_norm = _rms_norm(q, w["self_attn.q_norm.weight"])
    k_norm = _rms_norm(k_new, w["self_attn.k_norm.weight"])
    pos_tensor = torch.tensor([[pos_id]], device=h_norm.device, dtype=torch.long)
    cos, sin = _build_rope_cos_sin(pos_tensor, rotary_dim, rope_theta, h_norm.device, h_norm.dtype)
    q_rope = _apply_partial_rope(q_norm, cos, sin, rotary_dim)
    k_rope = _apply_partial_rope(k_norm, cos, sin, rotary_dim)
    K_used = K_cache[:, :, : state.seq_len + 1, :]
    V_used = V_cache[:, :, : state.seq_len + 1, :]
    return {
        "K_cache": K_cache,
        "V_cache": V_cache,
        "K_used": K_used,
        "V_used": V_used,
        "q": q,
        "gate": gate,
        "k_new": k_new,
        "v_new": v_new,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "H_Q": H_Q,
        "H_KV": H_KV,
        "head_dim": head_dim,
        "rotary_dim": rotary_dim,
        "cos": cos,
        "sin": sin,
    }


def _profile_attention(h_norm: torch.Tensor, pos_id: int, w: dict[str, Any], cfg: dict[str, Any], state: LynnInferenceState, layer: int, *, warmup: int, iters: int):
    mid = _full_attn_intermediates(h_norm, pos_id, w, cfg, state, layer)

    def q_proj():
        return _linear(h_norm, w["self_attn.q_proj.weight"])

    def k_proj():
        return _linear(h_norm, w["self_attn.k_proj.weight"])

    def v_proj():
        return _linear(h_norm, w["self_attn.v_proj.weight"])

    def reshape_chunk():
        q_full = q_proj()
        k_new = k_proj()
        v_new = v_proj()
        B = h_norm.shape[0]
        H_Q = mid["H_Q"]
        H_KV = mid["H_KV"]
        head_dim = mid["head_dim"]
        q_full_view = q_full.view(B, 1, H_Q, head_dim * 2)
        q, gate = q_full_view.chunk(2, dim=-1)
        return q.transpose(1, 2), gate.transpose(1, 2), k_new.view(B, 1, H_KV, head_dim).transpose(1, 2), v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)

    def qk_norm():
        return _rms_norm(mid["q"], w["self_attn.q_norm.weight"]), _rms_norm(mid["k_new"], w["self_attn.k_norm.weight"])

    def rope_apply():
        q = _apply_partial_rope(mid["q_norm"], mid["cos"], mid["sin"], mid["rotary_dim"])
        k = _apply_partial_rope(mid["k_norm"], mid["cos"], mid["sin"], mid["rotary_dim"])
        return q, k

    def cache_write():
        mid["K_cache"][:, :, state.seq_len : state.seq_len + 1, :] = mid["k_rope"]
        mid["V_cache"][:, :, state.seq_len : state.seq_len + 1, :] = mid["v_new"]
        return mid["K_cache"], mid["V_cache"]

    def sdpa_gqa():
        return F.scaled_dot_product_attention(
            mid["q_rope"],
            mid["K_used"],
            mid["V_used"],
            is_causal=False,
            enable_gqa=(mid["H_KV"] != mid["H_Q"]),
        )

    attn_core = sdpa_gqa()

    def attn_gate():
        return attn_core * torch.sigmoid(mid["gate"].float()).to(attn_core.dtype)

    gated = attn_gate()

    def o_proj():
        x = gated.transpose(1, 2).contiguous().view(h_norm.shape[0], 1, mid["H_Q"] * mid["head_dim"])
        return _linear(x, w["self_attn.o_proj.weight"])

    def full_decode_attn():
        return decode_full_attn(
            h_norm,
            pos_id,
            w,
            cfg,
            mid["K_cache"],
            mid["V_cache"],
            cached_seq_len=state.seq_len,
        )

    fns = {
        "attn.q_proj": q_proj,
        "attn.k_proj": k_proj,
        "attn.v_proj": v_proj,
        "attn.reshape_chunk": reshape_chunk,
        "attn.qk_norm": qk_norm,
        "attn.rope_apply": rope_apply,
        "attn.cache_write": cache_write,
        "attn.sdpa_gqa": sdpa_gqa,
        "attn.output_gate": attn_gate,
        "attn.o_proj": o_proj,
        "attn.full_decode_recomposed": full_decode_attn,
    }
    return {name: _bench(fn, warmup=warmup, iters=iters) for name, fn in fns.items()}


def _expert_ffn(x: torch.Tensor, w: dict[str, Any], expert_id: int) -> torch.Tensor:
    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
        gate_up = _linear(x, w["mlp.experts.gate_up_proj"][expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        return _linear(F.silu(gate) * up, w["mlp.experts.down_proj"][expert_id])
    gate = _linear(x, w[f"mlp.experts.{expert_id}.gate_proj.weight"])
    up = _linear(x, w[f"mlp.experts.{expert_id}.up_proj.weight"])
    return _linear(F.silu(gate) * up, w[f"mlp.experts.{expert_id}.down_proj.weight"])


def _profile_moe(h_norm: torch.Tensor, w: dict[str, Any], cfg: dict[str, Any], *, warmup: int, iters: int):
    h_flat = h_norm.view(-1, h_norm.shape[-1])
    router_logits = _linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, cfg["num_experts_per_tok"], dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(h_norm.dtype)
    expert_ids = expert_indices[0].tolist()

    def router():
        return _linear(h_flat, w["mlp.gate.weight"])

    def topk_softmax():
        logits = router()
        rw, idx = torch.topk(logits, cfg["num_experts_per_tok"], dim=-1)
        return F.softmax(rw, dim=-1, dtype=torch.float32).to(h_norm.dtype), idx

    def active_expert_loop():
        out = torch.zeros_like(h_flat)
        for slot, expert_id in enumerate(expert_ids):
            out = out + _expert_ffn(h_flat, w, expert_id) * routing_weights[0, slot]
        return out

    def shared_expert():
        gate = _linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up = _linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        out = _linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            out = out * torch.sigmoid(_linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        return out

    def full_moe_recomposed():
        return active_expert_loop() + shared_expert()

    fns = {
        "moe.router": router,
        "moe.topk_softmax": topk_softmax,
        "moe.active_expert_loop": active_expert_loop,
        "moe.shared_expert": shared_expert,
        "moe.full_recomposed": full_moe_recomposed,
    }
    return {name: _bench(fn, warmup=warmup, iters=iters) for name, fn in fns.items()}, expert_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")

    next_id, state = _prefill(runner, args.prompt, use_chat_template=args.chat_template)
    token = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
    pos_id = state.seq_len
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]

    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])

    def input_norm():
        return _rms_norm(h0, w["input_layernorm.weight"])

    h_attn_norm = input_norm()
    attn_lat = _profile_attention(h_attn_norm, pos_id, w, cfg, state, args.layer, warmup=args.warmup, iters=args.iters)
    attn_out = decode_full_attn(
        h_attn_norm,
        pos_id,
        w,
        cfg,
        state.kv_cache[args.layer][0],
        state.kv_cache[args.layer][1],
        cached_seq_len=state.seq_len,
    )
    h_after_attn = h0 + attn_out

    def post_attn_norm():
        return _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])

    h_moe_norm = post_attn_norm()
    moe_lat, expert_ids = _profile_moe(h_moe_norm, w, cfg, warmup=args.warmup, iters=args.iters)

    def full_layer_recomposed():
        h_norm = _rms_norm(h0, w["input_layernorm.weight"])
        attn = decode_full_attn(
            h_norm,
            pos_id,
            w,
            cfg,
            state.kv_cache[args.layer][0],
            state.kv_cache[args.layer][1],
            cached_seq_len=state.seq_len,
        )
        h = h0 + attn
        h_norm2 = _rms_norm(h, w["post_attention_layernorm.weight"])
        h_flat = h_norm2.view(-1, h_norm2.shape[-1])
        logits = _linear(h_flat, w["mlp.gate.weight"])
        rw, idx = torch.topk(logits, cfg["num_experts_per_tok"], dim=-1)
        rw = F.softmax(rw, dim=-1, dtype=torch.float32).to(h_norm2.dtype)
        out = torch.zeros_like(h_flat)
        for slot, expert_id in enumerate(idx[0].tolist()):
            out = out + _expert_ffn(h_flat, w, expert_id) * rw[0, slot]
        gate = _linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up = _linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = _linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in w:
            shared = shared * torch.sigmoid(_linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        return h + (out + shared).view_as(h)

    segment_ms = {
        "layer.input_rmsnorm": _bench(input_norm, warmup=args.warmup, iters=args.iters),
        **attn_lat,
        "layer.post_attn_rmsnorm": _bench(post_attn_norm, warmup=args.warmup, iters=args.iters),
        **moe_lat,
        "layer.full_recomposed": _bench(full_layer_recomposed, warmup=max(1, args.warmup // 2), iters=max(5, args.iters // 4)),
    }
    result = {
        "schema_version": "lynn-engine-p6g-full-attention-layer-profile-v1",
        "model": args.model,
        "layer": args.layer,
        "layer_type": LAYER_TYPES[args.layer],
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "load_seconds": runner.load_seconds,
        "prompt_tokens": int(state.seq_len),
        "active_experts": expert_ids,
        "segment_ms": segment_ms,
        "top_segments": sorted(
            [{"segment": k, "latency_ms": v} for k, v in segment_ms.items() if not k.endswith("recomposed")],
            key=lambda x: x["latency_ms"],
            reverse=True,
        )[:12],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
