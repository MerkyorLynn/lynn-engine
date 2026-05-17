#!/usr/bin/env python3
"""P27: service-shaped full-attention layer segment profile.

P26 shows the ten eager full-attention layers cost about 3.1 ms/token under the
R6000 Config D runtime. P6G is useful history, but it profiles a mostly BF16
handwritten path. P27 profiles one real full-attention layer through the
resident runner with the active `LYNN_MOE_IMPL` and packed NVFP4 aliases.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.incremental_decode import (  # noqa: E402
    _build_rope_cos_sin,
    _decode_weight,
    _linear,
    _qk_norm_rope_pair_decode,
    decode_full_attn,
)
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


DEFAULT_PROMPT = (
    "请连续输出一段关于 MoE 推理优化、NVFP4、CUDA graph 和工具调用服务化的中文技术说明。"
    "要求持续展开，不要提前结束。"
)


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


def _env_snapshot() -> dict[str, str | None]:
    names = [
        "LYNN_MOE_IMPL",
        "LYNN_MOE_FAST_FIXED",
        "LYNN_NATIVE_DOWN_BACKEND",
        "LYNN_PACKED_DECODE_FULL_ATTN",
        "LYNN_FULL_ATTN_DECODE_BACKEND",
        "LYNN_QK_NORM_ROPE_BACKEND",
    ]
    return {name: os.environ.get(name) for name in names}


def _prefill_and_first_token(
    runner: LynnIncrementalRunner,
    prompt: str,
    *,
    use_chat_template: bool,
) -> tuple[int, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    next_id = int(logits[0].argmax().item())
    if runner.device.startswith("cuda"):
        torch.cuda.synchronize()
    return next_id, state


def _hidden_before_layer(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.tensor([[int(token_id)]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for idx in range(layer_idx):
        h = runner._decode_layer_fast(h, pos_tensor, state, idx)
    if runner.device.startswith("cuda"):
        torch.cuda.synchronize()
    return h, pos_tensor


def _full_attn_mid(
    h_norm: torch.Tensor,
    pos_tensor: torch.Tensor,
    w: dict[str, Any],
    cfg: dict[str, Any],
    state: LynnInferenceState,
    layer_idx: int,
) -> dict[str, Any]:
    k_cache, v_cache = state.kv_cache[layer_idx]
    bsz = h_norm.shape[0]
    h_q = int(cfg["num_attention_heads"])
    h_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    rotary_dim = int(head_dim * float(cfg["partial_rotary_factor"]))
    rope_theta = float(cfg["rope_theta"])

    q_full = _linear(h_norm, _decode_weight(w, "self_attn.q_proj.weight"))
    k_new = _linear(h_norm, _decode_weight(w, "self_attn.k_proj.weight"))
    v_new = _linear(h_norm, _decode_weight(w, "self_attn.v_proj.weight"))
    q_full_view = q_full.view(bsz, 1, h_q, head_dim * 2)
    q, gate = q_full_view.chunk(2, dim=-1)
    q = q.transpose(1, 2)
    gate = gate.transpose(1, 2)
    k_new = k_new.view(bsz, 1, h_kv, head_dim).transpose(1, 2)
    v_new = v_new.view(bsz, 1, h_kv, head_dim).transpose(1, 2)
    cos, sin = _build_rope_cos_sin(pos_tensor, rotary_dim, rope_theta, h_norm.device, h_norm.dtype)
    q_rope, k_rope = _qk_norm_rope_pair_decode(
        q,
        k_new,
        w["self_attn.q_norm.weight"],
        w["self_attn.k_norm.weight"],
        cos,
        sin,
        rotary_dim,
    )
    new_total = int(state.seq_len) + 1
    return {
        "k_cache": k_cache,
        "v_cache": v_cache,
        "h_q": h_q,
        "h_kv": h_kv,
        "head_dim": head_dim,
        "q": q,
        "gate": gate,
        "k_new": k_new,
        "v_new": v_new,
        "q_rope": q_rope,
        "k_rope": k_rope,
        "k_used": k_cache[:, :, :new_total, :],
        "v_used": v_cache[:, :, :new_total, :],
    }


def _profile_layer(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    h_in: torch.Tensor,
    pos_tensor: torch.Tensor,
    layer_idx: int,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    h_norm = _rms_norm(h_in, w["input_layernorm.weight"])
    mid = _full_attn_mid(h_norm, pos_tensor, w, cfg, state, layer_idx)

    def q_proj():
        return _linear(h_norm, _decode_weight(w, "self_attn.q_proj.weight"))

    def k_proj():
        return _linear(h_norm, _decode_weight(w, "self_attn.k_proj.weight"))

    def v_proj():
        return _linear(h_norm, _decode_weight(w, "self_attn.v_proj.weight"))

    def qk_norm_rope():
        cos, sin = _build_rope_cos_sin(
            pos_tensor,
            int(mid["head_dim"] * float(cfg["partial_rotary_factor"])),
            float(cfg["rope_theta"]),
            h_norm.device,
            h_norm.dtype,
        )
        return _qk_norm_rope_pair_decode(
            mid["q"],
            mid["k_new"],
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
            cos,
            sin,
            int(mid["head_dim"] * float(cfg["partial_rotary_factor"])),
        )

    def cache_write():
        mid["k_cache"][:, :, state.seq_len : state.seq_len + 1, :] = mid["k_rope"]
        mid["v_cache"][:, :, state.seq_len : state.seq_len + 1, :] = mid["v_new"]
        return mid["k_cache"], mid["v_cache"]

    def sdpa_gqa():
        return F.scaled_dot_product_attention(
            mid["q_rope"],
            mid["k_used"],
            mid["v_used"],
            is_causal=False,
            enable_gqa=(mid["h_kv"] != mid["h_q"]),
        )

    attn_core = sdpa_gqa()

    def output_gate():
        return attn_core * torch.sigmoid(mid["gate"].float()).to(attn_core.dtype)

    gated = output_gate()

    def o_proj():
        x = gated.transpose(1, 2).contiguous().view(h_norm.shape[0], 1, mid["h_q"] * mid["head_dim"])
        return _linear(x, _decode_weight(w, "self_attn.o_proj.weight"))

    def full_attn():
        return decode_full_attn(
            h_norm,
            pos_tensor,
            w,
            cfg,
            mid["k_cache"],
            mid["v_cache"],
            cached_seq_len=state.seq_len,
        )

    attn_out = decode_full_attn(
        h_norm,
        pos_tensor,
        w,
        cfg,
        mid["k_cache"],
        mid["v_cache"],
        cached_seq_len=state.seq_len,
    )
    h_after_attn = h_in + attn_out
    h_moe_norm = _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])

    def post_norm():
        return _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])

    def moe():
        return runner.decode_moe_fn(h_moe_norm, w, cfg)

    def full_layer():
        return runner._decode_layer_fast(h_in, pos_tensor, state, layer_idx)

    segments = {
        "input_rmsnorm": lambda: _rms_norm(h_in, w["input_layernorm.weight"]),
        "attn.q_proj": q_proj,
        "attn.k_proj": k_proj,
        "attn.v_proj": v_proj,
        "attn.qk_norm_rope": qk_norm_rope,
        "attn.cache_write": cache_write,
        "attn.sdpa_gqa": sdpa_gqa,
        "attn.output_gate": output_gate,
        "attn.o_proj": o_proj,
        "attn.full_decode": full_attn,
        "post_attn_rmsnorm": post_norm,
        "moe.full": moe,
        "layer.full_decode": full_layer,
    }
    timings = {name: _bench(fn, warmup=warmup, iters=iters) for name, fn in segments.items()}
    attn_parts = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.qk_norm_rope",
        "attn.cache_write",
        "attn.sdpa_gqa",
        "attn.output_gate",
        "attn.o_proj",
    ]
    return {
        "layer": layer_idx,
        "seq_len": int(state.seq_len),
        "segment_ms": timings,
        "attn_parts_sum_ms": sum(float(timings[key]) for key in attn_parts),
        "top_segments": sorted(
            [{"segment": key, "ms": value} for key, value in timings.items() if key != "layer.full_decode"],
            key=lambda row: float(row["ms"]),
            reverse=True,
        )[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=True)
    token_id, state = _prefill_and_first_token(runner, args.prompt, use_chat_template=args.use_chat_template)
    h_in, pos_tensor = _hidden_before_layer(runner, state, token_id, args.layer)
    prof = _profile_layer(
        runner,
        state,
        h_in,
        pos_tensor,
        args.layer,
        warmup=args.warmup,
        iters=args.iters,
    )
    result = {
        "schema_version": "lynn-engine-p27-full-layer-configd-segment-profile-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "env": _env_snapshot(),
        "layer_profile": prof,
        "decision": (
            "Use this to decide whether full-attention eager cost is attention-core, "
            "packed MoE, or residual layer orchestration."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "layer": args.layer,
        "segment_ms": prof["segment_ms"],
        "top_segments": prof["top_segments"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
