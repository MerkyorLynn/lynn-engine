#!/usr/bin/env python3
"""P123: full-attention strict segment + RoPE cache probe.

Stream B Task 3 from ``docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md``.

Measures per-segment latency on one full-attention decode step, then re-runs
with the RoPE cache toggle flipped, so the refactor of the cache into an owned
module can be validated against this baseline.

Output: JSON with per-segment ms and a strict numerical-parity comparison
between cache-on vs cache-off paths (cosine, max abs diff). The probe is
read-only — it does not change runner defaults.

Companion to P10C (linear-attention segment probe) and P26 (whole-decode phase
profile). Use the numbers here to confirm whether the documented ~2.72 ms/token
full-attention budget is split as expected before promoting any cache refactor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import (  # noqa: E402
    _apply_partial_rope,
    _build_rope_cos_sin,
    _build_rope_cos_sin_cached,
    _decode_weight,
    _linear,
    _qk_norm_rope_pair_decode,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _bench(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Mean per-call latency in ms (CUDA event timed)."""
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


def _first_full_attn_layer() -> int:
    for i, t in enumerate(LAYER_TYPES):
        if t == "full_attention":
            return i
    raise RuntimeError("no full_attention layer in LAYER_TYPES")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float((af @ bf) / denom)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _prefill(runner: LynnIncrementalRunner, prompt: str, max_seq_len: int):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(
        batch=1, max_seq_len=max_seq_len, device=runner.device, dtype=runner.dtype
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(
            h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i
        )
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), state, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="-1 = first full_attention layer")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument(
        "--prompt",
        default="请连续输出一段关于 MoE 推理优化、NVFP4、CUDA graph 和工具调用服务化的中文技术说明。",
    )
    args = ap.parse_args()

    if args.layer < 0:
        args.layer = _first_full_attn_layer()
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(
            f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention"
        )

    runner = LynnIncrementalRunner(
        args.model, device="cuda", dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len, verbose=False,
    )
    next_id, state, ids = _prefill(runner, args.prompt, args.max_seq_len)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_new = _rms_norm(h_seed, w["input_layernorm.weight"])

    B = h_new.shape[0]
    H_Q = cfg["num_attention_heads"]
    H_KV = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    rope_theta = cfg["rope_theta"]
    rotary_dim = int(head_dim * cfg["partial_rotary_factor"])
    pos_tensor = torch.tensor([[state.seq_len]], device=runner.device, dtype=torch.long)

    fused_qkv_available = (
        os.environ.get("LYNN_FULL_ATTN_QKV_FUSED", "0") == "1"
        and "self_attn._qkv_proj.weight" in w
    )

    # ─── per-segment closures (mirror engine/incremental_decode.decode_full_attn) ───
    def qkv_proj_separate():
        q_full = _linear(h_new, _decode_weight(w, "self_attn.q_proj.weight"))
        k_new = _linear(h_new, _decode_weight(w, "self_attn.k_proj.weight"))
        v_new = _linear(h_new, _decode_weight(w, "self_attn.v_proj.weight"))
        return q_full, k_new, v_new

    def qkv_proj_fused():
        if not fused_qkv_available:
            return qkv_proj_separate()
        q_out = int(w["self_attn.q_proj.weight"].shape[0])
        k_out = int(w["self_attn.k_proj.weight"].shape[0])
        v_out = int(w["self_attn.v_proj.weight"].shape[0])
        qkv = _linear(h_new, w["self_attn._qkv_proj.weight"])
        return qkv.split((q_out, k_out, v_out), dim=-1)

    q_full, k_new, v_new = qkv_proj_separate()

    def reshape_chunk_gate(q_full=q_full, k_new=k_new, v_new=v_new):
        q_view = q_full.view(B, 1, H_Q, head_dim * 2)
        q, gate = q_view.chunk(2, dim=-1)
        q = q.transpose(1, 2)
        gate = gate.transpose(1, 2)
        k = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
        v = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
        return q, gate, k, v

    q, gate, k_decoded, v_decoded = reshape_chunk_gate()

    def rope_no_cache():
        return _build_rope_cos_sin(pos_tensor, rotary_dim, rope_theta, runner.device, runner.dtype)

    def rope_cached():
        return _build_rope_cos_sin_cached(
            pos_tensor, rotary_dim, rope_theta, runner.device, runner.dtype
        )

    cos_uc, sin_uc = rope_no_cache()
    cos_cc, sin_cc = rope_cached()
    parity_cos_cosine = _cosine(cos_uc, cos_cc)
    parity_cos_maxabs = _max_abs(cos_uc, cos_cc)
    parity_sin_cosine = _cosine(sin_uc, sin_cc)
    parity_sin_maxabs = _max_abs(sin_uc, sin_cc)

    def qk_norm_rope_pair():
        return _qk_norm_rope_pair_decode(
            q,
            k_decoded,
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
            cos_cc,
            sin_cc,
            rotary_dim,
        )

    def qk_norm_rope_split():
        qn = _rms_norm(q, w["self_attn.q_norm.weight"])
        kn = _rms_norm(k_decoded, w["self_attn.k_norm.weight"])
        qn = _apply_partial_rope(qn, cos_cc, sin_cc, rotary_dim)
        kn = _apply_partial_rope(kn, cos_cc, sin_cc, rotary_dim)
        return qn, kn

    q_post, k_post = qk_norm_rope_pair()

    # KV-cache write target. The probe uses a fresh slot just past the prefill
    # tail so SDPA still has a real key/value horizon to attend over.
    cached_seq_len = state.seq_len
    K_full = state.k_cache[args.layer]
    V_full = state.v_cache[args.layer]
    if (cached_seq_len + 1) > K_full.shape[2]:
        raise RuntimeError("max_seq_len too small for probe; increase --max-seq-len")

    def kv_cache_write():
        K_full[:, :, cached_seq_len:cached_seq_len + 1, :] = k_post
        V_full[:, :, cached_seq_len:cached_seq_len + 1, :] = v_decoded

    kv_cache_write()
    new_total = cached_seq_len + 1
    K_used = K_full[:, :, :new_total, :]
    V_used = V_full[:, :, :new_total, :]

    def sdpa_attn():
        if H_KV != H_Q:
            k_attn = K_used.repeat_interleave(H_Q // H_KV, dim=1)
            v_attn = V_used.repeat_interleave(H_Q // H_KV, dim=1)
        else:
            k_attn = K_used
            v_attn = V_used
        return F.scaled_dot_product_attention(q_post, k_attn, v_attn, is_causal=False)

    def manual_gqa_attn():
        group = H_Q // H_KV
        q_grouped = q_post.view(B, H_KV, group, 1, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum("bhgqd,bhkd->bhgqk", q_grouped.float(), K_used.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(V_used.dtype)
        attn_out = torch.einsum("bhgqk,bhkd->bhgqd", probs, V_used)
        return attn_out.reshape(B, H_Q, 1, head_dim)

    sdpa_out = sdpa_attn()

    def gate_apply(sdpa_out=sdpa_out):
        return sdpa_out * torch.sigmoid(gate.float()).to(sdpa_out.dtype)

    gated_out = gate_apply()
    gated_out_view = gated_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)

    def o_proj():
        return _linear(gated_out_view, _decode_weight(w, "self_attn.o_proj.weight"))

    # ─── strict greedy parity between rope-cache on/off ───
    q_pp_uc, k_pp_uc = _qk_norm_rope_pair_decode(
        q,
        k_decoded,
        w["self_attn.q_norm.weight"],
        w["self_attn.k_norm.weight"],
        cos_uc,
        sin_uc,
        rotary_dim,
    )
    qk_parity = {
        "q_cosine": _cosine(q_post, q_pp_uc),
        "q_max_abs": _max_abs(q_post, q_pp_uc),
        "k_cosine": _cosine(k_post, k_pp_uc),
        "k_max_abs": _max_abs(k_post, k_pp_uc),
    }

    # ─── timing pass ───
    timing = {
        "qkv_proj_separate": _bench(qkv_proj_separate, args.warmup, args.iters),
        "reshape_chunk_gate": _bench(
            lambda: reshape_chunk_gate(qkv_proj_separate()[0], *qkv_proj_separate()[1:]),
            args.warmup,
            args.iters,
        ),
        "rope_no_cache": _bench(rope_no_cache, args.warmup, args.iters),
        "rope_cached": _bench(rope_cached, args.warmup, args.iters),
        "qk_norm_rope_pair_fused": _bench(qk_norm_rope_pair, args.warmup, args.iters),
        "qk_norm_rope_split": _bench(qk_norm_rope_split, args.warmup, args.iters),
        "kv_cache_write": _bench(kv_cache_write, args.warmup, args.iters),
        "sdpa_attn": _bench(sdpa_attn, args.warmup, args.iters),
        "manual_gqa_attn": _bench(manual_gqa_attn, max(2, args.warmup // 4), max(20, args.iters // 4)),
        "gate_apply": _bench(gate_apply, args.warmup, args.iters),
        "o_proj_bf16": _bench(o_proj, args.warmup, args.iters),
    }
    if fused_qkv_available:
        timing["qkv_proj_fused"] = _bench(qkv_proj_fused, args.warmup, args.iters)

    # ─── envelope: full strict decode call composed from segments ───
    def full_layer_strict_decode():
        q_full, k_new, v_new = qkv_proj_separate()
        q_view = q_full.view(B, 1, H_Q, head_dim * 2)
        q_local, gate_local = q_view.chunk(2, dim=-1)
        q_local = q_local.transpose(1, 2)
        gate_local = gate_local.transpose(1, 2)
        k_local = k_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
        v_local = v_new.view(B, 1, H_KV, head_dim).transpose(1, 2)
        cos, sin = rope_cached()
        q_local, k_local = _qk_norm_rope_pair_decode(
            q_local,
            k_local,
            w["self_attn.q_norm.weight"],
            w["self_attn.k_norm.weight"],
            cos,
            sin,
            rotary_dim,
        )
        K_full[:, :, cached_seq_len:cached_seq_len + 1, :] = k_local
        V_full[:, :, cached_seq_len:cached_seq_len + 1, :] = v_local
        K_u = K_full[:, :, :new_total, :]
        V_u = V_full[:, :, :new_total, :]
        if H_KV != H_Q:
            k_attn = K_u.repeat_interleave(H_Q // H_KV, dim=1)
            v_attn = V_u.repeat_interleave(H_Q // H_KV, dim=1)
        else:
            k_attn = K_u
            v_attn = V_u
        attn_out = F.scaled_dot_product_attention(q_local, k_attn, v_attn, is_causal=False)
        attn_out = attn_out * torch.sigmoid(gate_local.float()).to(attn_out.dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, H_Q * head_dim)
        return _linear(attn_out, _decode_weight(w, "self_attn.o_proj.weight"))

    timing["full_layer_strict_decode"] = _bench(
        full_layer_strict_decode, max(2, args.warmup // 4), max(20, args.iters // 4)
    )

    # rope cache hit/miss histogram (one warm + one extra call paths) just
    # documents the indices the table will see during a real serving trace.
    rope_cache_seq_max = int(os.environ.get("LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ", "65536"))

    env_snapshot = {
        name: os.environ.get(name)
        for name in (
            "LYNN_FULL_ATTN_ROPE_CACHE",
            "LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ",
            "LYNN_FULL_ATTN_QKV_FUSED",
            "LYNN_FULL_ATTN_DECODE_BACKEND",
            "LYNN_QK_NORM_ROPE_BACKEND",
        )
    }

    top_segments = sorted(
        [
            {"segment": k, "latency_ms": v}
            for k, v in timing.items()
            if k not in ("full_layer_strict_decode",)
        ],
        key=lambda row: row["latency_ms"],
        reverse=True,
    )

    result = {
        "schema_version": "lynn-engine-p123-full-attn-strict-cache-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name("cuda"),
        "config": {
            "H_Q": H_Q,
            "H_KV": H_KV,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "rope_theta": rope_theta,
            "fused_qkv_available": fused_qkv_available,
            "prefill_tokens": int(ids.shape[1]),
            "cached_seq_len": cached_seq_len,
            "rope_cache_table_max_seq": rope_cache_seq_max,
        },
        "env": env_snapshot,
        "timing_ms": timing,
        "top_segments": top_segments,
        "rope_cache_parity": {
            "cos_cosine": parity_cos_cosine,
            "cos_max_abs": parity_cos_maxabs,
            "sin_cosine": parity_sin_cosine,
            "sin_max_abs": parity_sin_maxabs,
        },
        "qk_post_rope_parity_cache_on_vs_off": qk_parity,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
