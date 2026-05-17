#!/usr/bin/env python3
"""P10-C: segment profile for linear-attention core under native fused in-proj."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import _linear, _rms_norm_gated_decode  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    RMS_EPS,
    VALUE_DIM,
    V_PER_K,
)
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.gated_delta import (  # noqa: E402
    recurrent_gated_delta_fused_prepare,
    recurrent_gated_delta_fused_prepare_gqa,
)


def _bench(fn: Callable[[], torch.Tensor | tuple], warmup: int, iters: int) -> float:
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


def _prefill(runner: LynnIncrementalRunner, prompt: str):
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "linear_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected linear_attention")
    next_id, state = _prefill(runner, "用一句话解释 MoE active parameters")
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    h_new = _rms_norm(h0, w["input_layernorm.weight"])
    B = h_new.shape[0]

    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")

    def fused_inproj():
        proj_all = _linear(h_new, w[fused_key])
        return torch.split(
            proj_all,
            [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
            dim=-1,
        )

    mixed_new, z, b, a = fused_inproj()
    mixed_new_t = mixed_new.transpose(1, 2)
    use_gqa_recurrent = (
        V_PER_K > 1
        and os.environ.get("LYNN_LINEAR_ATTN_GQA_RECURRENT", "0") == "1"
    )

    def conv_update():
        conv_input = torch.cat([state.conv_state[args.layer], mixed_new_t], dim=-1)
        out_conv = F.conv1d(
            conv_input,
            w["linear_attn.conv1d.weight"],
            bias=None,
            padding=0,
            groups=mixed_new_t.shape[1],
        )
        return F.silu(out_conv).transpose(1, 2), conv_input[:, :, 1:].contiguous()

    out_conv, _ = conv_update()

    def split_qkv():
        q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)
        return q, k, v

    q, k, v = split_qkv()
    z = z.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b.sigmoid()
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"].float())

    def recurrent():
        if use_gqa_recurrent:
            return recurrent_gated_delta_fused_prepare_gqa(q, k, v, g, beta, state.recurrent_state[args.layer])
        return recurrent_gated_delta_fused_prepare(q, k, v, g, beta, state.recurrent_state[args.layer])

    core_attn_out, _ = recurrent()

    def gated_norm():
        flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
        flat_z = z.reshape(-1, HEAD_V_DIM)
        y = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
        return y.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)

    core_normed = gated_norm()

    def out_proj():
        return _linear(core_normed, w["linear_attn.out_proj.weight"])

    def full_core():
        mixed, z0, b0, a0 = fused_inproj()
        mixed_t = mixed.transpose(1, 2)
        conv_input = torch.cat([state.conv_state[args.layer], mixed_t], dim=-1)
        conv = F.conv1d(
            conv_input,
            w["linear_attn.conv1d.weight"],
            bias=None,
            padding=0,
            groups=mixed_t.shape[1],
        )
        conv = F.silu(conv).transpose(1, 2)
        q0, k0, v0 = torch.split(conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q0 = q0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        k0 = k0.reshape(B, 1, NUM_K_HEADS, HEAD_K_DIM)
        v0 = v0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1 and not use_gqa_recurrent:
            q0 = q0.repeat_interleave(V_PER_K, dim=2)
            k0 = k0.repeat_interleave(V_PER_K, dim=2)
        z0 = z0.reshape(B, 1, NUM_V_HEADS, HEAD_V_DIM)
        beta0 = b0.sigmoid()
        g0 = -w["linear_attn.A_log"].float().exp() * F.softplus(a0.float() + w["linear_attn.dt_bias"].float())
        if use_gqa_recurrent:
            attn, _ = recurrent_gated_delta_fused_prepare_gqa(
                q0, k0, v0, g0, beta0, state.recurrent_state[args.layer]
            )
        else:
            attn, _ = recurrent_gated_delta_fused_prepare(q0, k0, v0, g0, beta0, state.recurrent_state[args.layer])
        flat_x = attn.reshape(-1, HEAD_V_DIM)
        flat_z = z0.reshape(-1, HEAD_V_DIM)
        normed = _rms_norm_gated_decode(flat_x, w["linear_attn.norm.weight"], flat_z)
        normed = normed.reshape(B, 1, NUM_V_HEADS * HEAD_V_DIM)
        return _linear(normed, w["linear_attn.out_proj.weight"])

    timing = {
        "fused_native_fp4_inproj": _bench(fused_inproj, args.warmup, args.iters),
        "conv_update": _bench(conv_update, args.warmup, args.iters),
        "split_qkv_repeat": _bench(split_qkv, args.warmup, args.iters),
        "recurrent_fused_prepare": _bench(recurrent, args.warmup, args.iters),
        "gated_rmsnorm": _bench(gated_norm, args.warmup, args.iters),
        "out_proj_bf16": _bench(out_proj, args.warmup, args.iters),
        "full_core_recomposed": _bench(full_core, max(1, args.warmup // 2), max(10, args.iters // 4)),
    }
    result = {
        "schema_version": "lynn-engine-p10c-linear-attn-core-segment-profile-v1",
        "model": args.model,
        "layer": args.layer,
        "use_gqa_recurrent": use_gqa_recurrent,
        "timing_ms": timing,
        "top_segments": sorted(
            [{"segment": k, "latency_ms": v} for k, v in timing.items() if not k.endswith("recomposed")],
            key=lambda row: row["latency_ms"],
            reverse=True,
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
