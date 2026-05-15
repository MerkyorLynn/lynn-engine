#!/usr/bin/env python3
"""P6-L: full linear-attention layer profile under current best decode config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.incremental_decode import decode_linear_attn  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.moe_expert_ffn import moe_forward_decode_triton  # noqa: E402


def _bench(fn: Callable[[], Any], warmup: int, iters: int) -> float:
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


def _prepare_triton_moe(w: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
        gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
        w["mlp.experts._gate_stacked"] = gate_stacked
        w["mlp.experts._up_stacked"] = up_stacked
        w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
        cfg = dict(cfg)
        cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "linear_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected linear_attention")
    next_id, state = _prefill(runner, "用一句话解释 MoE active parameters")
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    cfg = _prepare_triton_moe(w, runner.layer_cfgs[args.layer])

    def input_norm():
        return _rms_norm(h0, w["input_layernorm.weight"])

    h_attn_norm = input_norm()

    def linear_core():
        return decode_linear_attn(
            h_attn_norm,
            w,
            state.recurrent_state[args.layer],
            state.conv_state[args.layer],
            recurrent_backend="triton_fused_prepare",
        )[0]

    attn_out = linear_core()
    h_after_attn = h0 + attn_out

    def post_attn_norm():
        return _rms_norm(h_after_attn, w["post_attention_layernorm.weight"])

    h_moe_norm = post_attn_norm()

    def triton_moe():
        return moe_forward_decode_triton(h_moe_norm, w, cfg)

    def full_layer():
        hn = _rms_norm(h0, w["input_layernorm.weight"])
        attn = decode_linear_attn(
            hn,
            w,
            state.recurrent_state[args.layer],
            state.conv_state[args.layer],
            recurrent_backend="triton_fused_prepare",
        )[0]
        h = h0 + attn
        hm = _rms_norm(h, w["post_attention_layernorm.weight"])
        return h + moe_forward_decode_triton(hm, w, cfg)

    segment_ms = {
        "layer.input_rmsnorm": _bench(input_norm, args.warmup, args.iters),
        "linear_attn.core_decode": _bench(linear_core, args.warmup, args.iters),
        "layer.post_attn_rmsnorm": _bench(post_attn_norm, args.warmup, args.iters),
        "moe.triton_full": _bench(triton_moe, args.warmup, args.iters),
        "layer.full_recomposed": _bench(full_layer, max(1, args.warmup // 2), max(5, args.iters // 4)),
    }
    result = {
        "schema_version": "lynn-engine-p6l-linear-layer-profile-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name("cuda"),
        "segment_ms": segment_ms,
        "top_segments": sorted(
            [{"segment": k, "latency_ms": v} for k, v in segment_ms.items() if not k.endswith("recomposed")],
            key=lambda x: x["latency_ms"],
            reverse=True,
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
