#!/usr/bin/env python3
"""P6-H: compare MoE decode expert kernels on a real resident layer input."""
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
from engine.incremental_decode import decode_full_attn  # noqa: E402
from engine.moe_optimized import moe_forward_decode_optimized  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.moe_expert_ffn import (  # noqa: E402
    HAS_TRITON,
    moe_forward_decode_indexed_bmm,
    moe_forward_decode_triton,
    stack_expert_weights,
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
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=80)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")
    next_id, state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h0 = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]

    # Build a realistic post-attention MoE input for this layer.
    h_attn_norm = _rms_norm(h0, w["input_layernorm.weight"])
    attn = decode_full_attn(
        h_attn_norm,
        state.seq_len,
        w,
        cfg,
        state.kv_cache[args.layer][0],
        state.kv_cache[args.layer][1],
        cached_seq_len=state.seq_len,
    )
    h_moe = _rms_norm(h0 + attn, w["post_attention_layernorm.weight"])

    # Stack once; this is the intended resident serving layout for the fast paths.
    # Variable-pruned 27B already stores experts as a fused gate_up/down layout.
    if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
        gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
        w["mlp.experts._gate_stacked"] = gate_stacked.contiguous()
        w["mlp.experts._up_stacked"] = up_stacked.contiguous()
        w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"].contiguous()
        cfg = dict(cfg)
        cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
    else:
        stack_expert_weights(w, num_experts=cfg["num_experts"])

    ref = moe_forward_decode_optimized(h_moe, w, cfg)
    indexed = moe_forward_decode_indexed_bmm(h_moe, w, cfg)
    triton_out = moe_forward_decode_triton(h_moe, w, cfg) if HAS_TRITON else None

    cases = {
        "optimized_active_loop": lambda: moe_forward_decode_optimized(h_moe, w, cfg),
        "indexed_bmm_stacked": lambda: moe_forward_decode_indexed_bmm(h_moe, w, cfg),
    }
    if HAS_TRITON:
        cases["triton_two_kernel"] = lambda: moe_forward_decode_triton(h_moe, w, cfg)

    latency_ms = {name: _bench(fn, warmup=args.warmup, iters=args.iters) for name, fn in cases.items()}
    diff = {
        "indexed_vs_optimized_max_abs": float((indexed.float() - ref.float()).abs().max().item()),
        "indexed_vs_optimized_cosine": float(F.cosine_similarity(indexed.float().flatten(), ref.float().flatten(), dim=0).item()),
    }
    if triton_out is not None:
        diff.update(
            {
                "triton_vs_optimized_max_abs": float((triton_out.float() - ref.float()).abs().max().item()),
                "triton_vs_optimized_cosine": float(F.cosine_similarity(triton_out.float().flatten(), ref.float().flatten(), dim=0).item()),
            }
        )
    result = {
        "schema_version": "lynn-engine-p6h-moe-expert-kernel-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name(args.device),
        "has_triton": HAS_TRITON,
        "load_seconds": runner.load_seconds,
        "latency_ms": latency_ms,
        "diff": diff,
        "speedup_vs_optimized": {
            name: latency_ms["optimized_active_loop"] / value for name, value in latency_ms.items() if value > 0
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
