#!/usr/bin/env python3
"""P6-M: CUDA graph upper-bound probe for one resident decode token.

This is a performance ceiling probe. It replays the same static token/state
shape to estimate how much Python + kernel launch overhead remains after P6-H/K.
Correctness serving will need input/state buffer discipline, but this answers
whether CUDA graphs are worth productizing for the resident decode loop.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _prepare_decode_moe_fast_layout(layer_weights: list[dict[str, Any]], layer_cfgs: list[dict[str, Any]]) -> None:
    from triton_kernels.moe_expert_ffn import stack_expert_weights

    for w, cfg in zip(layer_weights, layer_cfgs):
        if "mlp.experts._gate_stacked" in w:
            continue
        if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
            gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            w["mlp.experts._gate_stacked"] = gate_stacked
            w["mlp.experts._up_stacked"] = up_stacked
            w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
            cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
        else:
            stack_expert_weights(w, num_experts=cfg["num_experts"])


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


def _bench_eager(fn, warmup: int, iters: int) -> float:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    requested_moe = os.environ.get("LYNN_MOE_IMPL", "optimized")
    if requested_moe == "triton":
        os.environ["LYNN_MOE_IMPL"] = "optimized"
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    os.environ["LYNN_MOE_IMPL"] = requested_moe
    if requested_moe in {"triton", "indexed_bmm"}:
        _prepare_decode_moe_fast_layout(runner.layer_weights, runner.layer_cfgs)

    next_id, state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    pos_id = int(state.seq_len)
    pos_tensor = torch.tensor([[pos_id]], device=runner.device, dtype=torch.long)
    out_buf = torch.empty((1, 1, runner.cfg["hidden_size"]), device=runner.device, dtype=runner.dtype)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)

    def full_token_static():
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
        out_buf.copy_(h)
        logits_buf.copy_(logits)
        return logits_buf

    eager_ms = _bench_eager(full_token_static, args.warmup, max(10, args.iters // 5))

    # Warm allocations before capture.
    for _ in range(5):
        full_token_static()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full_token_static()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph_ms = float(start.elapsed_time(end) / args.iters)

    result = {
        "schema_version": "lynn-engine-p6m-cuda-graph-full-token-probe-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name("cuda"),
        "moe_impl": requested_moe,
        "linear_recurrent_backend": os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch"),
        "qk_norm_rope_backend": os.environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch"),
        "eager_ms": eager_ms,
        "eager_tps": 1000.0 / eager_ms,
        "cuda_graph_ms": graph_ms,
        "cuda_graph_tps": 1000.0 / graph_ms,
        "speedup": eager_ms / graph_ms,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
