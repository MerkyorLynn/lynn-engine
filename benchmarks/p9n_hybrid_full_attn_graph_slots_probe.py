#!/usr/bin/env python3
"""P9-N: hybrid token probe with linear blocks + pre-captured full-attn slots.

This combines the two proven graphable pieces:
  - reusable 3-layer linear-attention block graphs
  - pre-captured full-attention layer graph slots

It measures one full decode token and checks logits/KV parity against eager.
This is the closest benchmark shape to the next resident-runner implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import FULL_ATTN_INDICES, LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cos = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / denom.item()),
        "cosine": float(cos.item()),
    }


def _snapshot_state(state: LynnInferenceState) -> dict[str, Any]:
    return {
        "seq_len": state.seq_len,
        "kv": {i: (kv[0].clone(), kv[1].clone()) for i, kv in state.kv_cache.items()},
        "recurrent": {i: t.clone() for i, t in state.recurrent_state.items()},
        "conv": {i: t.clone() for i, t in state.conv_state.items()},
    }


def _restore_state(state: LynnInferenceState, snap: dict[str, Any]) -> None:
    state.seq_len = int(snap["seq_len"])
    for i, (k, v) in snap["kv"].items():
        state.kv_cache[i][0].copy_(k)
        state.kv_cache[i][1].copy_(v)
    for i, t in snap["recurrent"].items():
        state.recurrent_state[i].copy_(t)
    for i, t in snap["conv"].items():
        state.conv_state[i].copy_(t)


def _prefill_into_state(runner: LynnIncrementalRunner, state: LynnInferenceState, prompt: str) -> tuple[int, torch.Tensor]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), ids


def _capture_full_attn_slots(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    position: int,
) -> dict[int, dict[str, Any]]:
    slots: dict[int, dict[str, Any]] = {}
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    for layer in FULL_ATTN_INDICES:
        input_buf = torch.zeros((1, 1, int(runner.cfg["hidden_size"])), device=runner.device, dtype=runner.dtype)
        output_buf = torch.empty_like(input_buf)

        def graph_body(layer=layer, input_buf=input_buf, output_buf=output_buf) -> None:
            state.seq_len = position
            out = _decode_layer(
                input_buf,
                pos_tensor,
                LAYER_TYPES[layer],
                runner.layer_weights[layer],
                runner.layer_cfgs[layer],
                state,
                layer,
            )
            output_buf.copy_(out)

        graph_body()
        torch.cuda.synchronize()
        state.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_body()
        torch.cuda.synchronize()
        state.reset()
        slots[layer] = {"input": input_buf, "output": output_buf, "graph": graph}
    return slots


def _bench(fn, warmup: int, iters: int) -> float:
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
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    # Tokenize first so graph slots know the first decode position. Capture
    # before real KV values exist, then reset and run prefill into the same state.
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    decode_position = int(ids.shape[1])
    full_slots = _capture_full_attn_slots(runner, state, decode_position)
    next_id, _ = _prefill_into_state(runner, state, args.prompt)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    linear_blocks, linear_capture_s = runner._capture_linear_block_graphs(state, h_seed, pos_tensor)
    base = _snapshot_state(state)

    def eager_token() -> torch.Tensor:
        _restore_state(state, base)
        h = h_seed
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        state.seq_len = decode_position + 1
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    def graph_token_body() -> torch.Tensor:
        h = h_seed
        for bi, block in enumerate(linear_blocks):
            block["input"].copy_(h)
            block["graph"].replay()
            h = block["output"]
            full_layer = bi * 4 + 3
            slot = full_slots[full_layer]
            slot["input"].copy_(h)
            slot["graph"].replay()
            h = slot["output"]
        state.seq_len = decode_position + 1
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    def graph_token() -> torch.Tensor:
        _restore_state(state, base)
        return graph_token_body()

    eager_logits = eager_token().clone()
    _restore_state(state, base)
    one_shot_start = torch.cuda.Event(enable_timing=True)
    one_shot_end = torch.cuda.Event(enable_timing=True)
    one_shot_start.record()
    graph_logits = graph_token_body().clone()
    one_shot_end.record()
    torch.cuda.synchronize()
    one_shot_graph_ms = float(one_shot_start.elapsed_time(one_shot_end))
    diff = _cmp(graph_logits, eager_logits)
    eager_ms = _bench(eager_token, max(1, args.warmup), max(10, args.iters // 5))
    graph_ms = _bench(graph_token, args.warmup, args.iters)
    graph_next = int(graph_logits[0].argmax().item())
    eager_next = int(eager_logits[0].argmax().item())
    greedy_pass = graph_next == eager_next
    strict_logit_pass = diff["max_abs"] == 0.0
    result = {
        "schema_version": "lynn-engine-p9n-hybrid-full-attn-graph-slots-probe-v1",
        "model": args.model,
        "decode_position": decode_position,
        "full_attn_graph_slots": len(full_slots),
        "linear_block_graphs": len(linear_blocks),
        "linear_capture_s": linear_capture_s,
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "speedup": eager_ms / graph_ms,
        "graph_tps": 1000.0 / graph_ms,
        "one_shot_graph_ms": one_shot_graph_ms,
        "one_shot_graph_tps": 1000.0 / one_shot_graph_ms,
        "logit_diff": diff,
        "graph_next_id": graph_next,
        "eager_next_id": eager_next,
        "greedy_pass": greedy_pass,
        "strict_logit_pass": strict_logit_pass,
        "pass": greedy_pass,
        "note": "graph_ms includes benchmark state-restore overhead; one_shot_graph_ms is closer to serving hot-path latency.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
