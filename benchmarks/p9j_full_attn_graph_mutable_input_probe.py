#!/usr/bin/env python3
"""P9-J: full-attention graph mutable-input probe.

P9-I measured fixed-position replay speed. Serving also needs the captured graph
to consume new hidden values through a stable input buffer. This probe captures
one full-attention layer graph, then replays it with two different input tensors
at the same position and compares both paths with eager.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _decode_layer, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
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


def _snapshot_kv(state: LynnInferenceState, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = state.kv_cache[layer]
    return k.clone(), v.clone()


def _restore_kv(state: LynnInferenceState, layer: int, snap: tuple[torch.Tensor, torch.Tensor]) -> None:
    k, v = state.kv_cache[layer]
    k.copy_(snap[0])
    v.copy_(snap[1])


def _kv_write_slice(state: LynnInferenceState, layer: int, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    k, v = state.kv_cache[layer]
    return k[:, :, pos : pos + 1, :].clone(), v[:, :, pos : pos + 1, :].clone()


def _probe_layer(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    next_id: int,
    layer: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if LAYER_TYPES[layer] != "full_attention":
        raise ValueError(f"layer {layer} is {LAYER_TYPES[layer]!r}, expected full_attention")

    vocab = int(runner.outside["model.language_model.embed_tokens.weight"].shape[0])
    alt_id = (int(next_id) + 1009) % vocab
    token_a = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    token_b = torch.tensor([[alt_id]], device=runner.device, dtype=torch.long)
    h_a = F.embedding(token_a, runner.outside["model.language_model.embed_tokens.weight"])
    h_b = F.embedding(token_b, runner.outside["model.language_model.embed_tokens.weight"])
    input_buf = torch.empty_like(h_a)
    output_buf = torch.empty_like(h_a)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    write_pos = int(state.seq_len)
    kv_snap = _snapshot_kv(state, layer)

    def eager_layer(h: torch.Tensor):
        return _decode_layer(
            h,
            pos_tensor,
            LAYER_TYPES[layer],
            runner.layer_weights[layer],
            runner.layer_cfgs[layer],
            state,
            layer,
        )

    def eager_case(h: torch.Tensor):
        _restore_kv(state, layer, kv_snap)
        out = eager_layer(h)
        write = _kv_write_slice(state, layer, write_pos)
        return out.clone(), write

    eager_a, eager_write_a = eager_case(h_a)
    eager_b, eager_write_b = eager_case(h_b)

    def graph_body():
        out = eager_layer(input_buf)
        output_buf.copy_(out)

    input_buf.copy_(h_a)
    for _ in range(10):
        _restore_kv(state, layer, kv_snap)
        graph_body()
    torch.cuda.synchronize()
    _restore_kv(state, layer, kv_snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_body()
    torch.cuda.synchronize()

    def graph_case(h: torch.Tensor):
        _restore_kv(state, layer, kv_snap)
        input_buf.copy_(h)
        graph.replay()
        torch.cuda.synchronize()
        out = output_buf.clone()
        write = _kv_write_slice(state, layer, write_pos)
        return out, write

    graph_a, graph_write_a = graph_case(h_a)
    graph_b, graph_write_b = graph_case(h_b)

    _restore_kv(state, layer, kv_snap)
    input_buf.copy_(h_a)
    replay_ms = _bench(graph.replay, warmup, iters)
    _restore_kv(state, layer, kv_snap)
    input_buf.copy_(h_a)
    eager_ms = _bench(lambda: eager_layer(input_buf), warmup, max(20, iters // 5))
    _restore_kv(state, layer, kv_snap)

    out_delta = _cmp(graph_b, graph_a)
    del graph
    torch.cuda.empty_cache()

    return {
        "layer": layer,
        "position": write_pos,
        "token_a": int(next_id),
        "token_b": int(alt_id),
        "eager_ms": eager_ms,
        "graph_replay_ms": replay_ms,
        "speedup": eager_ms / replay_ms,
        "case_a_output_diff": _cmp(graph_a, eager_a),
        "case_b_output_diff": _cmp(graph_b, eager_b),
        "case_a_kv_write_diff": {
            "k": _cmp(graph_write_a[0], eager_write_a[0]),
            "v": _cmp(graph_write_a[1], eager_write_a[1]),
        },
        "case_b_kv_write_diff": {
            "k": _cmp(graph_write_b[0], eager_write_b[0]),
            "v": _cmp(graph_write_b[1], eager_write_b[1]),
        },
        "graph_output_delta_a_to_b": out_delta,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(records),
        "layers": [int(r["layer"]) for r in records],
        "position": int(records[0]["position"]) if records else None,
        "eager_ms_mean": statistics.fmean(float(r["eager_ms"]) for r in records),
        "graph_replay_ms_mean": statistics.fmean(float(r["graph_replay_ms"]) for r in records),
        "speedup_min": min(float(r["speedup"]) for r in records),
        "speedup_mean": statistics.fmean(float(r["speedup"]) for r in records),
        "speedup_max": max(float(r["speedup"]) for r in records),
        "max_case_a_output_diff": max(float(r["case_a_output_diff"]["max_abs"]) for r in records),
        "max_case_b_output_diff": max(float(r["case_b_output_diff"]["max_abs"]) for r in records),
        "max_case_a_kv_k_diff": max(float(r["case_a_kv_write_diff"]["k"]["max_abs"]) for r in records),
        "max_case_b_kv_k_diff": max(float(r["case_b_kv_write_diff"]["k"]["max_abs"]) for r in records),
        "min_graph_output_delta_rel_l2": min(float(r["graph_output_delta_a_to_b"]["rel_l2"]) for r in records),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", dest="layers", type=int, action="append")
    ap.add_argument("--prompt", default="Explain MoE active parameters in one sentence.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    layers = args.layers or [3, 15, 31, 39]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    next_id, state = _prefill(runner, args.prompt)
    records = [_probe_layer(runner, state, next_id, layer, args.warmup, args.iters) for layer in layers]

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p9j-full-attn-graph-mutable-input-probe-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name("cuda"),
        "qk_norm_rope_backend": __import__("os").environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch"),
        "prompt": args.prompt,
        "summary": _summarize(records),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
