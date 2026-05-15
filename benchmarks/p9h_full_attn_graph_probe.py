#!/usr/bin/env python3
"""P9-H: fixed-position full-attention layer CUDA graph probe.

This is a ceiling probe for one full-attention layer. It captures a fixed
position/KV-slice decode and checks graph replay parity against eager. Serving
still needs a graph family or fixed-shape KV strategy before this can be used
for arbitrary decode positions.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")

    next_id, state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    write_pos = int(state.seq_len)
    out_buf = torch.empty_like(h_seed)

    kv_snap = _snapshot_kv(state, args.layer)

    def eager_layer():
        return _decode_layer(
            h_seed,
            pos_tensor,
            LAYER_TYPES[args.layer],
            runner.layer_weights[args.layer],
            runner.layer_cfgs[args.layer],
            state,
            args.layer,
        )

    _restore_kv(state, args.layer, kv_snap)
    eager_out = eager_layer()
    eager_kv = _snapshot_kv(state, args.layer)
    eager_write = _kv_write_slice(state, args.layer, write_pos)

    _restore_kv(state, args.layer, kv_snap)

    def graph_body():
        out = _decode_layer(
            h_seed,
            pos_tensor,
            LAYER_TYPES[args.layer],
            runner.layer_weights[args.layer],
            runner.layer_cfgs[args.layer],
            state,
            args.layer,
        )
        out_buf.copy_(out)

    for _ in range(10):
        graph_body()
        _restore_kv(state, args.layer, kv_snap)
    torch.cuda.synchronize()
    _restore_kv(state, args.layer, kv_snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_body()
    torch.cuda.synchronize()
    _restore_kv(state, args.layer, kv_snap)
    graph.replay()
    torch.cuda.synchronize()
    graph_out = out_buf.clone()
    graph_kv = _snapshot_kv(state, args.layer)
    graph_write = _kv_write_slice(state, args.layer, write_pos)

    _restore_kv(state, args.layer, kv_snap)
    eager_ms = _bench(lambda: (_restore_kv(state, args.layer, kv_snap), eager_layer())[1], args.warmup, max(20, args.iters // 5))
    _restore_kv(state, args.layer, kv_snap)
    graph_ms = _bench(lambda: (_restore_kv(state, args.layer, kv_snap), graph.replay())[1], args.warmup, args.iters)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p9h-full-attn-graph-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "device": torch.cuda.get_device_name("cuda"),
        "qk_norm_rope_backend": __import__("os").environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch"),
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "speedup": eager_ms / graph_ms,
        "output_diff": _cmp(graph_out, eager_out),
        "kv_diff": {
            "k": _cmp(graph_kv[0], eager_kv[0]),
            "v": _cmp(graph_kv[1], eager_kv[1]),
        },
        "kv_write_slice_diff": {
            "position": write_pos,
            "k": _cmp(graph_write[0], eager_write[0]),
            "v": _cmp(graph_write[1], eager_write[1]),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
