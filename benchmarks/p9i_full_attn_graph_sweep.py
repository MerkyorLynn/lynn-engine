#!/usr/bin/env python3
"""P9-I: full-attention fixed-position CUDA graph sweep.

P9-H proved one layer/position can replay much faster than eager. This sweep
keeps the same strict parity check but runs multiple full-attention layers and
prompt lengths in one model load. It reports both conservative timings with KV
restore in the loop and replay-only graph timings for the runtime ceiling.
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


DEFAULT_PROMPTS = [
    "Explain MoE active parameters in one sentence.",
    "Return only a compact JSON object with city and status for Shanghai weather.",
    (
        "Write five concise bullet points comparing W4A8 and W4A4 quantization "
        "for a deployment engineer, focusing on latency, memory, and quality risk."
    ),
]


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
    prompt_index: int,
    prompt: str,
    layer: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if LAYER_TYPES[layer] != "full_attention":
        raise ValueError(f"layer {layer} is {LAYER_TYPES[layer]!r}, expected full_attention")

    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    write_pos = int(state.seq_len)
    out_buf = torch.empty_like(h_seed)
    kv_snap = _snapshot_kv(state, layer)

    def eager_layer():
        return _decode_layer(
            h_seed,
            pos_tensor,
            LAYER_TYPES[layer],
            runner.layer_weights[layer],
            runner.layer_cfgs[layer],
            state,
            layer,
        )

    _restore_kv(state, layer, kv_snap)
    eager_out = eager_layer()
    eager_kv = _snapshot_kv(state, layer)
    eager_write = _kv_write_slice(state, layer, write_pos)
    _restore_kv(state, layer, kv_snap)

    def graph_body():
        out = _decode_layer(
            h_seed,
            pos_tensor,
            LAYER_TYPES[layer],
            runner.layer_weights[layer],
            runner.layer_cfgs[layer],
            state,
            layer,
        )
        out_buf.copy_(out)

    for _ in range(10):
        graph_body()
        _restore_kv(state, layer, kv_snap)
    torch.cuda.synchronize()
    _restore_kv(state, layer, kv_snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_body()
    torch.cuda.synchronize()
    _restore_kv(state, layer, kv_snap)
    graph.replay()
    torch.cuda.synchronize()
    graph_out = out_buf.clone()
    graph_kv = _snapshot_kv(state, layer)
    graph_write = _kv_write_slice(state, layer, write_pos)

    output_diff = _cmp(graph_out, eager_out)
    kv_write_diff = {
        "position": write_pos,
        "k": _cmp(graph_write[0], eager_write[0]),
        "v": _cmp(graph_write[1], eager_write[1]),
    }

    _restore_kv(state, layer, kv_snap)
    eager_restore_ms = _bench(
        lambda: (_restore_kv(state, layer, kv_snap), eager_layer())[1],
        warmup,
        max(20, iters // 5),
    )
    _restore_kv(state, layer, kv_snap)
    graph_restore_ms = _bench(
        lambda: (_restore_kv(state, layer, kv_snap), graph.replay())[1],
        warmup,
        iters,
    )
    _restore_kv(state, layer, kv_snap)
    eager_no_restore_ms = _bench(eager_layer, warmup, max(20, iters // 5))
    _restore_kv(state, layer, kv_snap)
    graph_replay_ms = _bench(graph.replay, warmup, iters)

    _restore_kv(state, layer, kv_snap)
    del graph
    torch.cuda.empty_cache()

    return {
        "prompt_index": prompt_index,
        "prompt": prompt,
        "position": write_pos,
        "layer": layer,
        "eager_restore_ms": eager_restore_ms,
        "graph_restore_ms": graph_restore_ms,
        "speedup_restore": eager_restore_ms / graph_restore_ms,
        "eager_no_restore_ms": eager_no_restore_ms,
        "graph_replay_ms": graph_replay_ms,
        "speedup_replay": eager_no_restore_ms / graph_replay_ms,
        "output_diff": output_diff,
        "kv_write_slice_diff": kv_write_diff,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in records]

    max_output = max(float(r["output_diff"]["max_abs"]) for r in records)
    max_k = max(float(r["kv_write_slice_diff"]["k"]["max_abs"]) for r in records)
    max_v = max(float(r["kv_write_slice_diff"]["v"]["max_abs"]) for r in records)
    return {
        "case_count": len(records),
        "layers": sorted({int(r["layer"]) for r in records}),
        "positions": sorted({int(r["position"]) for r in records}),
        "eager_restore_ms_mean": statistics.fmean(vals("eager_restore_ms")),
        "graph_restore_ms_mean": statistics.fmean(vals("graph_restore_ms")),
        "speedup_restore_min": min(vals("speedup_restore")),
        "speedup_restore_mean": statistics.fmean(vals("speedup_restore")),
        "speedup_restore_max": max(vals("speedup_restore")),
        "eager_no_restore_ms_mean": statistics.fmean(vals("eager_no_restore_ms")),
        "graph_replay_ms_mean": statistics.fmean(vals("graph_replay_ms")),
        "speedup_replay_min": min(vals("speedup_replay")),
        "speedup_replay_mean": statistics.fmean(vals("speedup_replay")),
        "speedup_replay_max": max(vals("speedup_replay")),
        "max_output_diff": max_output,
        "max_kv_write_k_diff": max_k,
        "max_kv_write_v_diff": max_v,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", dest="layers", type=int, action="append")
    ap.add_argument("--prompt", dest="prompts", action="append")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    layers = args.layers or [3, 15, 31, 39]
    prompts = args.prompts or DEFAULT_PROMPTS

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        next_id, state = _prefill(runner, prompt)
        for layer in layers:
            records.append(
                _probe_layer(
                    runner,
                    state,
                    next_id,
                    prompt_index,
                    prompt,
                    layer,
                    args.warmup,
                    args.iters,
                )
            )

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p9i-full-attn-graph-sweep-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name("cuda"),
        "qk_norm_rope_backend": __import__("os").environ.get("LYNN_QK_NORM_ROPE_BACKEND", "torch"),
        "summary": _summarize(records),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
