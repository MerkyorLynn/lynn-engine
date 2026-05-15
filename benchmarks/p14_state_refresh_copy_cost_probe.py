#!/usr/bin/env python3
"""P14: state-refresh copy cost probe for reusable graph slots.

Future graph windows are not generally safe. A safer design is a reusable
current-position graph with graph-owned state buffers:

  real state -> graph state -> replay -> real state

This probe measures the copy cost before implementing the full reusable graph
slot. If state refresh is slower than graph capture, this route is not useful.
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


def _prefill(runner: LynnIncrementalRunner, prompt: str) -> tuple[int, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), state


def _decode_prefix(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    first_id: int,
    prefix_new: int,
) -> int:
    token_id = int(first_id)
    token = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    for _ in range(prefix_new):
        token.fill_(token_id)
        pos_tensor.fill_(int(state.seq_len))
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        state.seq_len += 1
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits = runner._lm_head_logits(h_final)
        token_id = int(logits[0].argmax().item())
    return token_id


def _copy_state(dst: LynnInferenceState, src: LynnInferenceState) -> None:
    dst.seq_len = int(src.seq_len)
    for i, (src_k, src_v) in src.kv_cache.items():
        dst_k, dst_v = dst.kv_cache[i]
        dst_k.copy_(src_k)
        dst_v.copy_(src_v)
    for i, t in src.recurrent_state.items():
        dst.recurrent_state[i].copy_(t)
    for i, t in src.conv_state.items():
        dst.conv_state[i].copy_(t)


def _bench_copy(dst: LynnInferenceState, src: LynnInferenceState, iters: int) -> dict[str, float]:
    # Warmup.
    _copy_state(dst, src)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _copy_state(dst, src)
    end.record()
    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    return {
        "iters": iters,
        "total_ms": total_ms,
        "avg_ms": total_ms / iters,
        "copies_per_second": 1000.0 / (total_ms / iters),
    }


def _state_bytes(state: LynnInferenceState) -> dict[str, Any]:
    kv_bytes = sum((k.numel() * k.element_size()) + (v.numel() * v.element_size()) for k, v in state.kv_cache.values())
    recurrent_bytes = sum(t.numel() * t.element_size() for t in state.recurrent_state.values())
    conv_bytes = sum(t.numel() * t.element_size() for t in state.conv_state.values())
    return {
        "kv_gib": kv_bytes / (1024**3),
        "recurrent_gib": recurrent_bytes / (1024**3),
        "conv_gib": conv_bytes / (1024**3),
        "total_gib": (kv_bytes + recurrent_bytes + conv_bytes) / (1024**3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--prefix-new", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    first_id, real_state = _prefill(runner, args.prompt)
    token_id = _decode_prefix(runner, real_state, first_id, args.prefix_new)
    graph_state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    forward = _bench_copy(graph_state, real_state, args.iters)
    backward = _bench_copy(real_state, graph_state, args.iters)
    result = {
        "schema_version": "lynn-engine-p14-state-refresh-copy-cost-probe-v1",
        "model": args.model,
        "prompt": args.prompt,
        "prefix_new": args.prefix_new,
        "seq_len": int(real_state.seq_len),
        "next_token_id": token_id,
        "state_bytes": _state_bytes(real_state),
        "copy_real_to_graph": forward,
        "copy_graph_to_real": backward,
        "roundtrip_avg_ms": forward["avg_ms"] + backward["avg_ms"],
        "roundtrip_tps_equivalent": 1000.0 / (forward["avg_ms"] + backward["avg_ms"]),
        "note": "Measures full-state copy cost; later probes should restrict copies to active prefix/state slices.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
