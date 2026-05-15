#!/usr/bin/env python3
"""P9-M: pre-captured full-attention graph replay on populated KV.

P9-H proved a full-attention layer graph is exact when captured after prefill at
a fixed position. For serving, we need a stronger property: capture graph slots
before real KV values exist, then let prefill/decode populate the same cache
tensors and replay the pre-captured graph later.

If this passes, full-attention layer graph families can be pre-captured after
tokenization/state allocation instead of paying graph capture on the hot path.
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


def _decode_one(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
    position: int,
) -> torch.Tensor:
    token = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for i in range(runner.n_layers):
        h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len += 1
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--prefix-new", type=int, default=32)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")

    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    base_seq_len = int(ids.shape[1])
    target_position = base_seq_len + args.prefix_new

    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    input_buf = torch.zeros((1, 1, int(runner.cfg["hidden_size"])), device=runner.device, dtype=runner.dtype)
    out_buf = torch.empty_like(input_buf)
    pos_tensor = torch.tensor([[target_position]], device=runner.device, dtype=torch.long)

    def graph_body() -> None:
        state.seq_len = target_position
        out = _decode_layer(
            input_buf,
            pos_tensor,
            LAYER_TYPES[args.layer],
            runner.layer_weights[args.layer],
            runner.layer_cfgs[args.layer],
            state,
            args.layer,
        )
        out_buf.copy_(out)

    # Capture before prompt prefill has populated any real KV values.
    graph_body()
    torch.cuda.synchronize()
    zero_kv_after_warm = _snapshot_kv(state, args.layer)
    _restore_kv(state, args.layer, (torch.zeros_like(zero_kv_after_warm[0]), torch.zeros_like(zero_kv_after_warm[1])))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_body()
    torch.cuda.synchronize()

    # Populate the same state object with real prompt + eager prefix values.
    state.reset()
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(base_seq_len, device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = base_seq_len
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    token_id = int(logits[0].argmax().item())
    generated_prefix: list[int] = []
    for _ in range(args.prefix_new):
        logits = _decode_one(runner, state, token_id, int(state.seq_len))
        token_id = int(logits[0].argmax().item())
        generated_prefix.append(token_id)

    real_kv = _snapshot_kv(state, args.layer)
    h_seed = F.embedding(torch.tensor([[token_id]], device=runner.device, dtype=torch.long), runner.outside["model.language_model.embed_tokens.weight"])
    input_buf.copy_(h_seed)

    _restore_kv(state, args.layer, real_kv)
    eager_out = _decode_layer(
        h_seed,
        pos_tensor,
        LAYER_TYPES[args.layer],
        runner.layer_weights[args.layer],
        runner.layer_cfgs[args.layer],
        state,
        args.layer,
    )
    eager_kv = _snapshot_kv(state, args.layer)
    eager_write = _kv_write_slice(state, args.layer, target_position)

    _restore_kv(state, args.layer, real_kv)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph_ms = float(start.elapsed_time(end))
    graph_out = out_buf.clone()
    graph_kv = _snapshot_kv(state, args.layer)
    graph_write = _kv_write_slice(state, args.layer, target_position)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p9m-precaptured-full-attn-graph-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "base_seq_len": base_seq_len,
        "prefix_new": args.prefix_new,
        "target_position": target_position,
        "input_token_id": token_id,
        "generated_prefix": generated_prefix,
        "graph_ms": graph_ms,
        "graph_tps": 1000.0 / graph_ms,
        "output_diff": _cmp(graph_out, eager_out),
        "kv_diff": {
            "k": _cmp(graph_kv[0], eager_kv[0]),
            "v": _cmp(graph_kv[1], eager_kv[1]),
        },
        "kv_write_slice_diff": {
            "position": target_position,
            "k": _cmp(graph_write[0], eager_write[0]),
            "v": _cmp(graph_write[1], eager_write[1]),
        },
    }
    result["pass"] = (
        result["output_diff"]["max_abs"] == 0.0
        and result["kv_write_slice_diff"]["k"]["max_abs"] == 0.0
        and result["kv_write_slice_diff"]["v"]["max_abs"] == 0.0
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
