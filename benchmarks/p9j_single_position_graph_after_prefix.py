#!/usr/bin/env python3
"""P9-J: single full-token graph parity after an eager prefix.

P9-E/P9-I show stable greedy parity for an initial short window but drift at the
next position. This probe removes window complexity: run an eager prefix, then
capture exactly one full-token decode graph at the resulting sequence position
and compare it with one eager decode step from the same state.
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
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), state


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


def _logit_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    a_top = torch.topk(af, 10).indices.tolist()
    b_top = torch.topk(bf, 10).indices.tolist()
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
        "top1_match": int(a_top[0]) == int(b_top[0]),
        "top10_overlap": len(set(int(x) for x in a_top) & set(int(x) for x in b_top)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--prefix-new", type=int, default=8)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    token_id, state = _prefill(runner, args.prompt)
    generated_prefix: list[int] = []
    for step in range(args.prefix_new):
        logits = _decode_one(runner, state, token_id, int(state.seq_len))
        token_id = int(logits[0].argmax().item())
        generated_prefix.append(token_id)

    base = _snapshot_state(state)
    position = int(state.seq_len)
    token_buf = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)

    def full_token_graph() -> None:
        state.seq_len = position
        h = F.embedding(token_buf, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits_buf.copy_(F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"]))

    _restore_state(state, base)
    full_token_graph()
    torch.cuda.synchronize()
    _restore_state(state, base)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full_token_graph()
    torch.cuda.synchronize()

    _restore_state(state, base)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    graph_ms = float(start.elapsed_time(end))
    graph_logits = logits_buf.clone()
    graph_next = int(graph_logits[0].argmax().item())

    _restore_state(state, base)
    eager_logits = _decode_one(runner, state, token_id, position)
    eager_next = int(eager_logits[0].argmax().item())
    diff = _logit_diff(graph_logits, eager_logits)
    result = {
        "schema_version": "lynn-engine-p9j-single-position-graph-after-prefix-v1",
        "model": args.model,
        "prompt": args.prompt,
        "prefix_new": args.prefix_new,
        "generated_prefix": generated_prefix,
        "position": position,
        "input_token_id": token_id,
        "graph_next_id": graph_next,
        "eager_next_id": eager_next,
        "graph_ms": graph_ms,
        "graph_tps": 1000.0 / graph_ms,
        "diff": diff,
        "pass": graph_next == eager_next and diff["top1_match"],
        "note": "Captures exactly one full-token graph after an eager prefix.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
