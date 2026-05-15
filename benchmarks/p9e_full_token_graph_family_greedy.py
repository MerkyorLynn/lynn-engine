#!/usr/bin/env python3
"""P9-E: full-token CUDA graph family greedy parity.

P9-D proved a single captured graph can consume mutable token-buffer contents.
P9-E captures a small family of full-token graphs, one per fixed decode
position, then replays them sequentially for greedy generation.

This is still a benchmark gate, not the final server implementation. But it is
the first proof that a bucket/window of graph-captured decode steps can advance
state and produce the same greedy tokens as eager.
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


def _decode_eager_one(
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
    ap.add_argument("--max-new", type=int, default=4)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    first_id, graph_state = _prefill(runner, args.prompt)
    _, eager_state = _prefill(runner, args.prompt)
    graph_base = _snapshot_state(graph_state)
    eager_base = _snapshot_state(eager_state)
    base_seq_len = int(graph_state.seq_len)

    token_buf = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)
    graphs = []

    # Capture one graph per decode position. The graph reads token_buf and
    # writes into the fixed KV slot implied by captured_seq_len.
    for step in range(args.max_new):
        captured_seq_len = base_seq_len + step
        pos_tensor = torch.tensor([[captured_seq_len]], device=runner.device, dtype=torch.long)

        def full_token_graph(pos_tensor=pos_tensor, captured_seq_len=captured_seq_len) -> None:
            graph_state.seq_len = captured_seq_len
            h = F.embedding(token_buf, runner.outside["model.language_model.embed_tokens.weight"])
            for i in range(runner.n_layers):
                h = _decode_layer(
                    h,
                    pos_tensor,
                    LAYER_TYPES[i],
                    runner.layer_weights[i],
                    runner.layer_cfgs[i],
                    graph_state,
                    i,
                )
            h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
            logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
            logits_buf.copy_(logits)

        _restore_state(graph_state, graph_base)
        token_buf.fill_(first_id)
        graph_state.seq_len = captured_seq_len
        full_token_graph()
        torch.cuda.synchronize()
        _restore_state(graph_state, graph_base)
        token_buf.fill_(first_id)
        graph_state.seq_len = captured_seq_len
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_token_graph()
        graphs.append({"graph": graph, "seq_len": captured_seq_len})
        torch.cuda.synchronize()

    # Replay graph family greedily.
    _restore_state(graph_state, graph_base)
    graph_token = first_id
    graph_ids = []
    graph_logits = []
    graph_step_ms = []
    for item in graphs:
        token_buf.fill_(graph_token)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        item["graph"].replay()
        end.record()
        torch.cuda.synchronize()
        graph_step_ms.append(float(start.elapsed_time(end)))
        logits = logits_buf.clone()
        graph_logits.append(logits)
        graph_token = int(logits[0].argmax().item())
        graph_ids.append(graph_token)
        graph_state.seq_len = int(item["seq_len"]) + 1

    # Eager greedy from the same prefill state.
    _restore_state(eager_state, eager_base)
    eager_token = first_id
    eager_ids = []
    eager_logits = []
    for step in range(args.max_new):
        logits = _decode_eager_one(runner, eager_state, eager_token, base_seq_len + step)
        eager_logits.append(logits.clone())
        eager_token = int(logits[0].argmax().item())
        eager_ids.append(eager_token)
    torch.cuda.synchronize()

    rows = []
    for i, (g, e) in enumerate(zip(graph_logits, eager_logits)):
        rows.append({
            "step": i,
            "graph_next_id": graph_ids[i],
            "eager_next_id": eager_ids[i],
            "diff": _logit_diff(g, e),
            "graph_replay_ms": graph_step_ms[i],
            "graph_replay_tps": 1000.0 / graph_step_ms[i],
        })

    strict_logit_pass = all(
        row["diff"]["cosine"] > 0.9999
        and row["diff"]["top1_match"]
        and row["diff"]["top10_overlap"] >= 9
        for row in rows
    )
    greedy_pass = graph_ids == eager_ids and all(row["diff"]["top1_match"] for row in rows)
    result = {
        "schema_version": "lynn-engine-p9e-full-token-graph-family-greedy-v1",
        "model": args.model,
        "prompt": args.prompt,
        "base_seq_len": base_seq_len,
        "first_id": first_id,
        "graph_ids": graph_ids,
        "eager_ids": eager_ids,
        "avg_graph_replay_ms": sum(graph_step_ms) / len(graph_step_ms),
        "avg_graph_replay_tps": 1000.0 / (sum(graph_step_ms) / len(graph_step_ms)),
        "rows": rows,
        "greedy_pass": greedy_pass,
        "strict_logit_pass": strict_logit_pass,
        "pass": greedy_pass,
        "note": (
            "Graph family captures one fixed-position graph per decode step. "
            "Greedy parity is the serving gate; strict logits are diagnostic "
            "because graph/eager may differ in reduction/order while preserving top-1."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if greedy_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
