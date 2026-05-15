#!/usr/bin/env python3
"""P9-K: sequentially captured full-token graph family greedy parity.

P9-E captured every future decode-position graph from the same base state and
only changed seq_len. P9-J proved a graph captured *after* an eager prefix is
exact at that later position. Therefore the correct construction is sequential:
capture step N from the state produced by the already-captured graph path, then
use that replayed graph state as the capture base for step N+1.

This is still a benchmark/prototype because capture cost is paid upfront. The
serving design goal is to prewarm or lazily extend this family, then replay at
~graph speed while preserving greedy parity.
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
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    first_id, graph_state = _prefill(runner, args.prompt)
    _, eager_state = _prefill(runner, args.prompt)
    graph_base = _snapshot_state(graph_state)
    eager_base = _snapshot_state(eager_state)
    base_seq_len = int(graph_state.seq_len)

    token_buf = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)
    graphs: list[dict[str, Any]] = []
    capture_rows: list[dict[str, Any]] = []

    capture_token = first_id
    total_capture_ms = 0.0
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
            logits_buf.copy_(F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"]))

        pre_step = _snapshot_state(graph_state)
        token_buf.fill_(capture_token)
        graph_state.seq_len = captured_seq_len
        full_token_graph()
        torch.cuda.synchronize()
        _restore_state(graph_state, pre_step)
        token_buf.fill_(capture_token)
        graph_state.seq_len = captured_seq_len
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_token_graph()
        end.record()
        torch.cuda.synchronize()
        capture_ms = float(start.elapsed_time(end))
        total_capture_ms += capture_ms

        # Replay immediately to advance the graph-state path used to capture
        # the next decode position.
        token_buf.fill_(capture_token)
        graph.replay()
        torch.cuda.synchronize()
        capture_token = int(logits_buf[0].argmax().item())
        graph_state.seq_len = captured_seq_len + 1
        graphs.append({"graph": graph, "seq_len": captured_seq_len, "capture_token": int(capture_token)})
        capture_rows.append({
            "step": step,
            "captured_seq_len": captured_seq_len,
            "capture_ms": capture_ms,
            "warm_next_id": capture_token,
        })

    # Replay captured graph family from the original prefill state.
    _restore_state(graph_state, graph_base)
    graph_token = first_id
    graph_ids: list[int] = []
    graph_logits: list[torch.Tensor] = []
    graph_step_ms: list[float] = []
    for item in graphs:
        token_buf.fill_(graph_token)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        item["graph"].replay()
        end.record()
        torch.cuda.synchronize()
        step_ms = float(start.elapsed_time(end))
        graph_step_ms.append(step_ms)
        logits = logits_buf.clone()
        graph_logits.append(logits)
        graph_token = int(logits[0].argmax().item())
        graph_ids.append(graph_token)
        graph_state.seq_len = int(item["seq_len"]) + 1

    _restore_state(eager_state, eager_base)
    eager_token = first_id
    eager_ids: list[int] = []
    eager_logits: list[torch.Tensor] = []
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

    greedy_pass = graph_ids == eager_ids and all(row["diff"]["top1_match"] for row in rows)
    avg_replay_ms = sum(graph_step_ms) / len(graph_step_ms)
    result = {
        "schema_version": "lynn-engine-p9k-sequential-capture-graph-family-greedy-v1",
        "model": args.model,
        "prompt": args.prompt,
        "base_seq_len": base_seq_len,
        "first_id": first_id,
        "max_new": args.max_new,
        "graph_ids": graph_ids,
        "eager_ids": eager_ids,
        "capture_rows": capture_rows,
        "total_capture_ms": total_capture_ms,
        "avg_capture_ms": total_capture_ms / args.max_new,
        "avg_graph_replay_ms": avg_replay_ms,
        "avg_graph_replay_tps": 1000.0 / avg_replay_ms,
        "amortized_ms_including_capture": (total_capture_ms + sum(graph_step_ms)) / args.max_new,
        "amortized_tps_including_capture": 1000.0 / ((total_capture_ms + sum(graph_step_ms)) / args.max_new),
        "rows": rows,
        "greedy_pass": greedy_pass,
        "pass": greedy_pass,
        "note": (
            "Each future graph is captured from the state produced by prior "
            "captured graphs, not from the original prefill state."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if greedy_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
