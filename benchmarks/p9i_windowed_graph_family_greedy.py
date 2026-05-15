#!/usr/bin/env python3
"""P9-I: windowed full-token CUDA graph family greedy parity.

P9-E showed an important boundary: a full-token graph family is stable for the
first ~8 decode steps, then tiny numerical differences can flip a near-tie
token. This probe recaptures a graph family at window boundaries instead of
capturing the entire generation upfront.

The goal is to separate two risks:
  1. fixed-position graph plumbing is wrong for later decode positions; or
  2. graph/eager numerical drift accumulates across too long a window.

If windowed recapture passes, the server path can use bounded graph windows. If
it still drifts after the first window, the next target is state refresh or
stricter graph/eager parity inside specific kernels.
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


def _copy_state(dst: LynnInferenceState, src: LynnInferenceState) -> None:
    dst.seq_len = int(src.seq_len)
    for i, (k, v) in src.kv_cache.items():
        dst.kv_cache[i][0].copy_(k)
        dst.kv_cache[i][1].copy_(v)
    for i, t in src.recurrent_state.items():
        dst.recurrent_state[i].copy_(t)
    for i, t in src.conv_state.items():
        dst.conv_state[i].copy_(t)


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


def _capture_window(
    runner: LynnIncrementalRunner,
    graph_state: LynnInferenceState,
    token_buf: torch.Tensor,
    logits_buf: torch.Tensor,
    window_start_seq_len: int,
    window_size: int,
    warm_token_id: int,
) -> tuple[list[dict[str, Any]], float]:
    """Capture one fixed-position graph per step in the current window."""
    base = _snapshot_state(graph_state)
    graphs: list[dict[str, Any]] = []
    capture_start = torch.cuda.Event(enable_timing=True)
    capture_end = torch.cuda.Event(enable_timing=True)
    capture_start.record()

    for local_step in range(window_size):
        captured_seq_len = window_start_seq_len + local_step
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

        _restore_state(graph_state, base)
        token_buf.fill_(warm_token_id)
        graph_state.seq_len = captured_seq_len
        full_token_graph()
        torch.cuda.synchronize()

        _restore_state(graph_state, base)
        token_buf.fill_(warm_token_id)
        graph_state.seq_len = captured_seq_len
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_token_graph()
        graphs.append({"graph": graph, "seq_len": captured_seq_len})
        torch.cuda.synchronize()

    _restore_state(graph_state, base)
    capture_end.record()
    torch.cuda.synchronize()
    return graphs, float(capture_start.elapsed_time(capture_end))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument(
        "--refresh-from-eager",
        action="store_true",
        help=(
            "Diagnostic only: copy eager state into graph state at window "
            "boundaries. This proves graph plumbing per window, not product "
            "speed, because eager remains the source of truth."
        ),
    )
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    first_id, graph_state = _prefill(runner, args.prompt)
    _, eager_state = _prefill(runner, args.prompt)
    base_seq_len = int(graph_state.seq_len)

    token_buf = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)

    graph_token = first_id
    eager_token = first_id
    graph_ids: list[int] = []
    eager_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    graph_replay_ms: list[float] = []

    for window_start in range(0, args.max_new, args.window):
        current_window = min(args.window, args.max_new - window_start)
        if args.refresh_from_eager and window_start:
            _copy_state(graph_state, eager_state)
            graph_token = eager_token

        graphs, capture_ms = _capture_window(
            runner,
            graph_state,
            token_buf,
            logits_buf,
            base_seq_len + window_start,
            current_window,
            graph_token,
        )
        windows.append({
            "window_start_step": window_start,
            "window_size": current_window,
            "capture_ms": capture_ms,
        })

        for local_step, item in enumerate(graphs):
            step = window_start + local_step
            token_buf.fill_(graph_token)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            item["graph"].replay()
            end.record()
            torch.cuda.synchronize()
            step_ms = float(start.elapsed_time(end))
            graph_replay_ms.append(step_ms)
            graph_logits = logits_buf.clone()
            graph_token = int(graph_logits[0].argmax().item())
            graph_ids.append(graph_token)
            graph_state.seq_len = int(item["seq_len"]) + 1

            eager_logits = _decode_eager_one(runner, eager_state, eager_token, base_seq_len + step)
            eager_token = int(eager_logits[0].argmax().item())
            eager_ids.append(eager_token)

            rows.append({
                "step": step,
                "window_start_step": window_start,
                "graph_next_id": graph_token,
                "eager_next_id": eager_token,
                "diff": _logit_diff(graph_logits, eager_logits),
                "graph_replay_ms": step_ms,
                "graph_replay_tps": 1000.0 / step_ms,
            })

    greedy_pass = graph_ids == eager_ids and all(row["diff"]["top1_match"] for row in rows)
    avg_replay_ms = sum(graph_replay_ms) / len(graph_replay_ms)
    total_capture_ms = sum(w["capture_ms"] for w in windows)
    total_replay_ms = sum(graph_replay_ms)
    result = {
        "schema_version": "lynn-engine-p9i-windowed-graph-family-greedy-v1",
        "model": args.model,
        "prompt": args.prompt,
        "base_seq_len": base_seq_len,
        "first_id": first_id,
        "max_new": args.max_new,
        "window": args.window,
        "refresh_from_eager": args.refresh_from_eager,
        "graph_ids": graph_ids,
        "eager_ids": eager_ids,
        "windows": windows,
        "avg_graph_replay_ms": avg_replay_ms,
        "avg_graph_replay_tps": 1000.0 / avg_replay_ms,
        "total_capture_ms": total_capture_ms,
        "total_replay_ms": total_replay_ms,
        "amortized_ms_including_capture": (total_capture_ms + total_replay_ms) / args.max_new,
        "amortized_tps_including_capture": 1000.0 / ((total_capture_ms + total_replay_ms) / args.max_new),
        "rows": rows,
        "greedy_pass": greedy_pass,
        "pass": greedy_pass,
        "note": (
            "Recaptures fixed-position graph families at bounded windows. "
            "refresh_from_eager is diagnostic only; production must either keep "
            "graph state coherent or implement a cheaper refresh."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if greedy_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
