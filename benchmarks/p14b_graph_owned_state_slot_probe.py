#!/usr/bin/env python3
"""P14-B: graph-owned full-token slot probe.

P13 proved that a current-position full-token CUDA graph is strict, but the
capture-per-token path is far too slow. P14 measured state-refresh copy cost and
found it cheap. This probe combines the two ideas:

  real request state -> graph-owned state -> replay -> committed request state

The slot is still fixed to one sequence position. It is a correctness and cost
gate for a later slot cache, not a production serving path yet.
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
from engine.resident_runner import FullTokenGraphSlot, LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _prefill(runner: LynnIncrementalRunner, prompt: str) -> tuple[int, LynnInferenceState]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = int(ids.shape[1])
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), state


def _decode_one(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    token_id: int,
) -> tuple[int, torch.Tensor]:
    token = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    token.fill_(int(token_id))
    pos_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    pos_tensor.fill_(int(state.seq_len))
    h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for i in range(runner.n_layers):
        h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len += 1
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = runner._lm_head_logits(h_final)
    return int(logits[0].argmax().item()), logits


def _decode_prefix(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    first_id: int,
    prefix_new: int,
) -> int:
    token_id = int(first_id)
    for _ in range(prefix_new):
        token_id, _ = _decode_one(runner, state, token_id)
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


def _new_state_like(runner: LynnIncrementalRunner) -> LynnInferenceState:
    return LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )


def _capture_graph_owned_slot(
    runner: LynnIncrementalRunner,
    graph_state: LynnInferenceState,
    token_id: int,
) -> FullTokenGraphSlot:
    # Use the runner helper that P10-T already proved strict across prompts and
    # prefixes. P14-B is about graph-owned state refresh, not reimplementing the
    # capture lifecycle.
    return runner._capture_full_token_graph_slot(graph_state, token_id)


def _logit_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a[0].float()
    bf = b[0].float()
    top_a = torch.topk(af, k=10).indices
    top_b = torch.topk(bf, k=10).indices
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
        "top1_match": int(top_a[0].item()) == int(top_b[0].item()),
        "top10_overlap": len(set(int(x) for x in top_a.tolist()) & set(int(x) for x in top_b.tolist())),
        "eager_top1": int(top_a[0].item()),
        "graph_top1": int(top_b[0].item()),
    }


def _bench_slot(
    slot: FullTokenGraphSlot,
    graph_state: LynnInferenceState,
    real_state: LynnInferenceState,
    commit_state: LynnInferenceState,
    token_id: int,
    iters: int,
) -> dict[str, float]:
    # Warm up one full refresh/replay/commit cycle.
    _copy_state(graph_state, real_state)
    slot.replay(token_id)
    graph_state.seq_len = slot.seq_len + 1
    _copy_state(commit_state, graph_state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _copy_state(graph_state, real_state)
        slot.replay(token_id)
        graph_state.seq_len = slot.seq_len + 1
        _copy_state(commit_state, graph_state)
    end.record()
    torch.cuda.synchronize()
    total_ms = float(start.elapsed_time(end))
    return {
        "iters": iters,
        "total_ms": total_ms,
        "avg_ms": total_ms / iters,
        "tps_equivalent": 1000.0 / (total_ms / iters),
    }


def _run_case(
    runner: LynnIncrementalRunner,
    *,
    prompt: str,
    prefix_new: int,
    iters: int,
) -> dict[str, Any]:
    first_id, real_state = _prefill(runner, prompt)
    token_id = _decode_prefix(runner, real_state, first_id, prefix_new)
    seq_len = int(real_state.seq_len)

    eager_state = _new_state_like(runner)
    graph_state = _new_state_like(runner)
    diag_state = _new_state_like(runner)
    commit_state = _new_state_like(runner)
    _copy_state(eager_state, real_state)
    _copy_state(graph_state, real_state)
    _copy_state(diag_state, real_state)

    capture_start = torch.cuda.Event(enable_timing=True)
    capture_end = torch.cuda.Event(enable_timing=True)
    capture_start.record()
    slot = _capture_graph_owned_slot(runner, graph_state, token_id)
    capture_end.record()
    torch.cuda.synchronize()

    # Reset graph state to the real request state after capture mutation.
    _copy_state(graph_state, real_state)
    eager_next, eager_logits = _decode_one(runner, eager_state, token_id)
    graph_logits = slot.replay(token_id).clone()
    graph_state.seq_len = slot.seq_len + 1
    _copy_state(commit_state, graph_state)
    graph_next = int(graph_logits[0].argmax().item())
    same_state_eager_next, same_state_eager_logits = _decode_one(runner, diag_state, token_id)
    diff = _logit_diff(eager_logits, graph_logits)
    same_state_diff = _logit_diff(same_state_eager_logits, graph_logits)
    eager_state_diff = _logit_diff(eager_logits, same_state_eager_logits)

    bench = _bench_slot(slot, graph_state, real_state, commit_state, token_id, iters)
    return {
        "prompt": prompt,
        "prefix_new": prefix_new,
        "seq_len": seq_len,
        "input_token_id": int(token_id),
        "eager_next_id": int(eager_next),
        "graph_next_id": int(graph_next),
        "same_state_eager_next_id": int(same_state_eager_next),
        "same_next_id": int(eager_next) == int(graph_next),
        "committed_seq_len": int(commit_state.seq_len),
        "capture_ms": float(capture_start.elapsed_time(capture_end)),
        "refresh_replay_commit": bench,
        "logit_diff": diff,
        "same_state_logit_diff": same_state_diff,
        "eager_state_logit_diff": eager_state_diff,
        "pass": int(eager_next) == int(graph_next) and diff["max_abs"] == 0.0 and diff["top10_overlap"] == 10,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--prefix-new", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_case(runner, prompt=args.prompt, prefix_new=prefix_new, iters=args.iters)
        for prefix_new in args.prefix_new
    ]
    result = {
        "schema_version": "lynn-engine-p14b-graph-owned-state-slot-probe-v1",
        "model": args.model,
        "prompt": args.prompt,
        "cases": cases,
        "pass": all(case["pass"] for case in cases),
        "note": "Fixed-position graph-owned state slot; proves refresh/replay/commit viability before slot cache integration.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
