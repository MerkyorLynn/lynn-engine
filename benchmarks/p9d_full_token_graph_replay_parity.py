#!/usr/bin/env python3
"""P9-D: full-token CUDA graph replay parity with mutable token input.

P6-M measured the upper-bound by replaying one captured token/state over and
over. This gate is closer to productization: capture a full decode token graph
once, then mutate the token input buffer before replay and compare logits/state
against eager for the same fixed prefill state.

This still is not a full serving loop because the cached position is fixed.
It answers a narrower prerequisite question: can a captured full-token graph
consume changed token-buffer contents and remain numerically aligned?
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


def _diff_logits(a: torch.Tensor, b: torch.Tensor, *, top_k: int = 10) -> dict[str, Any]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    a_top = torch.topk(af, k=top_k)
    b_top = torch.topk(bf, k=top_k)
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
        "top1_match": int(a_top.indices[0].item()) == int(b_top.indices[0].item()),
        "top10_overlap": len(set(int(x) for x in a_top.indices.tolist()) & set(int(x) for x in b_top.indices.tolist())),
        "graph_top1": int(a_top.indices[0].item()),
        "eager_top1": int(b_top.indices[0].item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--token-offsets", default="0,1,2")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    next_id, graph_state = _prefill(runner, args.prompt)
    _, eager_state = _prefill(runner, args.prompt)
    base_graph = _snapshot_state(graph_state)
    base_eager = _snapshot_state(eager_state)

    token_ids = [max(0, next_id + int(x.strip())) for x in args.token_offsets.split(",") if x.strip()]
    pos_tensor = torch.tensor([[int(graph_state.seq_len)]], device=runner.device, dtype=torch.long)
    token_buf = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    logits_buf = torch.empty((1, runner.outside["lm_head.weight"].shape[0]), device=runner.device, dtype=runner.dtype)

    def full_token_graph() -> None:
        h = F.embedding(token_buf, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], graph_state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
        logits_buf.copy_(logits)

    # Warm allocations and capture on token_ids[0], restoring the state values
    # after every mutation so the graph starts from the same prompt cache.
    _restore_state(graph_state, base_graph)
    token_buf.fill_(token_ids[0])
    full_token_graph()
    torch.cuda.synchronize()
    _restore_state(graph_state, base_graph)
    token_buf.fill_(token_ids[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        full_token_graph()
    torch.cuda.synchronize()

    rows = []
    for token_id in token_ids:
        _restore_state(graph_state, base_graph)
        token_buf.fill_(token_id)
        graph.replay()
        torch.cuda.synchronize()
        graph_logits = logits_buf.clone()

        _restore_state(eager_state, base_eager)
        token = torch.tensor([[token_id]], device=runner.device, dtype=torch.long)
        h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], eager_state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        eager_logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
        torch.cuda.synchronize()

        rows.append({
            "token_id": int(token_id),
            "logits_diff": _diff_logits(graph_logits, eager_logits),
        })

    passed = all(
        row["logits_diff"]["cosine"] > 0.9999
        and row["logits_diff"]["top1_match"]
        and row["logits_diff"]["top10_overlap"] >= 9
        for row in rows
    )
    result = {
        "schema_version": "lynn-engine-p9d-full-token-graph-replay-parity-v1",
        "model": args.model,
        "prompt": args.prompt,
        "base_next_id": int(next_id),
        "tokens_tested": token_ids,
        "fixed_position": int(pos_tensor.item()),
        "rows": rows,
        "pass": passed,
        "note": "Fixed-position graph replay parity only; serving still needs position/KV-slot discipline.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
