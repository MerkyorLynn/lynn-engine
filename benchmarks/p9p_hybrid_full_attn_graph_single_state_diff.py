#!/usr/bin/env python3
"""P9-P: single-state layerwise diff for the hybrid graph token path.

P9-O used separate eager and graph states, which is useful but can mix in any
prefill nondeterminism. This probe mirrors P9-N more closely: capture graph
slots, prefill once, snapshot one base state, run eager from that snapshot, then
restore the same state and run the hybrid graph path while comparing each
boundary against saved eager hidden states.
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
from engine.inference_state import FULL_ATTN_INDICES, LAYER_TYPES, LynnInferenceState  # noqa: E402
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


def _snapshot_state(state: LynnInferenceState) -> dict[str, Any]:
    return {
        "seq_len": int(state.seq_len),
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


def _prefill_into_state(runner: LynnIncrementalRunner, state: LynnInferenceState, prompt: str) -> tuple[int, torch.Tensor]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = ids.shape[1]
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    logits = F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])
    return int(logits[0].argmax().item()), ids


def _capture_full_attn_slots(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    position: int,
) -> dict[int, dict[str, Any]]:
    slots: dict[int, dict[str, Any]] = {}
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    for layer in FULL_ATTN_INDICES:
        input_buf = torch.zeros((1, 1, int(runner.cfg["hidden_size"])), device=runner.device, dtype=runner.dtype)
        output_buf = torch.empty_like(input_buf)

        def graph_body(layer=layer, input_buf=input_buf, output_buf=output_buf) -> None:
            state.seq_len = position
            out = _decode_layer(
                input_buf,
                pos_tensor,
                LAYER_TYPES[layer],
                runner.layer_weights[layer],
                runner.layer_cfgs[layer],
                state,
                layer,
            )
            output_buf.copy_(out)

        graph_body()
        torch.cuda.synchronize()
        state.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_body()
        torch.cuda.synchronize()
        state.reset()
        slots[layer] = {"input": input_buf, "output": output_buf, "graph": graph}
    return slots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    decode_position = int(ids.shape[1])
    full_slots = _capture_full_attn_slots(runner, state, decode_position)
    next_id, _ = _prefill_into_state(runner, state, args.prompt)
    pos_tensor = torch.tensor([[decode_position]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    linear_blocks, linear_capture_s = runner._capture_linear_block_graphs(state, h_seed, pos_tensor)
    base = _snapshot_state(state)

    eager_hidden: dict[str, torch.Tensor] = {}
    _restore_state(state, base)
    h_eager = h_seed
    for block in linear_blocks:
        start_layer = int(block["start_layer"])
        for layer in range(start_layer, start_layer + 3):
            h_eager = _decode_layer(
                h_eager,
                pos_tensor,
                LAYER_TYPES[layer],
                runner.layer_weights[layer],
                runner.layer_cfgs[layer],
                state,
                layer,
            )
        eager_hidden[f"linear_{start_layer}_{start_layer + 2}"] = h_eager.clone()
        full_layer = start_layer + 3
        h_eager = _decode_layer(
            h_eager,
            pos_tensor,
            LAYER_TYPES[full_layer],
            runner.layer_weights[full_layer],
            runner.layer_cfgs[full_layer],
            state,
            full_layer,
        )
        eager_hidden[f"full_{full_layer}"] = h_eager.clone()
    eager_logits = runner._lm_head_logits(_rms_norm(h_eager, runner.outside["model.language_model.norm.weight"]))

    _restore_state(state, base)
    h_graph = h_seed
    rows: list[dict[str, Any]] = []
    for bi, block in enumerate(linear_blocks):
        start_layer = int(block["start_layer"])
        block["input"].copy_(h_graph)
        block["graph"].replay()
        h_graph = block["output"]
        key = f"linear_{start_layer}_{start_layer + 2}"
        rows.append(
            {
                "kind": "linear_block",
                "block_index": bi,
                "start_layer": start_layer,
                "end_layer": start_layer + 2,
                "diff": _cmp(h_graph, eager_hidden[key]),
            }
        )

        full_layer = start_layer + 3
        slot = full_slots[full_layer]
        slot["input"].copy_(h_graph)
        slot["graph"].replay()
        h_graph = slot["output"]
        rows.append(
            {
                "kind": "full_attention_slot",
                "layer": full_layer,
                "diff": _cmp(h_graph, eager_hidden[f"full_{full_layer}"]),
            }
        )

    graph_logits = runner._lm_head_logits(_rms_norm(h_graph, runner.outside["model.language_model.norm.weight"]))
    logit_diff = _cmp(graph_logits, eager_logits)
    graph_next_id = int(graph_logits[0].argmax().item())
    eager_next_id = int(eager_logits[0].argmax().item())
    first_drift = next((row for row in rows if float(row["diff"]["max_abs"]) != 0.0), None)
    result = {
        "schema_version": "lynn-engine-p9p-hybrid-full-attn-graph-single-state-diff-v1",
        "model": args.model,
        "decode_position": decode_position,
        "linear_capture_s": linear_capture_s,
        "full_attn_graph_slots": len(full_slots),
        "linear_block_graphs": len(linear_blocks),
        "layer_rows": rows,
        "first_drift": first_drift,
        "logit_diff": logit_diff,
        "graph_next_id": graph_next_id,
        "eager_next_id": eager_next_id,
        "greedy_pass": graph_next_id == eager_next_id,
        "strict_logit_pass": logit_diff["max_abs"] == 0.0,
        "pass": graph_next_id == eager_next_id,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
