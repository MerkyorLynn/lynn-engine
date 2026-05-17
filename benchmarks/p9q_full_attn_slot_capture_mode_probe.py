#!/usr/bin/env python3
"""P9-Q: compare full-attn slot capture modes at the first hybrid drift layer."""
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
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / denom.item()),
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


def _capture_precaptured_slot(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    *,
    layer: int,
    position: int,
) -> dict[str, Any]:
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    input_buf = torch.zeros((1, 1, int(runner.cfg["hidden_size"])), device=runner.device, dtype=runner.dtype)
    output_buf = torch.empty_like(input_buf)

    def graph_body() -> None:
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
    return {"input": input_buf, "output": output_buf, "graph": graph}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if LAYER_TYPES[args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is {LAYER_TYPES[args.layer]!r}, expected full_attention")
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    position = int(ids.shape[1])

    pre_slot = _capture_precaptured_slot(runner, state, layer=args.layer, position=position)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for i in range(runner.n_layers):
        h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
    state.seq_len = position
    h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
    next_id = int(F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])[0].argmax().item())
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    pos_tensor = torch.tensor([[position]], device=runner.device, dtype=torch.long)
    h_block = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    for layer in range(args.layer):
        h_block = _decode_layer(
            h_block,
            pos_tensor,
            LAYER_TYPES[layer],
            runner.layer_weights[layer],
            runner.layer_cfgs[layer],
            state,
            layer,
        )
    before_full = _snapshot_state(state)

    eager_out = _decode_layer(
        h_block,
        pos_tensor,
        LAYER_TYPES[args.layer],
        runner.layer_weights[args.layer],
        runner.layer_cfgs[args.layer],
        state,
        args.layer,
    ).clone()

    _restore_state(state, before_full)
    real_slot = runner._capture_full_attn_layer_graph_slot(state, h_block, pos_tensor, args.layer)
    _restore_state(state, before_full)
    real_slot.input_buf.copy_(h_block)
    real_slot.graph.replay()
    real_out = real_slot.output_buf.clone()

    _restore_state(state, before_full)
    pre_slot["input"].copy_(h_block)
    pre_slot["graph"].replay()
    pre_out = pre_slot["output"].clone()

    result = {
        "schema_version": "lynn-engine-p9q-full-attn-slot-capture-mode-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "position": position,
        "next_id": next_id,
        "real_capture_diff": _cmp(real_out, eager_out),
        "precapture_diff": _cmp(pre_out, eager_out),
        "real_vs_precapture_diff": _cmp(pre_out, real_out),
        "real_capture_exact": bool(torch.equal(real_out, eager_out)),
        "precapture_exact": bool(torch.equal(pre_out, eager_out)),
    }
    result["decision"] = (
        "GREEN: both capture modes are exact."
        if result["real_capture_exact"] and result["precapture_exact"]
        else "AMBER: capture modes differ from eager; inspect diffs."
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["precapture_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
