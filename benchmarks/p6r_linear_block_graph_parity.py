#!/usr/bin/env python3
"""P6-R: correctness gate for graph-captured linear-attention blocks."""
from __future__ import annotations

import argparse
import json
import os
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


def _prepare_decode_moe_fast_layout(layer_weights: list[dict[str, Any]], layer_cfgs: list[dict[str, Any]]) -> None:
    from triton_kernels.moe_expert_ffn import stack_expert_weights

    for w, cfg in zip(layer_weights, layer_cfgs):
        if "mlp.experts._gate_stacked" in w:
            continue
        if "mlp.experts.gate_up_proj" in w and "mlp.experts.down_proj" in w:
            gate_stacked, up_stacked = w["mlp.experts.gate_up_proj"].chunk(2, dim=1)
            w["mlp.experts._gate_stacked"] = gate_stacked
            w["mlp.experts._up_stacked"] = up_stacked
            w["mlp.experts._down_stacked"] = w["mlp.experts.down_proj"]
            cfg["num_experts"] = int(w["mlp.experts.gate_up_proj"].shape[0])
        else:
            stack_expert_weights(w, num_experts=cfg["num_experts"])


def _prefill(runner: LynnIncrementalRunner, prompt: str):
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


def _snapshot_state(state: LynnInferenceState, layers: list[int]) -> dict[str, Any]:
    return {
        "seq_len": state.seq_len,
        "recurrent": {i: state.recurrent_state[i].clone() for i in layers},
        "conv": {i: state.conv_state[i].clone() for i in layers},
    }


def _restore_state(state: LynnInferenceState, snap: dict[str, Any]) -> None:
    state.seq_len = int(snap["seq_len"])
    for i, t in snap["recurrent"].items():
        state.recurrent_state[i].copy_(t)
    for i, t in snap["conv"].items():
        state.conv_state[i].copy_(t)


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--block-layers", type=int, default=3)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    args = ap.parse_args()

    layers = list(range(args.start_layer, args.start_layer + args.block_layers))
    for i in layers:
        if LAYER_TYPES[i] != "linear_attention":
            raise ValueError(f"layer {i} is {LAYER_TYPES[i]!r}, expected linear_attention")

    requested_moe = os.environ.get("LYNN_MOE_IMPL", "optimized")
    if requested_moe == "triton":
        os.environ["LYNN_MOE_IMPL"] = "optimized"
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    os.environ["LYNN_MOE_IMPL"] = requested_moe
    if requested_moe in {"triton", "indexed_bmm"}:
        _prepare_decode_moe_fast_layout(runner.layer_weights, runner.layer_cfgs)

    next_id, graph_state = _prefill(runner, args.prompt)
    _, eager_state = _prefill(runner, args.prompt)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    pos_tensor = torch.tensor([[int(graph_state.seq_len)]], device=runner.device, dtype=torch.long)

    graph_snap = _snapshot_state(graph_state, layers)
    input_buf = torch.empty_like(h_seed)
    output_buf = torch.empty_like(h_seed)

    def graph_block():
        h = input_buf
        for i in layers:
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], graph_state, i)
        output_buf.copy_(h)

    # Warm and capture on the same tensor addresses, then restore values.
    input_buf.copy_(h_seed)
    graph_block()
    _restore_state(graph_state, graph_snap)
    input_buf.copy_(h_seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_block()
    torch.cuda.synchronize()

    _restore_state(graph_state, graph_snap)
    input_buf.copy_(h_seed)
    graph.replay()
    torch.cuda.synchronize()
    graph_out = output_buf.clone()

    eager_h = h_seed
    for i in layers:
        eager_h = _decode_layer(eager_h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], eager_state, i)
    torch.cuda.synchronize()

    state_diffs = {}
    for i in layers:
        state_diffs[f"layer_{i}_recurrent"] = _diff(graph_state.recurrent_state[i], eager_state.recurrent_state[i])
        state_diffs[f"layer_{i}_conv"] = _diff(graph_state.conv_state[i], eager_state.conv_state[i])

    result = {
        "schema_version": "lynn-engine-p6r-linear-block-graph-parity-v1",
        "model": args.model,
        "layers": layers,
        "moe_impl": requested_moe,
        "linear_recurrent_backend": os.environ.get("LYNN_LINEAR_ATTN_RECURRENT_BACKEND", "torch"),
        "linear_inproj_fused": os.environ.get("LYNN_LINEAR_ATTN_INPROJ_FUSED", "0"),
        "output_diff": _diff(graph_out, eager_h),
        "state_diffs": state_diffs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
