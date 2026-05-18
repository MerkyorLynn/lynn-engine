#!/usr/bin/env python3
"""P9-V: hybrid token probe with full-attn slots captured on real layer states."""
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

from benchmarks.p9n_hybrid_full_attn_graph_slots_probe import (  # noqa: E402
    _bench,
    _cmp,
    _prefill_into_state,
    _restore_state,
    _snapshot_state,
)
from engine.full_forward import _decode_layer, _rms_norm  # noqa: E402
from engine.inference_state import LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _capture_real_state_full_slots(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    base: dict[str, Any],
    h_seed: torch.Tensor,
    pos_tensor: torch.Tensor,
) -> dict[int, Any]:
    """Capture each full-attn slot at the state it sees in an eager token pass."""
    _restore_state(state, base)
    h = h_seed
    slots: dict[int, Any] = {}
    for block_idx in range(runner.n_layers // 4):
        for layer in range(block_idx * 4, block_idx * 4 + 3):
            h = _decode_layer(
                h,
                pos_tensor,
                runner.layer_types[layer],
                runner.layer_weights[layer],
                runner.layer_cfgs[layer],
                state,
                layer,
            )
        full_layer = block_idx * 4 + 3
        before_full = _snapshot_state(state)
        slots[full_layer] = runner._capture_full_attn_layer_graph_slot(state, h, pos_tensor, full_layer)
        _restore_state(state, before_full)
        h = _decode_layer(
            h,
            pos_tensor,
            runner.layer_types[full_layer],
            runner.layer_weights[full_layer],
            runner.layer_cfgs[full_layer],
            state,
            full_layer,
        )
    _restore_state(state, base)
    return slots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    state = LynnInferenceState.from_config(
        runner.cfg,
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    decode_position = int(ids.shape[1])
    next_id, _ = _prefill_into_state(runner, state, args.prompt)
    pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
    linear_blocks, linear_capture_s = runner._capture_linear_block_graphs(state, h_seed, pos_tensor)
    base = _snapshot_state(state)
    full_slots = _capture_real_state_full_slots(runner, state, base, h_seed, pos_tensor)

    def eager_token() -> torch.Tensor:
        _restore_state(state, base)
        h = h_seed
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, runner.layer_types[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    def graph_token_body() -> torch.Tensor:
        h = h_seed
        for bi, block in enumerate(linear_blocks):
            block["input"].copy_(h)
            block["graph"].replay()
            h = block["output"]
            full_layer = bi * 4 + 3
            slot = full_slots[full_layer]
            slot.input_buf.copy_(h)
            slot.graph.replay()
            h = slot.output_buf
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    def graph_token() -> torch.Tensor:
        _restore_state(state, base)
        return graph_token_body()

    eager_logits = eager_token().clone()
    _restore_state(state, base)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph_logits = graph_token_body().clone()
    end.record()
    torch.cuda.synchronize()
    one_shot_graph_ms = float(start.elapsed_time(end))
    eager_ms = _bench(eager_token, max(1, args.warmup), max(10, args.iters // 5))
    graph_ms = _bench(graph_token, args.warmup, args.iters)
    diff = _cmp(graph_logits, eager_logits)
    graph_next = int(graph_logits[0].argmax().item())
    eager_next = int(eager_logits[0].argmax().item())
    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p9v-hybrid-real-state-full-attn-slots-probe-v1",
        "model": args.model,
        "decode_position": decode_position,
        "full_slot_capture_mode": "real_per_layer_state",
        "full_attn_graph_slots": len(full_slots),
        "linear_block_graphs": len(linear_blocks),
        "linear_capture_s": linear_capture_s,
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "speedup": eager_ms / graph_ms,
        "one_shot_graph_ms": one_shot_graph_ms,
        "one_shot_graph_tps": 1000.0 / one_shot_graph_ms,
        "logit_diff": diff,
        "graph_next_id": graph_next,
        "eager_next_id": eager_next,
        "greedy_pass": graph_next == eager_next,
        "strict_logit_pass": diff["max_abs"] == 0.0,
        "pass": graph_next == eager_next,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
