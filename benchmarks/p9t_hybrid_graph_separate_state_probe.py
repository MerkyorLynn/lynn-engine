#!/usr/bin/env python3
"""P9-T: hybrid graph token with separate linear and full-attn graph states."""
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
    _capture_full_attn_slots,
    _cmp,
    _prefill_into_state,
    _restore_state,
    _snapshot_state,
)
from engine.full_forward import _decode_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _copy_linear_state(dst: LynnInferenceState, src: LynnInferenceState) -> None:
    dst.seq_len = int(src.seq_len)
    for i, t in src.recurrent_state.items():
        dst.recurrent_state[i].copy_(t)
    for i, t in src.conv_state.items():
        dst.conv_state[i].copy_(t)


def _copy_full_kv_state(dst: LynnInferenceState, src: LynnInferenceState) -> None:
    dst.seq_len = int(src.seq_len)
    for i, (k, v) in src.kv_cache.items():
        dst.kv_cache[i][0].copy_(k)
        dst.kv_cache[i][1].copy_(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    request_state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    decode_position = int(ids.shape[1])
    next_id, _ = _prefill_into_state(runner, request_state, args.prompt)
    request_base = _snapshot_state(request_state)
    pos_tensor = torch.tensor([[decode_position]], device=runner.device, dtype=torch.long)
    token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
    h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])

    linear_state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    full_state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    _copy_linear_state(linear_state, request_state)
    _copy_full_kv_state(full_state, request_state)
    linear_blocks, linear_capture_s = runner._capture_linear_block_graphs(linear_state, h_seed, pos_tensor)
    linear_base = _snapshot_state(linear_state)
    full_base = _snapshot_state(full_state)
    full_slots = _capture_full_attn_slots(runner, full_state, decode_position)
    _restore_state(full_state, full_base)

    def eager_token() -> torch.Tensor:
        _restore_state(request_state, request_base)
        h = h_seed
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], request_state, i)
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
            slot["input"].copy_(h)
            slot["graph"].replay()
            h = slot["output"]
        h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
        return F.linear(h_final[:, -1, :], runner.outside["lm_head.weight"])

    def graph_token() -> torch.Tensor:
        _restore_state(linear_state, linear_base)
        _restore_state(full_state, full_base)
        return graph_token_body()

    eager_logits = eager_token().clone()
    _restore_state(linear_state, linear_base)
    _restore_state(full_state, full_base)
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
        "schema_version": "lynn-engine-p9t-hybrid-graph-separate-state-probe-v1",
        "model": args.model,
        "decode_position": decode_position,
        "state_mode": "separate_linear_and_full_kv_graph_states",
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
