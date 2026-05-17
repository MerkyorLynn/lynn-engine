#!/usr/bin/env python3
"""P9-U: hybrid full-token graph probe with explicit CUDA graph pool modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
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
from engine.inference_state import FULL_ATTN_INDICES, LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402


def _new_pool() -> Any:
    if not hasattr(torch.cuda, "graph_pool_handle"):
        raise RuntimeError("torch.cuda.graph_pool_handle is unavailable")
    return torch.cuda.graph_pool_handle()


def _pool_for(pool_mode: str, family: str, layer: int, pools: dict[str, Any]) -> Any | None:
    if pool_mode == "default":
        return None
    if pool_mode == "shared":
        return pools.setdefault("shared", _new_pool())
    if pool_mode == "separate":
        return pools.setdefault(family, _new_pool())
    if pool_mode == "per_slot" and family == "full":
        return pools.setdefault(f"full_{layer}", _new_pool())
    if pool_mode == "per_slot":
        return pools.setdefault("linear", _new_pool())
    raise ValueError(f"unknown pool_mode={pool_mode}")


def _capture_linear_block_graphs_with_pool(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    h_seed: torch.Tensor,
    pos_tensor: torch.Tensor,
    *,
    pool_mode: str,
    pools: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    t0 = time.time()
    snap = runner._snapshot_linear_state(state)
    blocks: list[dict[str, Any]] = []
    for start_layer in range(0, runner.n_layers, 4):
        layers = [start_layer, start_layer + 1, start_layer + 2]
        if any(LAYER_TYPES[i] != "linear_attention" for i in layers):
            raise RuntimeError(f"unexpected linear block layout at {layers}")
        input_buf = torch.empty_like(h_seed)
        output_buf = torch.empty_like(h_seed)

        def block_fn(layers=layers, input_buf=input_buf, output_buf=output_buf) -> None:
            h = input_buf
            for i in layers:
                h = runner._decode_layer_fast(h, pos_tensor, state, i)
            output_buf.copy_(h)

        input_buf.copy_(h_seed)
        block_fn()
        runner._restore_linear_state(state, snap)
        input_buf.copy_(h_seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=_pool_for(pool_mode, "linear", start_layer, pools)):
            block_fn()
        blocks.append({"start_layer": start_layer, "input": input_buf, "output": output_buf, "graph": graph})
    torch.cuda.synchronize()
    runner._restore_linear_state(state, snap)
    return blocks, time.time() - t0


def _capture_full_attn_slots_with_pool(
    runner: LynnIncrementalRunner,
    state: LynnInferenceState,
    position: int,
    *,
    pool_mode: str,
    pools: dict[str, Any],
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
        with torch.cuda.graph(graph, pool=_pool_for(pool_mode, "full", layer, pools)):
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
    ap.add_argument("--pool-mode", default="default", choices=["default", "shared", "separate", "per_slot"])
    ap.add_argument("--capture-order", default="full_first", choices=["full_first", "linear_first"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    ids = _encode_prompt(runner.tokenizer, args.prompt, runner.device, use_chat_template=False)
    decode_position = int(ids.shape[1])
    pools: dict[str, Any] = {}

    if args.capture_order == "full_first":
        full_slots = _capture_full_attn_slots_with_pool(
            runner,
            state,
            decode_position,
            pool_mode=args.pool_mode,
            pools=pools,
        )
        next_id, _ = _prefill_into_state(runner, state, args.prompt)
        pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
        token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
        h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        linear_blocks, linear_capture_s = _capture_linear_block_graphs_with_pool(
            runner,
            state,
            h_seed,
            pos_tensor,
            pool_mode=args.pool_mode,
            pools=pools,
        )
        base = _snapshot_state(state)
    else:
        next_id, _ = _prefill_into_state(runner, state, args.prompt)
        pos_tensor = torch.tensor([[int(state.seq_len)]], device=runner.device, dtype=torch.long)
        token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
        h_seed = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        linear_blocks, linear_capture_s = _capture_linear_block_graphs_with_pool(
            runner,
            state,
            h_seed,
            pos_tensor,
            pool_mode=args.pool_mode,
            pools=pools,
        )
        base = _snapshot_state(state)
        full_slots = _capture_full_attn_slots_with_pool(
            runner,
            state,
            decode_position,
            pool_mode=args.pool_mode,
            pools=pools,
        )
        _restore_state(state, base)

    def eager_token() -> torch.Tensor:
        _restore_state(state, base)
        h = h_seed
        for i in range(runner.n_layers):
            h = _decode_layer(h, pos_tensor, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
        state.seq_len = decode_position + 1
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
        state.seq_len = decode_position + 1
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
        "schema_version": "lynn-engine-p9u-hybrid-graph-pool-mode-probe-v1",
        "model": args.model,
        "decode_position": decode_position,
        "pool_mode": args.pool_mode,
        "capture_order": args.capture_order,
        "pool_count": len(pools),
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
