#!/usr/bin/env python3
"""P23: per-layer active MoE latency sweep for packed NVFP4 decode.

P16-P22 showed that the remaining R6000 gap to 155 TPS lives in active routed
experts. This probe loads the resident model once, collects representative
layer inputs from one prefill, and times the MoE sub-pieces for each layer:

  - router + top-k + softmax
  - packed NVFP4 gate/up
  - packed NVFP4 down
  - active routed experts end-to-end
  - shared expert
  - full MoE path

The goal is diagnosis, not a new production path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in {"0", "false", "no", "off"}


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _collect_layer_inputs(runner: LynnIncrementalRunner, prompt: str) -> list[torch.Tensor]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=False)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    layer_inputs: list[torch.Tensor] = []
    for layer_idx in range(runner.n_layers):
        layer_inputs.append(h[:, -1:, :].contiguous())
        h = _prefill_layer(
            h,
            pos,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    return layer_inputs


def _shared_expert(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return torch.zeros_like(h_flat)
    if (
        _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True)
        and "mlp.shared_expert._gate_up_proj.weight" in w
    ):
        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate_s, up_s = gate_up.chunk(2, dim=-1)
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    else:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _layer_row(
    runner: LynnIncrementalRunner,
    layer_idx: int,
    h_in: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> dict:
    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    h_moe = _rms_norm(h_in, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    top_k = int(cfg["num_experts_per_tok"])

    def router_only() -> torch.Tensor:
        router_logits = F.linear(h_flat, w["mlp.gate.weight"])
        routing_weights, expert_indices = torch.topk(
            router_logits,
            top_k,
            dim=-1,
            sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
        )
        return F.softmax(routing_weights, dim=-1, dtype=torch.float32) + expert_indices.float() * 0.0

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        top_k,
        dim=-1,
        sorted=_env_bool("LYNN_ROUTER_TOPK_SORTED", False),
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids = expert_indices[0].to(torch.long).contiguous()

    def gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=_env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
            block_hidden=_env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
            num_warps=_env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
        )

    inter_cached = gateup()

    def down_only() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter_cached,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
        )

    def active_only() -> torch.Tensor:
        inter = gateup()
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=_env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            block_inter=_env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            num_warps=_env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
        )

    def shared_only() -> torch.Tensor:
        return _shared_expert(h_flat, w)

    def full_moe() -> torch.Tensor:
        return moe_forward_decode_packed_nvfp4(h_moe, w, cfg)

    router_ms = _bench(router_only, warmup, iters)
    gateup_ms = _bench(gateup, warmup, iters)
    down_ms = _bench(down_only, warmup, iters)
    active_ms = _bench(active_only, warmup, iters)
    shared_ms = _bench(shared_only, warmup, iters)
    full_ms = _bench(full_moe, warmup, iters)

    expert_count = int(w["mlp.experts._gate_up_packed"].shape[0])
    row = {
        "layer": layer_idx,
        "layer_type": LAYER_TYPES[layer_idx],
        "expert_count": expert_count,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.tolist()],
        "router_ms": router_ms,
        "gateup_ms": gateup_ms,
        "down_ms": down_ms,
        "active_ms": active_ms,
        "shared_ms": shared_ms,
        "full_moe_ms": full_ms,
        "active_sum_ms": gateup_ms + down_ms,
        "full_minus_parts_ms": full_ms - router_ms - active_ms - shared_ms,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--layers", default="all", help="all or comma-separated layer ids")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    layer_inputs = _collect_layer_inputs(runner, args.prompt)
    if args.layers == "all":
        layer_ids = list(range(runner.n_layers))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]

    rows = [
        _layer_row(runner, layer_idx, layer_inputs[layer_idx], warmup=args.warmup, iters=args.iters)
        for layer_idx in layer_ids
    ]

    def total(key: str) -> float:
        return float(sum(row[key] for row in rows))

    by_type: dict[str, dict[str, float | int]] = {}
    for row in rows:
        bucket = by_type.setdefault(row["layer_type"], {"count": 0})
        bucket["count"] = int(bucket["count"]) + 1
        for key in ("router_ms", "gateup_ms", "down_ms", "active_ms", "shared_ms", "full_moe_ms"):
            bucket[key] = float(bucket.get(key, 0.0)) + float(row[key])

    result = {
        "schema_version": "lynn-engine-p23-active-moe-layer-sweep-v1",
        "model": args.model,
        "env": {
            "LYNN_MOE_GATE_BLOCK_INTER": _env_int("LYNN_MOE_GATE_BLOCK_INTER", 8),
            "LYNN_MOE_GATE_BLOCK_HIDDEN": _env_int("LYNN_MOE_GATE_BLOCK_HIDDEN", 256),
            "LYNN_MOE_DOWN_BLOCK_HIDDEN": _env_int("LYNN_MOE_DOWN_BLOCK_HIDDEN", 8),
            "LYNN_MOE_DOWN_BLOCK_INTER": _env_int("LYNN_MOE_DOWN_BLOCK_INTER", 512),
            "LYNN_MOE_GATE_NUM_WARPS": _env_int("LYNN_MOE_GATE_NUM_WARPS", 4),
            "LYNN_MOE_DOWN_NUM_WARPS": _env_int("LYNN_MOE_DOWN_NUM_WARPS", 8),
            "LYNN_ROUTER_TOPK_SORTED": _env_bool("LYNN_ROUTER_TOPK_SORTED", False),
            "LYNN_SHARED_EXPERT_GATE_UP_FUSED": _env_bool("LYNN_SHARED_EXPERT_GATE_UP_FUSED", True),
        },
        "layers": rows,
        "totals_ms": {
            "router_ms": total("router_ms"),
            "gateup_ms": total("gateup_ms"),
            "down_ms": total("down_ms"),
            "active_ms": total("active_ms"),
            "shared_ms": total("shared_ms"),
            "full_moe_ms": total("full_moe_ms"),
        },
        "by_type_ms": by_type,
        "top_active_layers": sorted(rows, key=lambda r: r["active_ms"], reverse=True)[:10],
        "top_full_moe_layers": sorted(rows, key=lambda r: r["full_moe_ms"], reverse=True)[:10],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
