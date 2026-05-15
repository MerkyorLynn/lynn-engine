#!/usr/bin/env python3
"""P10-H: grouped packed NVFP4 active expert end-to-end probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import (  # noqa: E402
    _load_grouped,
    _prefill_to_layer_input,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=128)
    ap.add_argument("--down-block-hidden", type=int, default=16)
    ap.add_argument("--down-block-inter", type=int, default=128)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    top_k = int(cfg["num_experts_per_tok"])

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.down_proj",
        runner.device,
    )

    def bf16_active_experts() -> torch.Tensor:
        gate_up = w["mlp.experts.gate_up_proj"][expert_ids]
        gate_w, up_w = gate_up.chunk(2, dim=1)
        down_w = w["mlp.experts.down_proj"][expert_ids]
        hidden_f = hidden.float()
        gate_out = torch.einsum("d,kid->ki", hidden_f, gate_w.float())
        up_out = torch.einsum("d,kid->ki", hidden_f, up_w.float())
        inter = F.silu(gate_out) * up_out
        out = torch.einsum("ki,kdi->kd", inter, down_w.float())
        return (out * routing_weights[:, None]).sum(dim=0).to(torch.bfloat16)

    def packed_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=args.gate_block_inter,
            block_hidden=args.gate_block_hidden,
        )

    inter_cached = packed_gateup()

    def packed_down_only() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter_cached,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=args.down_block_hidden,
            block_inter=args.down_block_inter,
        )

    def packed_active_experts() -> torch.Tensor:
        inter = packed_gateup()
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=args.down_block_hidden,
            block_inter=args.down_block_inter,
        )

    ref = bf16_active_experts()
    out = packed_active_experts()
    diff = (out.float() - ref.float()).abs()
    cosine = float(F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item())

    timing = {
        "bf16_active_experts_ms": _bench(bf16_active_experts, args.warmup, args.iters),
        "packed_gateup_ms": _bench(packed_gateup, args.warmup, args.iters),
        "packed_down_only_ms": _bench(packed_down_only, args.warmup, args.iters),
        "packed_active_experts_ms": _bench(packed_active_experts, args.warmup, args.iters),
    }
    timing["packed_vs_bf16_ratio"] = timing["bf16_active_experts_ms"] / timing["packed_active_experts_ms"]

    result = {
        "schema_version": "lynn-engine-p10h-grouped-packed-active-expert-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.tolist()],
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
        },
        "diff": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(out.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": cosine,
        },
        "timing_ms": timing,
        "pass": bool(cosine > 0.98),
        "notes": [
            "Packed path keeps BF16 activations and consumes NVFP4 packed weights directly.",
            "No resident BF16 expert weights are materialized in the packed path.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
