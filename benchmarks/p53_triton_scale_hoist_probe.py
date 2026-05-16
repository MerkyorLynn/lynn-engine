#!/usr/bin/env python3
"""P53: test Triton scale-hoist active MoE kernels.

This is an implementation probe for the external review suggestion that the
current Triton active-MoE kernels reload per-16 scales too often.  It compares
the production kernels against opt-in scale-hoisted variants while keeping the
same router, expert IDs, and routing weights.
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

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_scale_hoist,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_scale_hoist,
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


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _run_layer(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompt: str,
    warmup: int,
    iters: int,
) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids = expert_indices[0].to(torch.int32).contiguous()

    gate_up_packed = w["mlp.experts._gate_up_packed"]
    gate_up_scale = w["mlp.experts._gate_up_scale"]
    gate_up_global = w["mlp.experts._gate_up_global_scale"]
    down_packed = w["mlp.experts._down_packed"]
    down_scale = w["mlp.experts._down_scale"]
    down_global = w["mlp.experts._down_global_scale"]

    def gateup_ref() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def gateup_hoist() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_scale_hoist(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    inter_ref = gateup_ref()
    inter_hoist = gateup_hoist()

    def down_ref_from_ref() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter_ref,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def down_hoist_from_ref() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum_scale_hoist(
            inter_ref,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def active_ref() -> torch.Tensor:
        return down_ref_from_ref()

    def active_hoist_both() -> torch.Tensor:
        inter = gateup_hoist()
        return nvfp4_grouped_down_weighted_sum_scale_hoist(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    active_ref_out = active_ref()
    down_hoist_out = down_hoist_from_ref()
    active_hoist_out = active_hoist_both()
    timings = {
        "gateup_ref_ms": _bench(gateup_ref, warmup, iters),
        "gateup_hoist_ms": _bench(gateup_hoist, warmup, iters),
        "down_ref_from_ref_inter_ms": _bench(down_ref_from_ref, warmup, iters),
        "down_hoist_from_ref_inter_ms": _bench(down_hoist_from_ref, warmup, iters),
        "active_ref_ms": _bench(active_ref, warmup, iters),
        "active_hoist_both_ms": _bench(active_hoist_both, warmup, iters),
    }
    timings["gateup_hoist_speedup"] = timings["gateup_ref_ms"] / timings["gateup_hoist_ms"]
    timings["down_hoist_speedup"] = timings["down_ref_from_ref_inter_ms"] / timings["down_hoist_from_ref_inter_ms"]
    timings["active_hoist_speedup"] = timings["active_ref_ms"] / timings["active_hoist_both_ms"]
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_indices[0].tolist()],
        "diff_gateup_hoist_vs_ref": _diff(inter_ref, inter_hoist),
        "diff_down_hoist_vs_ref": _diff(active_ref_out, down_hoist_out),
        "diff_active_hoist_both_vs_ref": _diff(active_ref_out, active_hoist_out),
        "timings_ms": timings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(runner, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    result = {
        "schema_version": "lynn-engine-p53-triton-scale-hoist-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_gateup_ref_ms": sum(c["timings_ms"]["gateup_ref_ms"] for c in cases) / len(cases),
            "mean_gateup_hoist_ms": sum(c["timings_ms"]["gateup_hoist_ms"] for c in cases) / len(cases),
            "mean_down_ref_ms": sum(c["timings_ms"]["down_ref_from_ref_inter_ms"] for c in cases) / len(cases),
            "mean_down_hoist_ms": sum(c["timings_ms"]["down_hoist_from_ref_inter_ms"] for c in cases) / len(cases),
            "mean_active_ref_ms": sum(c["timings_ms"]["active_ref_ms"] for c in cases) / len(cases),
            "mean_active_hoist_ms": sum(c["timings_ms"]["active_hoist_both_ms"] for c in cases) / len(cases),
            "mean_gateup_speedup": sum(c["timings_ms"]["gateup_hoist_speedup"] for c in cases) / len(cases),
            "mean_down_speedup": sum(c["timings_ms"]["down_hoist_speedup"] for c in cases) / len(cases),
            "mean_active_speedup": sum(c["timings_ms"]["active_hoist_speedup"] for c in cases) / len(cases),
            "min_active_cosine": min(c["diff_active_hoist_both_vs_ref"]["cosine"] for c in cases),
            "max_active_rel_l2": max(c["diff_active_hoist_both_vs_ref"]["rel_l2"] for c in cases),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
