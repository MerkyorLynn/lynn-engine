#!/usr/bin/env python3
"""P74: active-MoE inner budget probe for grouped per-16 NVFP4.

P67 proved that the down tile can beat the Triton down sub-kernel. P73 proved
that simply wrapping gate/up + down behind a native-owned scratch boundary does
not clear the P69 active-MoE acceptance gate. This probe puts the pieces on the
same timing ledger so the next kernel branch targets the real budget, not the
most exciting isolated microbench.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from statistics import mean
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from benchmarks.p37_moe_config_generate_gate import BASE_ENV  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.native_cuda import load_lynn_native_extension  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
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


def _pct(part: float, total: float) -> float:
    return float(part / total) if total > 0 else float("nan")


def _run_layer(
    runner: LynnIncrementalRunner,
    ext,
    *,
    layer: int,
    prompt: str,
    tile_inter: int,
    tile_hidden: int,
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

    def triton_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def native_gateup() -> torch.Tensor:
        return ext.gate_up_silu_tile_inter_scalar(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            tile_inter,
        )

    inter_triton = triton_gateup()
    inter_native = native_gateup()

    def triton_down_from_triton_inter() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter_triton,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def native_down_from_triton_inter() -> torch.Tensor:
        return ext.down_grouped_per16_tile_reference(
            inter_triton,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            tile_hidden,
        )

    def native_down_from_native_inter() -> torch.Tensor:
        return ext.down_grouped_per16_tile_reference(
            inter_native,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            tile_hidden,
        )

    def triton_active() -> torch.Tensor:
        inter = triton_gateup()
        return nvfp4_grouped_down_weighted_sum(
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

    def p73_native_active() -> torch.Tensor:
        return ext.active_moe_grouped_per16_nonatomic_reference(
            hidden,
            expert_ids,
            routing_weights,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            down_packed,
            down_scale,
            down_global,
            tile_inter,
            tile_hidden,
        )

    triton_down = triton_down_from_triton_inter()
    native_down_triton_inter = native_down_from_triton_inter()
    native_down_native_inter = native_down_from_native_inter()
    triton_active_out = triton_active()
    p73_active_out = p73_native_active()

    triton_gateup_ms = _bench(triton_gateup, warmup, iters)
    native_gateup_ms = _bench(native_gateup, warmup, iters)
    triton_down_ms = _bench(triton_down_from_triton_inter, warmup, iters)
    native_down_triton_inter_ms = _bench(native_down_from_triton_inter, warmup, iters)
    native_down_native_inter_ms = _bench(native_down_from_native_inter, warmup, iters)
    triton_active_ms = _bench(triton_active, warmup, iters)
    p73_active_ms = _bench(p73_native_active, warmup, iters)

    summed_triton_ms = triton_gateup_ms + triton_down_ms
    summed_native_ms = native_gateup_ms + native_down_native_inter_ms
    predicted_down_only_ms = triton_gateup_ms + native_down_triton_inter_ms
    predicted_gateup_only_ms = native_gateup_ms + triton_down_ms

    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "top_k": int(expert_ids.numel()),
        "tile_inter": tile_inter,
        "tile_hidden": tile_hidden,
        "diff_native_gateup_vs_triton_gateup": _diff(inter_triton, inter_native),
        "diff_native_down_triton_inter_vs_triton_down": _diff(triton_down, native_down_triton_inter),
        "diff_native_down_native_inter_vs_triton_down": _diff(triton_down, native_down_native_inter),
        "diff_p73_active_vs_triton_active": _diff(triton_active_out, p73_active_out),
        "triton_gateup_ms": triton_gateup_ms,
        "native_gateup_tile_ms": native_gateup_ms,
        "triton_down_ms": triton_down_ms,
        "native_down_tile_on_triton_inter_ms": native_down_triton_inter_ms,
        "native_down_tile_on_native_inter_ms": native_down_native_inter_ms,
        "triton_active_ms": triton_active_ms,
        "p73_native_active_ms": p73_active_ms,
        "summed_triton_subkernels_ms": summed_triton_ms,
        "summed_native_subkernels_ms": summed_native_ms,
        "predicted_down_only_ms": predicted_down_only_ms,
        "predicted_gateup_only_ms": predicted_gateup_only_ms,
        "native_gateup_vs_triton_speedup": triton_gateup_ms / native_gateup_ms,
        "native_down_vs_triton_speedup": triton_down_ms / native_down_triton_inter_ms,
        "p73_active_vs_triton_speedup": triton_active_ms / p73_active_ms,
        "predicted_down_only_speedup": triton_active_ms / predicted_down_only_ms,
        "predicted_gateup_only_speedup": triton_active_ms / predicted_gateup_only_ms,
        "predicted_two_stage_native_speedup": triton_active_ms / summed_native_ms,
        "gateup_share_of_triton_sum": _pct(triton_gateup_ms, summed_triton_ms),
        "down_share_of_triton_sum": _pct(triton_down_ms, summed_triton_ms),
        "p73_vs_summed_native_delta_ms": p73_active_ms - summed_native_ms,
    }


def _mean(cases: list[dict], key: str) -> float:
    return mean(float(c[key]) for c in cases)


def _min_diff(cases: list[dict], key: str, metric: str) -> float:
    return min(float(c[key][metric]) for c in cases)


def _max_diff(cases: list[dict], key: str, metric: str) -> float:
    return max(float(c[key][metric]) for c in cases)


def _priority(summary: dict) -> str:
    gate_share = summary["mean_gateup_share_of_triton_sum"]
    down_share = summary["mean_down_share_of_triton_sum"]
    p73_speedup = summary["mean_p73_active_vs_triton_speedup"]
    down_only = summary["mean_predicted_down_only_speedup"]
    gate_only = summary["mean_predicted_gateup_only_speedup"]
    if p73_speedup < 1.15 and down_only > gate_only:
        return (
            "Down tile is the proven local win, but two-stage active speedup is "
            "still too small; next kernel should fuse or persist the down path "
            "without extra scratch/launch overhead."
        )
    if gate_share >= down_share:
        return (
            "Gate/up is the larger Triton subkernel budget; next kernel should "
            "target grouped per-16 gate/up math before adding more down variants."
        )
    return (
        "Down remains the larger Triton subkernel budget; continue from P67 "
        "toward a grouped per-16 down kernel with active-MoE boundary gates."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--tile-inter", type=int, default=2)
    ap.add_argument("--tile-hidden", type=int, default=2)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p74_inner_budget")

    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(
            runner,
            ext,
            layer=layer,
            prompt=args.prompt,
            tile_inter=args.tile_inter,
            tile_hidden=args.tile_hidden,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]

    summary = {
        "mean_triton_gateup_ms": _mean(cases, "triton_gateup_ms"),
        "mean_native_gateup_tile_ms": _mean(cases, "native_gateup_tile_ms"),
        "mean_triton_down_ms": _mean(cases, "triton_down_ms"),
        "mean_native_down_tile_on_triton_inter_ms": _mean(cases, "native_down_tile_on_triton_inter_ms"),
        "mean_native_down_tile_on_native_inter_ms": _mean(cases, "native_down_tile_on_native_inter_ms"),
        "mean_triton_active_ms": _mean(cases, "triton_active_ms"),
        "mean_p73_native_active_ms": _mean(cases, "p73_native_active_ms"),
        "mean_summed_triton_subkernels_ms": _mean(cases, "summed_triton_subkernels_ms"),
        "mean_summed_native_subkernels_ms": _mean(cases, "summed_native_subkernels_ms"),
        "mean_predicted_down_only_ms": _mean(cases, "predicted_down_only_ms"),
        "mean_predicted_gateup_only_ms": _mean(cases, "predicted_gateup_only_ms"),
        "mean_native_gateup_vs_triton_speedup": _mean(cases, "native_gateup_vs_triton_speedup"),
        "mean_native_down_vs_triton_speedup": _mean(cases, "native_down_vs_triton_speedup"),
        "mean_p73_active_vs_triton_speedup": _mean(cases, "p73_active_vs_triton_speedup"),
        "mean_predicted_down_only_speedup": _mean(cases, "predicted_down_only_speedup"),
        "mean_predicted_gateup_only_speedup": _mean(cases, "predicted_gateup_only_speedup"),
        "mean_predicted_two_stage_native_speedup": _mean(cases, "predicted_two_stage_native_speedup"),
        "mean_gateup_share_of_triton_sum": _mean(cases, "gateup_share_of_triton_sum"),
        "mean_down_share_of_triton_sum": _mean(cases, "down_share_of_triton_sum"),
        "mean_p73_vs_summed_native_delta_ms": _mean(cases, "p73_vs_summed_native_delta_ms"),
        "min_native_gateup_cosine_vs_triton": _min_diff(cases, "diff_native_gateup_vs_triton_gateup", "cosine"),
        "max_native_gateup_rel_l2_vs_triton": _max_diff(cases, "diff_native_gateup_vs_triton_gateup", "rel_l2"),
        "min_native_down_cosine_vs_triton": _min_diff(cases, "diff_native_down_triton_inter_vs_triton_down", "cosine"),
        "max_native_down_rel_l2_vs_triton": _max_diff(cases, "diff_native_down_triton_inter_vs_triton_down", "rel_l2"),
        "min_p73_active_cosine_vs_triton": _min_diff(cases, "diff_p73_active_vs_triton_active", "cosine"),
        "max_p73_active_rel_l2_vs_triton": _max_diff(cases, "diff_p73_active_vs_triton_active", "rel_l2"),
    }
    result = {
        "schema_version": "lynn-engine-p74-active-moe-inner-budget-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "prompt": args.prompt,
        "tile_inter": args.tile_inter,
        "tile_hidden": args.tile_hidden,
        "cases": cases,
        "summary": summary,
        "subkernel_contract_pass": bool(
            summary["min_p73_active_cosine_vs_triton"] >= 0.999999
            and summary["max_p73_active_rel_l2_vs_triton"] <= 0.01
        ),
        "runtime_promote": False,
        "next_priority": _priority(summary),
        "decision": (
            "P74 is a budget ledger, not a production backend. Use it to pick "
            "the next grouped per-16 kernel branch under the P69 acceptance gate."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
