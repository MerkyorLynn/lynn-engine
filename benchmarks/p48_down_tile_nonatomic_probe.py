#!/usr/bin/env python3
"""P48: tile-hidden non-atomic down-projection probe.

P46 proved the fused atomic single-kernel route is the wrong shape for active
MoE: it removes the intermediate tensor, but atomics dominate.  P48 starts the
non-atomic grouped-kernel line by shrinking the down projection's block count:

    tile=1: 2048 CTAs, 128 threads each
    tile=2: 1024 CTAs, 2 hidden rows per CTA
    tile=4:  512 CTAs, 4 hidden rows per CTA
    tile=8:  256 CTAs, 8 hidden rows per CTA

The probe isolates the down half only.  Gate/up still uses the current Triton
production kernel so the measurement answers one clean question: does a
multi-row non-atomic native down kernel beat the current Triton/scalar down
building block without numerical drift?
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

from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV  # noqa: E402
from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.native_cuda import load_lynn_native_extension  # noqa: E402
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


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    delta = bf - af
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(delta).item() / torch.linalg.vector_norm(af).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def _run_layer(
    runner: LynnIncrementalRunner,
    ext,
    *,
    layer: int,
    prompt: str,
    tiles: list[int],
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
    inter = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )

    def triton_down() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def scalar_down() -> torch.Tensor:
        return ext.down_weighted_sum_scalar(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
        )

    def tile_down(tile: int) -> Callable[[], torch.Tensor]:
        def _call() -> torch.Tensor:
            return ext.down_weighted_sum_tile_scalar(
                inter,
                expert_ids,
                routing_weights,
                w["mlp.experts._down_packed"],
                w["mlp.experts._down_scale"],
                w["mlp.experts._down_global_scale"],
                tile,
            )

        return _call

    ref = triton_down()
    scalar = scalar_down()
    timings = {
        "triton_down_ms": _bench(triton_down, warmup, iters),
        "cuda_scalar_down_ms": _bench(scalar_down, warmup, iters),
    }
    tile_cases = {}
    for tile in tiles:
        candidate_fn = tile_down(tile)
        candidate = candidate_fn()
        tile_ms = _bench(candidate_fn, warmup, iters)
        tile_cases[str(tile)] = {
            "cuda_tile_down_ms": tile_ms,
            "tile_vs_triton_speedup": timings["triton_down_ms"] / tile_ms,
            "tile_vs_scalar_speedup": timings["cuda_scalar_down_ms"] / tile_ms,
            "diff_vs_triton": _diff(ref, candidate),
            "diff_vs_scalar": _diff(scalar, candidate),
        }
    best_tile = min(tile_cases, key=lambda k: tile_cases[k]["cuda_tile_down_ms"])
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "timings_ms": timings,
        "scalar_vs_triton": _diff(ref, scalar),
        "tile_cases": tile_cases,
        "best_tile": int(best_tile),
        "best_tile_ms": tile_cases[best_tile]["cuda_tile_down_ms"],
        "best_tile_vs_triton_speedup": tile_cases[best_tile]["tile_vs_triton_speedup"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--tiles", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    for key, value in BEST_R6000_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p48_down_tile")
    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(
            runner,
            ext,
            layer=layer,
            prompt=args.prompt,
            tiles=args.tiles,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]
    tile_summary = {}
    for tile in args.tiles:
        key = str(tile)
        tile_summary[key] = {
            "mean_cuda_tile_down_ms": sum(c["tile_cases"][key]["cuda_tile_down_ms"] for c in cases) / len(cases),
            "mean_tile_vs_triton_speedup": sum(c["tile_cases"][key]["tile_vs_triton_speedup"] for c in cases) / len(cases),
            "mean_tile_vs_scalar_speedup": sum(c["tile_cases"][key]["tile_vs_scalar_speedup"] for c in cases) / len(cases),
            "min_cosine_vs_triton": min(c["tile_cases"][key]["diff_vs_triton"]["cosine"] for c in cases),
            "max_rel_l2_vs_triton": max(c["tile_cases"][key]["diff_vs_triton"]["rel_l2"] for c in cases),
        }
    best_tile = min(tile_summary, key=lambda k: tile_summary[k]["mean_cuda_tile_down_ms"])
    promote = (
        tile_summary[best_tile]["mean_tile_vs_triton_speedup"] > 1.03
        and tile_summary[best_tile]["min_cosine_vs_triton"] >= 0.99999
        and tile_summary[best_tile]["max_rel_l2_vs_triton"] <= 0.01
    )
    result = {
        "schema_version": "lynn-engine-p48-down-tile-nonatomic-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "tiles": args.tiles,
        "cases": cases,
        "summary": {
            "mean_triton_down_ms": sum(c["timings_ms"]["triton_down_ms"] for c in cases) / len(cases),
            "mean_cuda_scalar_down_ms": sum(c["timings_ms"]["cuda_scalar_down_ms"] for c in cases) / len(cases),
            "tile_summary": tile_summary,
            "best_tile": int(best_tile),
            "best_tile_mean_ms": tile_summary[best_tile]["mean_cuda_tile_down_ms"],
            "best_tile_mean_vs_triton_speedup": tile_summary[best_tile]["mean_tile_vs_triton_speedup"],
            "best_tile_mean_vs_scalar_speedup": tile_summary[best_tile]["mean_tile_vs_scalar_speedup"],
        },
        "promote": promote,
        "decision": "Promote only if the tile-hidden non-atomic down kernel beats Triton down with parity across sampled layers.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
