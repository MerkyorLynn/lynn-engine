#!/usr/bin/env python3
"""P75: gate/up tile-inter thread-count sweep.

P74 shows gate/up is the larger active-MoE sub-kernel budget, but the current
P55 tile_inter=2 CUDA scalar kernel only beats Triton by about 4% on the P74
layer set. P75 tests whether the fixed 128-thread launch shape is the culprit
before we stop investing in scalar gate/up variants and move to CuTe/CUTLASS.
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
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu_fast_decode  # noqa: E402


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
    ext,
    *,
    layer: int,
    prompt: str,
    tile_inters: list[int],
    thread_counts: list[int],
    warmup: int,
    iters: int,
) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1, sorted=False)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()

    def triton_ref() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    ref = triton_ref()
    triton_ms = _bench(triton_ref, warmup, iters)
    variants = {}
    for tile_inter in tile_inters:
        for threads in thread_counts:
            key = f"tile_inter_{tile_inter}_threads_{threads}"

            def candidate(tile_inter: int = tile_inter, threads: int = threads) -> torch.Tensor:
                return ext.gate_up_silu_tile_inter_threads_scalar(
                    hidden,
                    expert_ids,
                    w["mlp.experts._gate_up_packed"],
                    w["mlp.experts._gate_up_scale"],
                    w["mlp.experts._gate_up_global_scale"],
                    int(tile_inter),
                    int(threads),
                )

            out = candidate()
            ms = _bench(candidate, warmup, iters)
            variants[key] = {
                "tile_inter": tile_inter,
                "threads": threads,
                "ms": ms,
                "speedup_vs_triton": triton_ms / ms,
                "diff_vs_triton": _diff(ref, out),
            }
    best_key, best = min(variants.items(), key=lambda item: item[1]["ms"])
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "triton_fast_decode_ms": triton_ms,
        "variants": variants,
        "best_variant": best_key,
        "best_ms": best["ms"],
        "best_speedup_vs_triton": triton_ms / best["ms"],
    }


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--tile-inters", default="1,2")
    ap.add_argument("--threads", default="64,128,256")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p75_gateup_threads")

    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    tile_inters = _parse_ints(args.tile_inters)
    thread_counts = _parse_ints(args.threads)
    cases = [
        _run_layer(
            runner,
            ext,
            layer=layer,
            prompt=args.prompt,
            tile_inters=tile_inters,
            thread_counts=thread_counts,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]

    variant_keys = sorted(cases[0]["variants"])
    by_variant = {}
    for key in variant_keys:
        speedups = [float(case["variants"][key]["speedup_vs_triton"]) for case in cases]
        ms_values = [float(case["variants"][key]["ms"]) for case in cases]
        cosines = [float(case["variants"][key]["diff_vs_triton"]["cosine"]) for case in cases]
        rel_l2s = [float(case["variants"][key]["diff_vs_triton"]["rel_l2"]) for case in cases]
        by_variant[key] = {
            "mean_ms": mean(ms_values),
            "mean_speedup_vs_triton": mean(speedups),
            "min_speedup_vs_triton": min(speedups),
            "min_cosine_vs_triton": min(cosines),
            "max_rel_l2_vs_triton": max(rel_l2s),
        }
    best_key, best = max(by_variant.items(), key=lambda item: item[1]["mean_speedup_vs_triton"])
    result = {
        "schema_version": "lynn-engine-p75-gateup-tile-threads-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "prompt": args.prompt,
        "tile_inters": tile_inters,
        "thread_counts": thread_counts,
        "cases": cases,
        "summary": {
            "mean_triton_fast_decode_ms": mean(c["triton_fast_decode_ms"] for c in cases),
            "by_variant": by_variant,
            "best_variant": best_key,
            "best_mean_speedup_vs_triton": best["mean_speedup_vs_triton"],
            "best_min_speedup_vs_triton": best["min_speedup_vs_triton"],
            "best_min_cosine_vs_triton": best["min_cosine_vs_triton"],
            "best_max_rel_l2_vs_triton": best["max_rel_l2_vs_triton"],
        },
        "subkernel_contract_pass": bool(best["min_cosine_vs_triton"] >= 0.999999 and best["max_rel_l2_vs_triton"] <= 0.01),
        "runtime_promote": False,
        "decision": (
            "P75 only sweeps scalar launch shape. Promote nothing without full-generate gates. "
            "If best speedup remains small, scalar gate/up variants are exhausted."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
