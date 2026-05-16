#!/usr/bin/env python3
"""SP-09: Lynn-native grouped per-16 active-MoE tile_reference microbench on Spark sm_121.

Compares the native CUDA `active_moe_grouped_per16_tile_reference` kernel
(extracted from Codex main P65-P68 work) against Spark's SP-01.5 autotuned
Triton path on real Lynn 27B NVFP4 expert weights.

Codex R6000 P68 measured 1.108x with cosine 0.99999988 on R6000 sm_120
against `nvfp4_grouped_gate_up_silu_fast_decode`. Spark uses a different
baseline (SP-01.5 autotuned generic kernel), so the speedup number will
differ. Sub-1.0x is fine here — what matters is whether the native CUDA
path is buildable + numerically equivalent.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
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
from engine.native_cuda import load_lynn_native_extension  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_sp01_autotuned,
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_sp01_autotuned,
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = af.norm() * bf.norm()
    return float((af @ bf) / denom) if denom.item() > 0 else 0.0


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--prompt", default="解释 MoE active expert routing")
    ap.add_argument("--tile-inter", type=int, default=2)
    ap.add_argument("--tile-hidden", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/sp09")
    if not os.environ.get("LYNN_NATIVE_CUDA_ARCH"):
        os.environ["LYNN_NATIVE_CUDA_ARCH"] = "sm_121a"

    print(f"[sp09] arch flag: LYNN_NATIVE_CUDA_ARCH={os.environ['LYNN_NATIVE_CUDA_ARCH']}", flush=True)
    print(f"[sp09] device: {torch.cuda.get_device_name(0)} sm_{'.'.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)

    print("[sp09] loading native extension (JIT build)...", flush=True)
    t0 = time.time()
    ext = load_lynn_native_extension(verbose=False)
    print(f"[sp09] native build OK in {time.time() - t0:.1f}s", flush=True)
    if not hasattr(ext, "active_moe_grouped_per16_tile_reference"):
        print("[sp09] FAIL: active_moe_grouped_per16_tile_reference not in extension")
        return 2

    print(f"[sp09] loading runner from {args.model} (this loads all 40 layers, ~3-4 min)", flush=True)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    print(f"[sp09] runner ready", flush=True)

    model_dir = Path(args.model)
    layer_results = []

    for layer_idx in args.layers:
        print(f"\n[sp09] === layer {layer_idx} ===", flush=True)
        h_layer, _ = _prefill_to_layer_input(runner, layer_idx, args.prompt)
        w = runner.layer_weights[layer_idx]
        cfg = runner.layer_cfgs[layer_idx]
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
            f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj",
            runner.device,
        )
        down_packed, down_scale, down_global = _load_grouped(
            model_dir,
            f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj",
            runner.device,
        )

        # Triton baseline (autotuned, SP-01.5 win)
        def triton_active() -> torch.Tensor:
            inter = nvfp4_grouped_gate_up_silu_sp01_autotuned(
                hidden, expert_ids,
                gate_up_packed, gate_up_scale, gate_up_global,
            )
            return nvfp4_grouped_down_weighted_sum_sp01_autotuned(
                inter, expert_ids, routing_weights,
                down_packed, down_scale, down_global,
            )

        # Native CUDA tile_reference (Codex P68)
        expert_ids_int32 = expert_ids.to(torch.int32).contiguous()
        def native_active() -> torch.Tensor:
            return ext.active_moe_grouped_per16_tile_reference(
                hidden,
                expert_ids_int32,
                routing_weights,
                gate_up_packed, gate_up_scale, gate_up_global,
                down_packed, down_scale, down_global,
                args.tile_inter, args.tile_hidden,
            )

        # Parity check
        out_triton = triton_active()
        out_native = native_active()
        cos = _cosine(out_native, out_triton)
        rl2 = _rel_l2(out_native, out_triton)
        print(f"[sp09] L{layer_idx} parity: cos={cos:.7f} rel_l2={rl2:.2e}", flush=True)

        # Timing
        triton_ms = _bench(triton_active, args.warmup, args.iters)
        native_ms = _bench(native_active, args.warmup, args.iters)
        speedup = triton_ms / native_ms if native_ms > 0 else float("nan")
        print(f"[sp09] L{layer_idx} triton={triton_ms:.4f}ms native={native_ms:.4f}ms speedup={speedup:.3f}x", flush=True)

        layer_results.append({
            "layer": layer_idx,
            "cosine_vs_triton": cos,
            "rel_l2_vs_triton": rl2,
            "triton_ms": triton_ms,
            "native_ms": native_ms,
            "speedup_native_vs_triton": speedup,
        })

    summary = {
        "type": "sp09_native_active_moe_microbench",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "arch_flag": os.environ.get("LYNN_NATIVE_CUDA_ARCH"),
        "tile_inter": args.tile_inter,
        "tile_hidden": args.tile_hidden,
        "warmup": args.warmup,
        "iters": args.iters,
        "model": args.model,
        "layers": layer_results,
        "mean_speedup": sum(r["speedup_native_vs_triton"] for r in layer_results) / len(layer_results),
        "min_cosine": min(r["cosine_vs_triton"] for r in layer_results),
        "max_rel_l2": max(r["rel_l2_vs_triton"] for r in layer_results),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print(f"[sp09] === SUMMARY ===")
    print(f"[sp09] mean speedup native vs sp01-autotuned-triton: {summary['mean_speedup']:.3f}x")
    print(f"[sp09] min cosine vs triton: {summary['min_cosine']:.7f}")
    print(f"[sp09] max rel_l2 vs triton: {summary['max_rel_l2']:.2e}")
    print(f"[sp09] report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
