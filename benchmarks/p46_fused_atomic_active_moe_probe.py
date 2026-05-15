#!/usr/bin/env python3
"""P46: fused-atomic active MoE CUDA probe.

This is the first single-kernel active-MoE experiment behind the P45 ABI.  It
computes each `(slot, intermediate)` gate/up scalar and immediately contributes
to all down-projection output rows through fp32 atomics.  That avoids the
materialized `[top_k, 512]` intermediate tensor, but pays atomics.

It is a feasibility probe, not a production path.
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
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(af - bf).item() / torch.linalg.vector_norm(af).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def _run_layer(runner: LynnIncrementalRunner, ext, *, layer: int, prompt: str, warmup: int, iters: int) -> dict:
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

    def triton_active() -> torch.Tensor:
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

    def fused_atomic() -> torch.Tensor:
        return ext.active_moe_fused_atomic_scalar(
            hidden,
            expert_ids,
            routing_weights,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
        )

    ref = triton_active()
    fused = fused_atomic()
    timings = {
        "triton_active_ms": _bench(triton_active, warmup, iters),
        "fused_atomic_scalar_ms": _bench(fused_atomic, warmup, iters),
    }
    timings["fused_atomic_vs_triton_speedup"] = timings["triton_active_ms"] / timings["fused_atomic_scalar_ms"]
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "timings_ms": timings,
        "fused_atomic_vs_triton": _diff(ref, fused),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    for key, value in BEST_R6000_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p46_fused_atomic")
    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [_run_layer(runner, ext, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters) for layer in args.layers]
    result = {
        "schema_version": "lynn-engine-p46-fused-atomic-active-moe-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_triton_active_ms": sum(c["timings_ms"]["triton_active_ms"] for c in cases) / len(cases),
            "mean_fused_atomic_scalar_ms": sum(c["timings_ms"]["fused_atomic_scalar_ms"] for c in cases) / len(cases),
            "mean_fused_atomic_vs_triton_speedup": sum(
                c["timings_ms"]["fused_atomic_vs_triton_speedup"] for c in cases
            )
            / len(cases),
            "min_cosine": min(c["fused_atomic_vs_triton"]["cosine"] for c in cases),
            "max_rel_l2": max(c["fused_atomic_vs_triton"]["rel_l2"] for c in cases),
        },
        "promote": False,
        "decision": "Probe only; atomics are expected to be a bridge, not final grouped FP4 production kernel.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
