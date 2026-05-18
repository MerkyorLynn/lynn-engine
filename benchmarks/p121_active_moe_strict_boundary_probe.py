#!/usr/bin/env python3
"""P121: strict active-MoE boundary local parity probe.

This probe validates the new opt-in runtime name:

  LYNN_NATIVE_ACTIVE_MOE_BACKEND=strict_fused_boundary

Milestone 1 keeps the exact staged numerical contract:

  gate/up -> bf16 inter store -> down -> route weighted sum

The implementation is intentionally native-owned at the active-MoE boundary,
while still delegating the inner math to the proven scalar CUDA references.
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
from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.native_cuda import load_lynn_native_extension  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
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


def _gate_up_kernel() -> Callable[..., torch.Tensor]:
    if os.environ.get("LYNN_NATIVE_GATEUP_BACKEND", "triton") == "triton_fast_decode":
        return nvfp4_grouped_gate_up_silu_fast_decode
    return nvfp4_grouped_gate_up_silu


def _run_layer(
    runner: LynnIncrementalRunner,
    ext,
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
    gate_up = _gate_up_kernel()

    def triton_active() -> torch.Tensor:
        inter = gate_up(
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

    def strict_boundary() -> torch.Tensor:
        return ext.active_moe_strict_fused_boundary(
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

    def scalar_contract() -> torch.Tensor:
        return ext.active_moe_scalar_contract(
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

    triton = triton_active()
    strict = strict_boundary()
    scalar = scalar_contract()
    triton_ms = _bench(triton_active, warmup, iters)
    strict_ms = _bench(strict_boundary, warmup, iters)
    scalar_ms = _bench(scalar_contract, warmup, iters)
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "top_k": int(expert_ids.numel()),
        "timings_ms": {
            "triton_active_ms": triton_ms,
            "strict_boundary_ms": strict_ms,
            "scalar_contract_ms": scalar_ms,
            "strict_vs_triton_speedup": triton_ms / strict_ms,
            "strict_vs_scalar_contract_speedup": scalar_ms / strict_ms,
        },
        "strict_boundary_vs_triton": _diff(triton, strict),
        "strict_boundary_vs_scalar_contract": _diff(scalar, strict),
        "scalar_contract_vs_triton": _diff(triton, scalar),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    applied_env: dict[str, str] = {}
    for key, value in BEST_R6000_ENV.items():
        os.environ.setdefault(key, value)
        applied_env[key] = os.environ[key]
    os.environ["LYNN_MOE_FAST_FIXED"] = "0"
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p121_strict_boundary")
    applied_env["LYNN_MOE_FAST_FIXED"] = os.environ["LYNN_MOE_FAST_FIXED"]
    applied_env["LYNN_NATIVE_CUDA_BUILD_DIR"] = os.environ["LYNN_NATIVE_CUDA_BUILD_DIR"]

    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(
            runner,
            ext,
            layer=layer,
            prompt=args.prompt,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]

    min_cosine = min(c["strict_boundary_vs_triton"]["cosine"] for c in cases)
    max_rel_l2 = max(c["strict_boundary_vs_triton"]["rel_l2"] for c in cases)
    max_abs = max(c["strict_boundary_vs_triton"]["max_abs"] for c in cases)
    min_scalar_cosine = min(c["strict_boundary_vs_scalar_contract"]["cosine"] for c in cases)
    max_scalar_rel_l2 = max(c["strict_boundary_vs_scalar_contract"]["rel_l2"] for c in cases)
    max_scalar_abs = max(c["strict_boundary_vs_scalar_contract"]["max_abs"] for c in cases)
    subkernel_contract_pass = bool(min_cosine >= 0.999999 and max_rel_l2 <= 0.01)
    strict_alias_pass = bool(max_scalar_abs == 0.0 and max_scalar_rel_l2 == 0.0)
    result = {
        "schema_version": "lynn-engine-p121-active-moe-strict-boundary-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "prompt": args.prompt,
        "applied_env": applied_env,
        "cases": cases,
        "summary": {
            "mean_triton_active_ms": mean(c["timings_ms"]["triton_active_ms"] for c in cases),
            "mean_strict_boundary_ms": mean(c["timings_ms"]["strict_boundary_ms"] for c in cases),
            "mean_scalar_contract_ms": mean(c["timings_ms"]["scalar_contract_ms"] for c in cases),
            "mean_strict_vs_triton_speedup": mean(c["timings_ms"]["strict_vs_triton_speedup"] for c in cases),
            "min_cosine_vs_triton": min_cosine,
            "max_rel_l2_vs_triton": max_rel_l2,
            "max_abs_vs_triton": max_abs,
            "min_cosine_vs_scalar_contract": min_scalar_cosine,
            "max_rel_l2_vs_scalar_contract": max_scalar_rel_l2,
            "max_abs_vs_scalar_contract": max_scalar_abs,
        },
        "subkernel_contract_pass": subkernel_contract_pass,
        "strict_alias_pass": strict_alias_pass,
        "pass": bool(subkernel_contract_pass and strict_alias_pass),
        "runtime_promote": False,
        "decision": (
            "Strict fused boundary is an opt-in native-owned active-MoE ABI that preserves "
            "the BF16 intermediate contract. Keep it research-only until full generate gates pass."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
