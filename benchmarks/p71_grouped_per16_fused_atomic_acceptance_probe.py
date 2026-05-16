#!/usr/bin/env python3
"""P71: one-boundary fused atomic candidate under the P69 acceptance schema.

P46 already showed that the first fused atomic scalar kernel is not production
worthy. P71 re-runs that idea under the current grouped per-16 / P69 acceptance
contract, using the current fast Triton active-MoE baseline. This closes the
atomic branch in the same language future fused candidates will use.
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

    def triton_active() -> torch.Tensor:
        inter = nvfp4_grouped_gate_up_silu_fast_decode(
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

    triton = triton_active()
    candidate = fused_atomic()
    triton_ms = _bench(triton_active, warmup, iters)
    candidate_ms = _bench(fused_atomic, warmup, iters)
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "top_k": int(expert_ids.numel()),
        "diff_candidate_vs_triton": _diff(triton, candidate),
        "triton_active_ms": triton_ms,
        "candidate_active_ms": candidate_ms,
        "candidate_vs_triton_speedup": triton_ms / candidate_ms,
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

    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p71_fused_atomic_acceptance")

    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(runner, ext, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    min_cosine = min(c["diff_candidate_vs_triton"]["cosine"] for c in cases)
    max_rel_l2 = max(c["diff_candidate_vs_triton"]["rel_l2"] for c in cases)
    result = {
        "schema_version": "lynn-engine-p71-grouped-per16-fused-atomic-acceptance-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "prompt": args.prompt,
        "cases": cases,
        "summary": {
            "mean_triton_active_ms": mean(c["triton_active_ms"] for c in cases),
            "mean_candidate_active_ms": mean(c["candidate_active_ms"] for c in cases),
            "mean_candidate_vs_triton_speedup": mean(c["candidate_vs_triton_speedup"] for c in cases),
            "min_cosine_vs_triton": min_cosine,
            "max_rel_l2_vs_triton": max_rel_l2,
        },
        "subkernel_contract_pass": bool(min_cosine >= 0.999999 and max_rel_l2 <= 0.01),
        "runtime_promote": False,
        "decision": (
            "The fused atomic one-boundary candidate is measured under the P69 "
            "schema. It is expected to remain a negative/closure result, not a "
            "runtime default."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
