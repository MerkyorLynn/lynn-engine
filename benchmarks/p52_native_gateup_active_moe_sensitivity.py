#!/usr/bin/env python3
"""P52-A: native-FP4 gate/up sensitivity inside the active MoE path.

P44 showed that plain `_scaled_mm` composition is not the final grouped expert
kernel.  This probe narrows the question further:

    current active MoE: Triton gate/up -> Triton down
    candidate:         native FP4 selected gate/up -> Triton down

Down projection, router, and routing weights stay identical.  If this already
drifts at the final active-MoE output, P52 needs either a stricter per-16 native
kernel contract or a small retune/QAT gate before it can chase 155 TPS.  If it
holds, the remaining work is mostly kernel engineering.
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
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402
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


def _build_selected_gateup_native(
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    expert_ids_i64: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_packed = gate_up_packed.index_select(0, expert_ids_i64).reshape(-1, gate_up_packed.shape[-1]).contiguous()
    selected_scale = gate_up_scale.index_select(0, expert_ids_i64).reshape(-1, gate_up_scale.shape[-1]).contiguous()
    effective_scale = selected_scale.float() / gate_up_global.to(selected_scale.device).float()
    scale_b = _compact_scale_to_swizzled_fp8(
        effective_scale,
        outer_dim=int(selected_packed.shape[0]),
        k=int(selected_packed.shape[1] * 2),
    )
    return selected_packed, scale_b


def _native_selected_gateup(
    hidden_2d: torch.Tensor,
    selected_packed: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    act_packed, scale_a = quantize_fp4_m1_native(hidden_2d)
    y = torch._scaled_mm(
        act_packed.view(torch.float4_e2m1fn_x2),
        selected_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    ).float()
    gate_up = y.reshape(top_k, 1024)
    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate) * up).to(torch.bfloat16)


def _run_layer(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompt: str,
    warmup: int,
    iters: int,
) -> dict:
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("P52-A requires torch.float4_e2m1fn_x2 and torch._scaled_mm")

    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    hidden_2d = hidden.reshape(1, -1).contiguous()
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids_i32 = expert_indices[0].to(torch.int32).contiguous()
    expert_ids_i64 = expert_indices[0].to(torch.long).contiguous()

    gate_up_packed = w["mlp.experts._gate_up_packed"]
    gate_up_scale = w["mlp.experts._gate_up_scale"]
    gate_up_global = w["mlp.experts._gate_up_global_scale"]
    down_packed = w["mlp.experts._down_packed"]
    down_scale = w["mlp.experts._down_scale"]
    down_global = w["mlp.experts._down_global_scale"]

    selected_packed, scale_b = _build_selected_gateup_native(
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        expert_ids_i64,
    )

    def triton_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids_i32,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def triton_down(inter: torch.Tensor) -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids_i32,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def native_gateup_hot() -> torch.Tensor:
        return _native_selected_gateup(hidden_2d, selected_packed, scale_b, top_k=top_k)

    def native_gateup_cold() -> torch.Tensor:
        packed, scale = _build_selected_gateup_native(gate_up_packed, gate_up_scale, gate_up_global, expert_ids_i64)
        return _native_selected_gateup(hidden_2d, packed, scale, top_k=top_k)

    def triton_active() -> torch.Tensor:
        return triton_down(triton_gateup())

    def native_active_hot() -> torch.Tensor:
        return triton_down(native_gateup_hot())

    def native_active_cold() -> torch.Tensor:
        return triton_down(native_gateup_cold())

    ref_inter = triton_gateup()
    native_inter = native_gateup_hot()
    ref_out = triton_down(ref_inter)
    native_out = triton_down(native_inter)
    timings = {
        "triton_gateup_ms": _bench(triton_gateup, warmup, iters),
        "triton_down_from_ref_inter_ms": _bench(lambda: triton_down(ref_inter), warmup, iters),
        "triton_active_ms": _bench(triton_active, warmup, iters),
        "native_gateup_hot_ms": _bench(native_gateup_hot, warmup, iters),
        "native_gateup_cold_ms": _bench(native_gateup_cold, max(1, warmup // 2), max(10, iters // 4)),
        "native_active_hot_ms": _bench(native_active_hot, warmup, iters),
        "native_active_cold_ms": _bench(native_active_cold, max(1, warmup // 2), max(10, iters // 4)),
        "activation_quant_m1_ms": _bench(lambda: quantize_fp4_m1_native(hidden_2d)[0], warmup, iters),
        "selected_weight_scale_build_ms": _bench(
            lambda: _build_selected_gateup_native(gate_up_packed, gate_up_scale, gate_up_global, expert_ids_i64)[0],
            max(1, warmup // 2),
            max(10, iters // 4),
        ),
    }
    timings["native_gateup_hot_vs_triton_speedup"] = timings["triton_gateup_ms"] / timings["native_gateup_hot_ms"]
    timings["native_active_hot_vs_triton_speedup"] = timings["triton_active_ms"] / timings["native_active_hot_ms"]
    timings["native_active_cold_vs_triton_speedup"] = timings["triton_active_ms"] / timings["native_active_cold_ms"]
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids_i64.tolist()],
        "routing_weights": [float(x) for x in routing_weights.tolist()],
        "matrix_shapes": {
            "hidden_2d": list(hidden_2d.shape),
            "selected_packed": list(selected_packed.shape),
            "scale_b": list(scale_b.shape),
            "top_k": top_k,
        },
        "diff_native_gateup_vs_triton_gateup": _diff(ref_inter, native_inter),
        "diff_native_active_vs_triton_active": _diff(ref_out, native_out),
        "timings_ms": timings,
        "quality_pass_relaxed": bool(_diff(ref_out, native_out)["cosine"] >= 0.995),
        "quality_pass_strict": bool(_diff(ref_out, native_out)["cosine"] >= 0.9999 and _diff(ref_out, native_out)["rel_l2"] <= 0.01),
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

    # The probe consumes Lynn-native grouped NVFP4 aliases, not the BF16 shadow.
    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(runner, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    result = {
        "schema_version": "lynn-engine-p52-native-gateup-active-moe-sensitivity-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_triton_active_ms": sum(c["timings_ms"]["triton_active_ms"] for c in cases) / len(cases),
            "mean_native_active_hot_ms": sum(c["timings_ms"]["native_active_hot_ms"] for c in cases) / len(cases),
            "mean_native_active_cold_ms": sum(c["timings_ms"]["native_active_cold_ms"] for c in cases) / len(cases),
            "mean_native_active_hot_speedup": sum(c["timings_ms"]["native_active_hot_vs_triton_speedup"] for c in cases)
            / len(cases),
            "mean_native_active_cold_speedup": sum(c["timings_ms"]["native_active_cold_vs_triton_speedup"] for c in cases)
            / len(cases),
            "min_gateup_cosine": min(c["diff_native_gateup_vs_triton_gateup"]["cosine"] for c in cases),
            "min_active_cosine": min(c["diff_native_active_vs_triton_active"]["cosine"] for c in cases),
            "max_active_rel_l2": max(c["diff_native_active_vs_triton_active"]["rel_l2"] for c in cases),
            "all_relaxed_quality_pass": all(c["quality_pass_relaxed"] for c in cases),
            "all_strict_quality_pass": all(c["quality_pass_strict"] for c in cases),
        },
        "interpretation": [
            "This is not a production backend: it isolates native FP4 selected gate/up while keeping router and down identical.",
            "Hot timings assume selected rows/scale_b are prebuilt; cold timings include dynamic top-k gather and scale layout construction.",
            "If quality fails here, the grouped kernel must preserve a stricter per-16 scale contract or pair with targeted retune/QAT.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["all_relaxed_quality_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
