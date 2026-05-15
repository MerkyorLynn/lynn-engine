#!/usr/bin/env python3
"""P10-G: native FP4 selected-expert gate/up probe.

P10-F proved grouped scalar unpack is numerically correct, but it cannot be the
path to 100 TPS. This probe switches the same top-k gate/up workload to
Blackwell native FP4 tensor cores via `torch._scaled_mm`.

Two timings matter:

* hot/prebuilt: selected expert rows and scale_b are already resident for this
  router decision. This is the tensor-core upper bound for this shape.
* cold/gather: every call gathers the selected experts and rebuilds scale_b.
  This approximates the naive dynamic top-k implementation cost.
"""
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
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402


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


def _build_selected_gateup_native(
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_packed = gate_up_packed[expert_ids].reshape(-1, gate_up_packed.shape[-1]).contiguous()
    selected_scale = gate_up_scale[expert_ids].reshape(-1, gate_up_scale.shape[-1]).contiguous()
    effective_scale = selected_scale.float() / gate_up_global.to(selected_scale.device).float()
    scale_b = _compact_scale_to_swizzled_fp8(
        effective_scale,
        outer_dim=int(selected_packed.shape[0]),
        k=int(selected_packed.shape[1] * 2),
    )
    return selected_packed, scale_b


def _native_gateup_silu(
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
    gate, up = y.reshape(top_k, 1024).chunk(2, dim=1)
    return (F.silu(gate) * up).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("native FP4 requires torch.float4_e2m1fn_x2 and torch._scaled_mm")

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    hidden_2d = h_moe.view(-1, h_moe.shape[-1])[:1].contiguous()
    hidden = hidden_2d[0]

    router_logits = F.linear(hidden_2d, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    expert_ids = expert_indices[0].to(torch.long)
    top_k = int(expert_ids.numel())

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    selected_packed, selected_scale_b = _build_selected_gateup_native(
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        expert_ids,
    )

    def bf16_gateup() -> torch.Tensor:
        gate_up = w["mlp.experts.gate_up_proj"][expert_ids]
        gate_w, up_w = gate_up.chunk(2, dim=1)
        hidden_f = hidden.float()
        gate_out = torch.einsum("d,kid->ki", hidden_f, gate_w.float())
        up_out = torch.einsum("d,kid->ki", hidden_f, up_w.float())
        return (F.silu(gate_out) * up_out).to(torch.bfloat16)

    def native_hot_gateup() -> torch.Tensor:
        return _native_gateup_silu(
            hidden_2d,
            selected_packed,
            selected_scale_b,
            top_k=top_k,
        )

    def native_cold_gateup() -> torch.Tensor:
        packed, scale_b = _build_selected_gateup_native(
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            expert_ids,
        )
        return _native_gateup_silu(hidden_2d, packed, scale_b, top_k=top_k)

    def activation_quant_only() -> torch.Tensor:
        return quantize_fp4_m1_native(hidden_2d)[0]

    def gather_scale_only() -> torch.Tensor:
        packed, scale_b = _build_selected_gateup_native(
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            expert_ids,
        )
        # Touch both tensors so the benchmark cannot elide construction.
        return packed[:1].float().sum() + scale_b[:1].float().sum()

    ref = bf16_gateup()
    hot = native_hot_gateup()
    cold = native_cold_gateup()
    diff_hot = (hot.float() - ref.float()).abs()
    diff_cold = (cold.float() - ref.float()).abs()

    timing = {
        "bf16_gateup_ms": _bench(bf16_gateup, args.warmup, args.iters),
        "native_hot_gateup_ms": _bench(native_hot_gateup, args.warmup, args.iters),
        "native_cold_gateup_ms": _bench(native_cold_gateup, max(1, args.warmup // 2), max(10, args.iters // 4)),
        "activation_quant_only_ms": _bench(activation_quant_only, args.warmup, args.iters),
        "gather_scale_only_ms": _bench(gather_scale_only, max(1, args.warmup // 2), max(10, args.iters // 4)),
    }
    timing["native_hot_vs_bf16_ratio"] = timing["bf16_gateup_ms"] / timing["native_hot_gateup_ms"]
    timing["native_cold_vs_bf16_ratio"] = timing["bf16_gateup_ms"] / timing["native_cold_gateup_ms"]

    result = {
        "schema_version": "lynn-engine-p10g-native-fp4-selected-gateup-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "shape": {
            "top_k": top_k,
            "hidden": int(hidden_2d.shape[-1]),
            "selected_packed": list(selected_packed.shape),
            "selected_scale_b": list(selected_scale_b.shape),
        },
        "diff_hot_vs_bf16": {
            "max_abs": float(diff_hot.max().item()),
            "mean_abs": float(diff_hot.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(hot.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(hot.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "diff_cold_vs_bf16": {
            "max_abs": float(diff_cold.max().item()),
            "mean_abs": float(diff_cold.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(cold.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(cold.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "timing_ms": timing,
        "pass": bool(F.cosine_similarity(hot.float().flatten(), ref.float().flatten(), dim=0).item() > 0.98),
        "notes": [
            "hot path prebuilds the selected expert rows and scale_b for this router decision",
            "cold path includes dynamic expert gather plus scale_b construction",
            "This probe uses native FP4 tensor cores; no scalar unpack matvec is used.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
