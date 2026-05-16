#!/usr/bin/env python3
"""P104: W4A8/FP8 activation sensitivity for packed active MoE.

P102 closed the BF16-activation x FP4-weight mixed-MMA shortcut. P103 proved
that SM120a can execute FP8 activation x E2M1 FP4 weight MMA instructions. This
probe answers the next quality question before any training starts:

    If the active expert FFN sees FP8-rounded activations, how much does the
    current packed NVFP4 active-MoE output drift?

Router logits/top-k stay BF16 here on purpose. The candidate W4A8 runtime should
preserve routing semantics and only lower the activation precision at expert
GEMM boundaries. We test both:

* gate_up_only: quantize the hidden vector before gate/up; keep down activation
  BF16.
* full_active: quantize the hidden vector before gate/up and quantize the
  intermediate vector before down.

The script uses existing Triton packed kernels as the numerical reference path;
it is a fake-quant quality gate, not a final FP8 MMA performance benchmark.
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

from benchmarks.p10e_packed_active_expert_probe import _load_grouped, _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
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


def _fp8_dtype(name: str) -> torch.dtype:
    if name == "e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for e4m3 fake-quant")
        return torch.float8_e4m3fn
    if name == "e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is required for e5m2 fake-quant")
        return torch.float8_e5m2
    raise ValueError(f"unknown fp8 format: {name}")


def _fake_quant_fp8(x: torch.Tensor, *, fmt: str, granularity: str, group_size: int = 16) -> torch.Tensor:
    """Round-trip x through FP8 with dynamic scale, returning BF16-shaped values."""
    dtype = _fp8_dtype(fmt)
    max_fp8 = float(torch.finfo(dtype).max)
    x32 = x.float()
    if granularity == "tensor":
        scale = (x32.abs().amax() / max_fp8).clamp_min(1e-8)
        return ((x32 / scale).to(dtype).float() * scale).to(x.dtype)
    if granularity == "row":
        if x32.ndim == 1:
            x32 = x32.view(1, -1)
            squeeze = True
        else:
            squeeze = False
        scale = (x32.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        y = ((x32 / scale).to(dtype).float() * scale).to(x.dtype)
        return y.view(-1) if squeeze else y
    if granularity == "per16":
        if x32.shape[-1] % group_size != 0:
            raise ValueError(f"last dim must be divisible by {group_size}, got {tuple(x.shape)}")
        original_shape = x32.shape
        grouped = x32.reshape(-1, original_shape[-1] // group_size, group_size)
        scale = (grouped.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
        y = ((grouped / scale).to(dtype).float() * scale).reshape(original_shape).to(x.dtype)
        return y
    raise ValueError(f"unknown granularity: {granularity}")


def _run_layer(
    runner: LynnIncrementalRunner,
    *,
    model_dir: Path,
    layer: int,
    prompt: str,
    formats: list[str],
    granularities: list[str],
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

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.down_proj",
        runner.device,
    )

    def gate_up(x: torch.Tensor) -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            x,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def down(inter: torch.Tensor) -> torch.Tensor:
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

    ref_inter = gate_up(hidden)
    ref_out = down(ref_inter)
    cases: list[dict] = []
    for fmt in formats:
        for granularity in granularities:
            hidden_q = _fake_quant_fp8(hidden, fmt=fmt, granularity=granularity)
            gate_only_inter = gate_up(hidden_q)
            gate_only_out = down(gate_only_inter)
            inter_q = _fake_quant_fp8(gate_only_inter, fmt=fmt, granularity=granularity)
            full_active_out = down(inter_q)
            gate_only_diff = _diff(ref_out, gate_only_out)
            full_active_diff = _diff(ref_out, full_active_out)
            cases.append(
                {
                    "format": fmt,
                    "granularity": granularity,
                    "diff_gate_up_only_vs_bf16_active": gate_only_diff,
                    "diff_full_active_vs_bf16_active": full_active_diff,
                    "quality_pass_relaxed": bool(
                        full_active_diff["cosine"] >= 0.999 and full_active_diff["rel_l2"] <= 0.03
                    ),
                    "quality_pass_strict": bool(
                        full_active_diff["cosine"] >= 0.9999 and full_active_diff["rel_l2"] <= 0.01
                    ),
                    "hidden_quant_diff": _diff(hidden, hidden_q),
                    "inter_quant_diff": _diff(gate_only_inter, inter_q),
                }
            )

    timings = {
        "bf16_active_ms": _bench(lambda: down(gate_up(hidden)), warmup, iters),
    }
    # Keep timing light: use the most likely production-friendly variant.
    if "e4m3" in formats and "per16" in granularities:
        timings["w4a8_e4m3_per16_active_fake_ms"] = _bench(
            lambda: down(_fake_quant_fp8(gate_up(_fake_quant_fp8(hidden, fmt="e4m3", granularity="per16")), fmt="e4m3", granularity="per16")),
            max(1, warmup // 2),
            max(5, iters // 2),
        )

    return {
        "layer": layer,
        "top_k": top_k,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.tolist()],
        "cases": cases,
        "timings_ms": timings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 W4A8 activation 量化对 MoE 专家的影响")
    ap.add_argument("--formats", nargs="+", default=["e4m3", "e5m2"], choices=["e4m3", "e5m2"])
    ap.add_argument("--granularities", nargs="+", default=["tensor", "row", "per16"], choices=["tensor", "row", "per16"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    layers = [
        _run_layer(
            runner,
            model_dir=model_dir,
            layer=layer,
            prompt=args.prompt,
            formats=args.formats,
            granularities=args.granularities,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]
    flat_cases = [case for layer in layers for case in layer["cases"]]
    best_relaxed = sorted(
        flat_cases,
        key=lambda c: (
            not c["quality_pass_relaxed"],
            -c["diff_full_active_vs_bf16_active"]["cosine"],
            c["diff_full_active_vs_bf16_active"]["rel_l2"],
        ),
    )[0]
    by_variant: dict[str, list[dict]] = {}
    for case in flat_cases:
        by_variant.setdefault(f"{case['format']}_{case['granularity']}", []).append(case)
    variant_summary = {}
    for name, cases in by_variant.items():
        variant_summary[name] = {
            "all_gate_up_only_relaxed_quality_pass": all(
                c["diff_gate_up_only_vs_bf16_active"]["cosine"] >= 0.999
                and c["diff_gate_up_only_vs_bf16_active"]["rel_l2"] <= 0.03
                for c in cases
            ),
            "all_relaxed_quality_pass": all(c["quality_pass_relaxed"] for c in cases),
            "all_strict_quality_pass": all(c["quality_pass_strict"] for c in cases),
            "min_gate_up_only_cosine": min(c["diff_gate_up_only_vs_bf16_active"]["cosine"] for c in cases),
            "max_gate_up_only_rel_l2": max(c["diff_gate_up_only_vs_bf16_active"]["rel_l2"] for c in cases),
            "min_full_active_cosine": min(c["diff_full_active_vs_bf16_active"]["cosine"] for c in cases),
            "max_full_active_rel_l2": max(c["diff_full_active_vs_bf16_active"]["rel_l2"] for c in cases),
            "max_full_active_max_abs": max(c["diff_full_active_vs_bf16_active"]["max_abs"] for c in cases),
            "max_hidden_rel_l2": max(c["hidden_quant_diff"]["rel_l2"] for c in cases),
            "max_inter_rel_l2": max(c["inter_quant_diff"]["rel_l2"] for c in cases),
        }

    full_green = any(v["all_relaxed_quality_pass"] for v in variant_summary.values())
    gate_up_green_full_near = any(
        v["all_gate_up_only_relaxed_quality_pass"]
        and v["min_full_active_cosine"] >= 0.999
        and v["max_full_active_rel_l2"] <= 0.04
        for v in variant_summary.values()
    )
    result = {
        "schema_version": "lynn-engine-p104-w4a8-active-moe-sensitivity-v1",
        "model": args.model,
        "layers": args.layers,
        "formats": args.formats,
        "granularities": args.granularities,
        "layer_cases": layers,
        "variant_summary": variant_summary,
        "best_variant_by_full_active": {
            "format": best_relaxed["format"],
            "granularity": best_relaxed["granularity"],
            "diff": best_relaxed["diff_full_active_vs_bf16_active"],
        },
        "decision": (
            "GREEN: at least one W4A8 fake-quant variant keeps full active-MoE drift inside the relaxed gate; proceed to full generate/V8/V9 and MTP pilot."
            if full_green
            else (
                "AMBER: gate/up W4A8 is inside the relaxed gate and full active-MoE is near the gate; proceed with W4A8 QAT-lite/Recovery before full-active runtime promotion."
                if gate_up_green_full_near
                else "RED: W4A8 fake-quant drifts at active-MoE isolation; do not promote without stronger adaptation."
            )
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if full_green else (1 if gate_up_green_full_near else 2)


if __name__ == "__main__":
    raise SystemExit(main())
