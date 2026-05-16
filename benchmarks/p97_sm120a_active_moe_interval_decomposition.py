#!/usr/bin/env python3
"""P97: interval-decompose active-MoE gate/up and down compositions.

P95 proved that native_down_tile1 is much faster than Triton down in isolation,
but P96 showed that P93 gate/up + native_down_tile1 still does not beat the
current Triton active path. P97 measures CUDA-event intervals inside the same
stream:

    gate/up interval -> down interval -> total interval

for multiple active-MoE compositions. The goal is to identify whether the lost
speedup is gate/up cost, down cost, or two-stage scheduling overhead.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
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
from benchmarks.p93_sm120a_split16_gateup_topk_backend_probe import (  # noqa: E402
    _build_module as _build_p93_module,
    _diff,
    _prepare_path,
    _quantized_activation_reference,
)
from benchmarks.p94_sm120a_split16_active_moe_composition_probe import _down_reference  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.native_cuda import (  # noqa: E402
    discover_native_include_paths,
    load_lynn_native_extension,
    native_cuda_extra_cuda_cflags,
)
from engine.nvfp4_runtime import _quantize_activation_to_fp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def _measure_interval(
    make_inter: Callable[[], torch.Tensor],
    make_out: Callable[[torch.Tensor], torch.Tensor],
    *,
    repeats: int,
    warmup: int,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    out = None
    for _ in range(warmup):
        inter = make_inter()
        out = make_out(inter)
    torch.cuda.synchronize()

    gate_ms: list[float] = []
    down_ms: list[float] = []
    total_ms: list[float] = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e0.record()
        inter = make_inter()
        e1.record()
        out = make_out(inter)
        e2.record()
        torch.cuda.synchronize()
        gate_ms.append(float(e0.elapsed_time(e1)))
        down_ms.append(float(e1.elapsed_time(e2)))
        total_ms.append(float(e0.elapsed_time(e2)))
    assert out is not None
    return out, {"gate_ms": gate_ms, "down_ms": down_ms, "total_ms": total_ms}


def _summarize(
    name: str,
    out: torch.Tensor,
    timings: dict[str, list[float]],
    ref: torch.Tensor,
    baseline_total_median: float | None,
) -> dict:
    total_median = _median(timings["total_ms"])
    return {
        "name": name,
        "gate_median_ms": _median(timings["gate_ms"]),
        "down_median_ms": _median(timings["down_ms"]),
        "total_median_ms": total_median,
        "gate_mean_ms": float(statistics.fmean(timings["gate_ms"])),
        "down_mean_ms": float(statistics.fmean(timings["down_ms"])),
        "total_mean_ms": float(statistics.fmean(timings["total_ms"])),
        "speedup_vs_baseline_total_median": (
            None if baseline_total_median is None else float(baseline_total_median / total_median)
        ),
        "timings_ms": timings,
        "diff_vs_quantized_activation_active_reference": _diff(ref, out),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p97_sm120a_active_moe_interval_decomp")
    ap.add_argument("--native-build-dir", default="/tmp/lynn_engine_native_build/p97_runtime_native")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P97 requires CUDA")
    os.environ.setdefault("LYNN_NATIVE_CUDA_ARCH", "sm_120a")
    _prepare_path()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
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
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )
    down_packed, down_scale, down_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.down_proj",
        runner.device,
    )
    act_packed, act_scale = _quantize_activation_to_fp4(hidden.view(1, -1))
    act_packed_1d = act_packed[0].contiguous()
    act_scale_1d = act_scale[0].float().contiguous()

    t0 = time.time()
    p93_module = _build_p93_module(Path(args.build_dir), args.verbose)
    ext = load_lynn_native_extension(build_dir=args.native_build_dir, verbose=args.verbose)
    build_s = time.time() - t0

    ref_t0 = time.time()
    quantized_inter_ref = _quantized_activation_reference(
        act_packed_1d,
        act_scale_1d,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
    )
    quantized_active_ref = _down_reference(
        quantized_inter_ref,
        expert_ids,
        routing_weights,
        down_packed,
        down_scale,
        down_global,
    ).to(hidden.device)
    ref_seconds = time.time() - ref_t0

    def gate_triton() -> torch.Tensor:
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

    def gate_p93() -> torch.Tensor:
        return p93_module.split16_gateup_topk(
            act_packed_1d,
            act_scale_1d,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global.float(),
            args.scale_byte,
        )

    def down_triton(inter: torch.Tensor) -> torch.Tensor:
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

    def down_native_scalar(inter: torch.Tensor) -> torch.Tensor:
        return ext.down_weighted_sum_scalar(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
        )

    def down_native_tile1(inter: torch.Tensor) -> torch.Tensor:
        return ext.down_weighted_sum_tile_scalar(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            1,
        )

    cases: list[tuple[str, Callable[[], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = [
        ("triton_gateup_triton_down", gate_triton, down_triton),
        ("p93_gateup_triton_down", gate_p93, down_triton),
        ("p93_gateup_native_down_scalar", gate_p93, down_native_scalar),
        ("p93_gateup_native_down_tile1", gate_p93, down_native_tile1),
    ]

    variants = []
    baseline_total = None
    for name, make_inter, make_out in cases:
        out, timings = _measure_interval(make_inter, make_out, repeats=args.repeats, warmup=args.warmup)
        if baseline_total is None:
            baseline_total = _median(timings["total_ms"])
        variants.append(_summarize(name, out, timings, quantized_active_ref, baseline_total))

    best = min(variants, key=lambda item: item["total_median_ms"])
    quantized_activation_variants = [
        item for item in variants if item["name"] != "triton_gateup_triton_down"
    ]
    contract_pass = all(
        item["diff_vs_quantized_activation_active_reference"]["rel_l2"] <= 0.02
        and item["diff_vs_quantized_activation_active_reference"]["cosine"] >= 0.999
        and item["diff_vs_quantized_activation_active_reference"]["max_abs"] <= 0.20
        for item in quantized_activation_variants
    )
    result = {
        "schema_version": "lynn-engine-p97-sm120a-active-moe-interval-decomposition-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "top_k": top_k,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "reference_seconds": ref_seconds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "variants": variants,
        "contract_scope": "quantized_activation_variants_only",
        "baseline_triton_bf16_activation_is_timing_reference_only": True,
        "best_variant": best["name"],
        "best_total_median_ms": best["total_median_ms"],
        "contract_pass": contract_pass,
        "runtime_promote": False,
        "decision": (
            "PASS: interval decomposition completed; use gate/down medians to choose the next fusion target."
            if contract_pass
            else "FAIL: at least one interval variant diverged from the quantized-activation active-MoE reference."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if contract_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
