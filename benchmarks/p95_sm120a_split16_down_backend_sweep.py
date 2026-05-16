#!/usr/bin/env python3
"""P95: down-backend sweep after the P93 split16 gate/up backend.

P94 proved that P93 native gate/up composes with the packed down projection into
a full active-MoE numerical contract. P95 keeps the P93 gate/up output fixed and
sweeps the down half:

- production Triton down weighted-sum;
- native CUDA scalar down;
- native CUDA tile-hidden down with TILE_HIDDEN in {1,2,4,8}.

This is a routing decision probe: if a native down variant wins cleanly, the
next runtime candidate is P93 gate/up + native down. If not, the speed work must
move to fused/persistent scheduling rather than down-only tuning.
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
from triton_kernels.nvfp4_moe import nvfp4_grouped_down_weighted_sum  # noqa: E402


def _time(fn: Callable[[], torch.Tensor], repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))
    assert out is not None
    return out, times_ms


def _summarize_variant(
    name: str,
    out: torch.Tensor,
    times: list[float],
    ref: torch.Tensor,
    baseline_median: float,
) -> dict:
    diff = _diff(ref, out)
    median = statistics.median(times)
    return {
        "name": name,
        "times_ms": times,
        "mean_ms": statistics.fmean(times),
        "median_ms": median,
        "speedup_vs_triton_down_median": float(baseline_median / median),
        "diff_vs_quantized_activation_active_reference": diff,
        "contract_pass": bool(
            diff["rel_l2"] <= 0.02
            and diff["cosine"] >= 0.999
            and diff["max_abs"] <= 0.20
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p95_sm120a_split16_down_backend_sweep")
    ap.add_argument("--native-build-dir", default="/tmp/lynn_engine_native_build/p95_runtime_native")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P95 requires CUDA")
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

    inter = p93_module.split16_gateup_topk(
        act_packed_1d,
        act_scale_1d,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global.float(),
        args.scale_byte,
    )
    torch.cuda.synchronize()

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
    ).to(inter.device)
    ref_seconds = time.time() - ref_t0

    def triton_down() -> torch.Tensor:
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

    baseline_out, baseline_times = _time(triton_down, args.repeats, args.warmup)
    baseline_median = statistics.median(baseline_times)
    variants = [
        _summarize_variant(
            "triton_down",
            baseline_out,
            baseline_times,
            quantized_active_ref,
            baseline_median,
        )
    ]

    native_fns: list[tuple[str, Callable[[], torch.Tensor]]] = [
        (
            "native_down_scalar",
            lambda: ext.down_weighted_sum_scalar(
                inter,
                expert_ids,
                routing_weights,
                down_packed,
                down_scale,
                down_global,
            ),
        ),
    ]
    for tile_hidden in (1, 2, 4, 8):
        native_fns.append(
            (
                f"native_down_tile{tile_hidden}",
                lambda tile_hidden=tile_hidden: ext.down_weighted_sum_tile_scalar(
                    inter,
                    expert_ids,
                    routing_weights,
                    down_packed,
                    down_scale,
                    down_global,
                    tile_hidden,
                ),
            )
        )

    for name, fn in native_fns:
        out, times = _time(fn, args.repeats, args.warmup)
        variants.append(_summarize_variant(name, out, times, quantized_active_ref, baseline_median))

    best = min(variants, key=lambda item: item["median_ms"])
    result = {
        "schema_version": "lynn-engine-p95-sm120a-split16-down-backend-sweep-v1",
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
        "best_variant": best["name"],
        "best_median_ms": best["median_ms"],
        "best_speedup_vs_triton_down_median": best["speedup_vs_triton_down_median"],
        "contract_pass": all(item["contract_pass"] for item in variants),
        "runtime_promote": False,
        "decision": (
            "PASS: down sweep completed; use the fastest passing variant as the next composition candidate."
            if all(item["contract_pass"] for item in variants)
            else "FAIL: at least one down backend diverged from the quantized-activation active-MoE reference."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

