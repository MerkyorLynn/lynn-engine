#!/usr/bin/env python3
"""P94: compose P93 split16 gate/up with active-MoE down projection.

P93 proved a production-shaped top-k gate/up backend:

    quantized hidden -> native FP4 gate/up -> inter[top_k, 512]

P94 keeps that gate/up backend unchanged and composes it with the existing
packed down weighted-sum kernel. This is intentionally a two-stage probe, not a
runtime promotion. The goal is to prove the full active-MoE numerical contract
before attempting a down-fused or persistent grouped kernel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

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
    _e2m1_values,
    _prepare_path,
    _quantized_activation_reference,
    _unpack_codes,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.native_cuda import discover_native_include_paths, native_cuda_extra_cuda_cflags  # noqa: E402
from engine.nvfp4_runtime import _quantize_activation_to_fp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


def _time(fn, repeats: int, warmup: int) -> tuple[torch.Tensor, list[float]]:
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


def _down_reference(
    inter_bf16: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
) -> torch.Tensor:
    """CPU reference for down weighted-sum using BF16-rounded intermediate."""
    inter_cpu = inter_bf16.detach().cpu().to(torch.bfloat16).float().contiguous()
    expert_ids_cpu = expert_ids.detach().cpu().to(torch.long)
    routing_cpu = routing_weights.detach().cpu().float()
    packed_cpu = down_packed.detach().cpu().contiguous()
    scale_cpu = down_scale.detach().cpu().float().contiguous()
    global_val = float(down_global.detach().cpu().float().reshape(-1)[0].item())
    out = torch.empty((2048,), dtype=torch.float32)
    for hidden in range(2048):
        acc = 0.0
        for slot, expert in enumerate(expert_ids_cpu.tolist()):
            codes = _unpack_codes(packed_cpu[expert, hidden], 512)
            vals = _e2m1_values(codes) * (scale_cpu[expert, hidden].repeat_interleave(16) / global_val)
            acc += float(routing_cpu[slot].item()) * float(torch.sum(inter_cpu[slot] * vals).item())
        out[hidden] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/lynn_engine_native_build/p94_sm120a_split16_active_moe_composition")
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--scale-byte", type=int, default=127)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("P94 requires CUDA")
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
    module = _build_p93_module(Path(args.build_dir), args.verbose)
    build_s = time.time() - t0

    def native_composed() -> torch.Tensor:
        inter = module.split16_gateup_topk(
            act_packed_1d,
            act_scale_1d,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global.float(),
            args.scale_byte,
        )
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

    def current_triton_active() -> torch.Tensor:
        inter = nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )
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

    native, native_times = _time(native_composed, args.repeats, args.warmup)
    triton, triton_times = _time(current_triton_active, args.repeats, args.warmup)

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
    ).to(native.device)
    ref_seconds = time.time() - ref_t0

    diff_quantized_ref = _diff(quantized_active_ref, native)
    diff_triton_bf16_act = _diff(triton, native)
    contract_pass = bool(
        diff_quantized_ref["rel_l2"] <= 0.02
        and diff_quantized_ref["cosine"] >= 0.999
        and diff_quantized_ref["max_abs"] <= 0.20
    )
    result = {
        "schema_version": "lynn-engine-p94-sm120a-split16-active-moe-composition-v1",
        "model": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "top_k": top_k,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "scale_byte": args.scale_byte,
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "include_paths": discover_native_include_paths(),
        "cuda_cflags": native_cuda_extra_cuda_cflags(),
        "build_seconds": build_s,
        "reference_seconds": ref_seconds,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "native_composed_times_ms": native_times,
        "native_composed_mean_ms": statistics.fmean(native_times),
        "native_composed_median_ms": statistics.median(native_times),
        "current_triton_times_ms": triton_times,
        "current_triton_mean_ms": statistics.fmean(triton_times),
        "current_triton_median_ms": statistics.median(triton_times),
        "native_vs_triton_speedup_median": float(statistics.median(triton_times) / statistics.median(native_times)),
        "diff_native_vs_quantized_activation_active_reference": diff_quantized_ref,
        "diff_native_vs_current_triton_bf16_activation_active": diff_triton_bf16_act,
        "contract_pass": contract_pass,
        "runtime_promote": False,
        "decision": (
            "PASS: P93 native gate/up composes with down into a full active-MoE contract; next gate is fused/non-atomic scheduling."
            if contract_pass
            else "FAIL: native gate/up plus down diverged from the quantized-activation active-MoE reference."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if contract_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

