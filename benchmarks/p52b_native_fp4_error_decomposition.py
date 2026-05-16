#!/usr/bin/env python3
"""P52-B: decompose native-FP4 gate/up error sources.

P52-A showed that replacing only gate/up with `torch._scaled_mm` is both slower
and numerically loose.  This probe splits the error into:

1. activation FP4 quantization;
2. activation FP4 + FP8 weight-scale quantization;
3. residual `_scaled_mm` / layout / accumulation difference.

It compares all candidates against the current Triton grouped gate/up output on
true decode-state hidden vectors.
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
from benchmarks.p52_native_gateup_active_moe_sensitivity import (  # noqa: E402
    _build_selected_gateup_native,
    _native_selected_gateup,
)
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu  # noqa: E402


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


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


def _activation_fp4_qdq_like_scaled_mm(hidden: torch.Tensor) -> torch.Tensor:
    """Quantize/dequantize activation like the M=1 `_scaled_mm` helper.

    The quantizer writes activation scales as FP8 e4m3, so the QDQ simulation
    must round the per-16 activation scale through FP8 too.
    """
    table = _E2M1.to(device=hidden.device)
    x = hidden.float().reshape(-1, 16)
    scale = (x.abs().amax(dim=-1) / 6.0).clamp_min(1.0e-8)
    scale = scale.to(torch.float8_e4m3fn).float()
    normalized = x.abs() / scale[:, None]
    mag = torch.argmin((normalized[..., None] - table.view(1, 1, -1)).abs(), dim=-1)
    signed = table[mag] * torch.where(x < 0, -1.0, 1.0)
    return (signed * scale[:, None]).reshape_as(hidden).to(hidden.dtype)


def _effective_fp8_weight_scale(gate_up_scale: torch.Tensor, gate_up_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    effective = (gate_up_scale.float() / gate_up_global.to(gate_up_scale.device).float()).to(torch.float8_e4m3fn).float()
    global_one = torch.ones_like(gate_up_global, dtype=torch.float32, device=gate_up_global.device)
    return effective.contiguous(), global_one.contiguous()


def _run_layer(
    runner: LynnIncrementalRunner,
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
    hidden_qdq = _activation_fp4_qdq_like_scaled_mm(hidden).contiguous()
    hidden_2d = hidden.reshape(1, -1).contiguous()
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    expert_ids_i32 = expert_indices[0].to(torch.int32).contiguous()
    expert_ids_i64 = expert_indices[0].to(torch.long).contiguous()

    gate_up_packed = w["mlp.experts._gate_up_packed"]
    gate_up_scale = w["mlp.experts._gate_up_scale"]
    gate_up_global = w["mlp.experts._gate_up_global_scale"]
    gate_up_scale_eff_fp8, gate_up_global_one = _effective_fp8_weight_scale(gate_up_scale, gate_up_global)
    selected_packed, scale_b = _build_selected_gateup_native(gate_up_packed, gate_up_scale, gate_up_global, expert_ids_i64)

    def triton_ref() -> torch.Tensor:
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

    def triton_activation_qdq() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden_qdq,
            expert_ids_i32,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def triton_activation_qdq_weight_fp8() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden_qdq,
            expert_ids_i32,
            gate_up_packed,
            gate_up_scale_eff_fp8,
            gate_up_global_one,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def native_scaled_mm() -> torch.Tensor:
        return _native_selected_gateup(hidden_2d, selected_packed, scale_b, top_k=top_k)

    ref = triton_ref()
    act_qdq = triton_activation_qdq()
    act_qdq_wfp8 = triton_activation_qdq_weight_fp8()
    native = native_scaled_mm()
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids_i64.tolist()],
        "diff_activation_qdq_vs_ref": _diff(ref, act_qdq),
        "diff_activation_qdq_weight_fp8_vs_ref": _diff(ref, act_qdq_wfp8),
        "diff_native_scaled_mm_vs_ref": _diff(ref, native),
        "diff_native_scaled_mm_vs_activation_qdq_weight_fp8": _diff(act_qdq_wfp8, native),
        "timings_ms": {
            "triton_ref_ms": _bench(triton_ref, warmup, iters),
            "triton_activation_qdq_ms": _bench(triton_activation_qdq, warmup, iters),
            "triton_activation_qdq_weight_fp8_ms": _bench(triton_activation_qdq_weight_fp8, warmup, iters),
            "native_scaled_mm_ms": _bench(native_scaled_mm, warmup, iters),
            "activation_qdq_build_ms": _bench(lambda: _activation_fp4_qdq_like_scaled_mm(hidden), warmup, iters),
        },
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

    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(runner, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    result = {
        "schema_version": "lynn-engine-p52b-native-fp4-error-decomposition-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "min_activation_qdq_cosine": min(c["diff_activation_qdq_vs_ref"]["cosine"] for c in cases),
            "min_activation_qdq_weight_fp8_cosine": min(
                c["diff_activation_qdq_weight_fp8_vs_ref"]["cosine"] for c in cases
            ),
            "min_native_cosine": min(c["diff_native_scaled_mm_vs_ref"]["cosine"] for c in cases),
            "min_native_vs_qdq_weight_fp8_cosine": min(
                c["diff_native_scaled_mm_vs_activation_qdq_weight_fp8"]["cosine"] for c in cases
            ),
            "mean_triton_ref_ms": sum(c["timings_ms"]["triton_ref_ms"] for c in cases) / len(cases),
            "mean_native_scaled_mm_ms": sum(c["timings_ms"]["native_scaled_mm_ms"] for c in cases) / len(cases),
        },
        "interpretation": [
            "If activation_qdq already matches native drift, activation FP4 is the primary error source.",
            "If native differs strongly from activation_qdq_weight_fp8, the remaining issue is scale layout or accumulation contract.",
            "This is a diagnostic for P52 kernel design, not a production backend.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
