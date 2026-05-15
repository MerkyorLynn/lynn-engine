#!/usr/bin/env python3
"""P16: native-FP4 down-projection overcompute probe.

Current active expert decode is:

  gate/up scalar-unpack kernel -> inter[top_k, 512]
  down scalar-unpack kernel    -> hidden[2048]

The ideal kernel is grouped native FP4 down that computes each expert's down
projection only for its own inter row. PyTorch `_scaled_mm` cannot express that
grouped diagonal directly, but it can compute an overcomplete matrix:

  [top_k, 512] @ [top_k * hidden, 512]^T -> [top_k, top_k * hidden]

We then gather the diagonal expert blocks and routing-weight sum them.

This deliberately does extra cross-expert work. It is a probe, not a production
path. A positive timing signal would justify writing a true grouped native-FP4
down kernel.
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
from engine.dequant import E2M1_TO_FLOAT  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_down_weighted_sum, nvfp4_grouped_gate_up_silu  # noqa: E402


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


def _build_selected_down_native(
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_packed = down_packed[expert_ids].reshape(-1, down_packed.shape[-1]).contiguous()
    selected_scale = down_scale[expert_ids].reshape(-1, down_scale.shape[-1]).contiguous()
    effective_scale = selected_scale.float() / down_global.to(selected_scale.device).float()
    scale_b = _compact_scale_to_swizzled_fp8(
        effective_scale,
        outer_dim=int(selected_packed.shape[0]),
        k=int(selected_packed.shape[1] * 2),
    )
    return selected_packed, scale_b


def _quantize_activation_to_native_fp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[M, K]` activations and swizzle scale_a for `_scaled_mm`."""
    if x.ndim != 2:
        raise ValueError(f"x must be [M, K], got {tuple(x.shape)}")
    if x.shape[1] % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {x.shape[1]}")
    table = E2M1_TO_FLOAT.to(device=x.device)
    x32 = x.float()
    m, k = x32.shape
    groups = k // 16
    xg = x32.reshape(m, groups, 16)
    scale = (xg.abs().amax(dim=-1) / float(table[-1])).clamp_min(1e-8)
    normalized = xg.abs() / scale.unsqueeze(-1)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (xg < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale_a = _compact_scale_to_swizzled_fp8(scale, outer_dim=m, k=k)
    return packed, scale_a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
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
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]

    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)
    top_k = int(expert_ids.numel())
    hidden_size = int(hidden.numel())

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

    inter = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        gate_up_packed,
        gate_up_scale,
        gate_up_global,
        block_inter=8,
        block_hidden=64,
    )
    selected_down_packed, selected_down_scale_b = _build_selected_down_native(
        down_packed,
        down_scale,
        down_global,
        expert_ids,
    )

    def scalar_down() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            down_packed,
            down_scale,
            down_global,
            block_hidden=8,
            block_inter=256,
        )

    def native_overcompute_down() -> torch.Tensor:
        act_packed, scale_a = _quantize_activation_to_native_fp4(inter.contiguous())
        y = torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            selected_down_packed.view(torch.float4_e2m1fn_x2).t(),
            scale_a=scale_a,
            scale_b=selected_down_scale_b,
            out_dtype=torch.float16,
        ).float()
        pieces = []
        for slot in range(top_k):
            start = slot * hidden_size
            end = start + hidden_size
            pieces.append(y[slot, start:end] * routing_weights[slot])
        return torch.stack(pieces, dim=0).sum(dim=0).to(torch.bfloat16)

    ref = scalar_down()
    native = native_overcompute_down()
    diff = (native.float() - ref.float()).abs()
    timing = {
        "scalar_down_ms": _bench(scalar_down, args.warmup, args.iters),
        "native_overcompute_down_ms": _bench(native_overcompute_down, args.warmup, args.iters),
    }
    timing["native_vs_scalar_ratio"] = timing["scalar_down_ms"] / timing["native_overcompute_down_ms"]

    result = {
        "schema_version": "lynn-engine-p16-native-down-overcompute-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "shape": {
            "top_k": top_k,
            "hidden": hidden_size,
            "intermediate": int(inter.shape[1]),
            "selected_down_packed": list(selected_down_packed.shape),
            "overcompute_factor": top_k,
        },
        "diff_native_vs_scalar_down": {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(native.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(native.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "timing_ms": timing,
        "pass": bool(F.cosine_similarity(native.float().flatten(), ref.float().flatten(), dim=0).item() > 0.995),
        "notes": [
            "This computes cross-expert products and keeps only diagonal blocks.",
            "A true grouped native-FP4 down kernel should avoid the overcompute_factor.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
