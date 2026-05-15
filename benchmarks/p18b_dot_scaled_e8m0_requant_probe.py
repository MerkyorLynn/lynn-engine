#!/usr/bin/env python3
"""P18-B: test an engine-native e8m0/group32 FP4 quant artifact shape.

P18-A showed that cheaply folding Lynn's existing per-16 scales into
`tl.dot_scaled` e8m0/group32 scales is too inaccurate. P18-B asks the next
question:

> If we re-quantize weights directly into e8m0/group32, is the numerical quality
> good enough for a future engine-native artifact?

This probe quantizes only the selected top-k gate/up rows from the resident
BF16 shadow, runs the raw dot through the same P17/P18 Triton `dot_scaled`
kernel, and compares against BF16 gate/up. It is not a production runtime path;
it is an artifact-format feasibility gate.
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

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from benchmarks.p18_dot_scaled_scale_bridge_probe import _bench, _dot_scaled_selected, _silu_inter  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


_E2M1_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _quantize_e2m1_e8m0_group32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `[rows,K]` to packed E2M1 bytes plus e8m0 group32 scale bytes."""
    if x.ndim != 2:
        raise ValueError(f"x must be [rows,K], got {tuple(x.shape)}")
    if x.shape[1] % 32 != 0:
        raise ValueError(f"K must be divisible by 32, got {x.shape[1]}")
    table = _E2M1_TABLE.to(x.device)
    x32 = x.float()
    rows, k = x32.shape
    groups = k // 32
    grouped = x32.reshape(rows, groups, 32)
    max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-30)
    raw_scale = (max_abs / float(table[-1])).clamp_min(1e-30)
    exponent = torch.round(torch.log2(raw_scale)).clamp(-126, 127)
    scale = torch.pow(torch.full_like(exponent, 2.0), exponent)
    normalized = grouped.abs() / scale.unsqueeze(-1)
    mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs(), dim=-1)
    sign = (grouped < 0).to(torch.uint8) * 8
    codes = (mag.to(torch.uint8) | sign).reshape(rows, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scale_bytes = (exponent + 127).clamp(0, 255).to(torch.uint8).contiguous()
    return packed, scale_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--block-k-packed", type=int, default=256)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

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

    gate_up = w["mlp.experts.gate_up_proj"][expert_ids].contiguous()
    selected_weight = gate_up.reshape(-1, gate_up.shape[-1]).contiguous()

    act_packed, act_scale = _quantize_e2m1_e8m0_group32(hidden_2d)
    weight_packed, weight_scale = _quantize_e2m1_e8m0_group32(selected_weight)
    act_packed_1d = act_packed[0].contiguous()
    act_scale_1d = act_scale[0].contiguous()

    def bf16_raw() -> torch.Tensor:
        return torch.matmul(selected_weight.float(), hidden.float())

    def dot_scaled_raw() -> torch.Tensor:
        return _dot_scaled_selected(
            act_packed_1d,
            act_scale_1d,
            weight_packed,
            weight_scale,
            block_k_packed=args.block_k_packed,
            block_n=args.block_n,
        )

    ref_raw = bf16_raw()
    cand_raw = dot_scaled_raw()
    ref_inter = _silu_inter(ref_raw, top_k=top_k)
    cand_inter = _silu_inter(cand_raw, top_k=top_k)
    raw_diff = cand_raw.float() - ref_raw.float()
    inter_diff = cand_inter.float() - ref_inter.float()
    result = {
        "schema_version": "lynn-engine-p18b-dot-scaled-e8m0-requant-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "shape": {
            "top_k": top_k,
            "hidden": int(hidden_2d.shape[-1]),
            "selected_weight": list(selected_weight.shape),
            "weight_packed": list(weight_packed.shape),
            "weight_scale_e8m0": list(weight_scale.shape),
        },
        "raw": {
            "max_abs": float(raw_diff.abs().max().item()),
            "mean_abs": float(raw_diff.abs().mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(raw_diff).item() / torch.linalg.vector_norm(ref_raw.float()).item()),
            "cosine": float(F.cosine_similarity(cand_raw.float(), ref_raw.float(), dim=0).item()),
        },
        "inter": {
            "max_abs": float(inter_diff.abs().max().item()),
            "mean_abs": float(inter_diff.abs().mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(inter_diff).item() / torch.linalg.vector_norm(ref_inter.float()).item()),
            "cosine": float(F.cosine_similarity(cand_inter.float().flatten(), ref_inter.float().flatten(), dim=0).item()),
        },
        "timing_ms": {
            "bf16_raw_matmul_ms": _bench(bf16_raw, args.warmup, args.iters),
            "dot_scaled_raw_ms": _bench(dot_scaled_raw, args.warmup, args.iters),
        },
        "pass": bool(
            F.cosine_similarity(cand_inter.float().flatten(), ref_inter.float().flatten(), dim=0).item() > 0.995
            and (torch.linalg.vector_norm(inter_diff).item() / torch.linalg.vector_norm(ref_inter.float()).item()) < 0.08
        ),
        "notes": [
            "This quantizes selected BF16 gate/up rows into e8m0/group32, not the current Lynn per-16 artifact.",
            "A pass means a second engine-native artifact may be viable; a fail points toward custom CUDA scale handling or finer scale groups.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
