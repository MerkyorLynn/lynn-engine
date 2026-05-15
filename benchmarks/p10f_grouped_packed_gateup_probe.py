#!/usr/bin/env python3
"""P10-F: grouped packed NVFP4 MoE gate/up probe."""
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
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--block-inter", type=int, default=64)
    ap.add_argument("--block-hidden", type=int, default=64)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, "用一句话解释 MoE active parameters")
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    hidden = h_moe.view(-1, h_moe.shape[-1])[0]

    router_logits = F.linear(hidden.unsqueeze(0), w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1)
    expert_ids = expert_indices[0].to(torch.long)
    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    def bf16_gateup() -> torch.Tensor:
        gate_up = w["mlp.experts.gate_up_proj"][expert_ids]
        gate_w, up_w = gate_up.chunk(2, dim=1)
        hidden_f = hidden.float()
        gate_out = torch.einsum("d,kid->ki", hidden_f, gate_w.float())
        up_out = torch.einsum("d,kid->ki", hidden_f, up_w.float())
        return (F.silu(gate_out) * up_out).to(torch.bfloat16)

    def packed_loop_gateup() -> torch.Tensor:
        outs = []
        for expert in expert_ids.tolist():
            y = nvfp4_matvec_packed(hidden, gate_up_packed[expert], gate_up_scale[expert], gate_up_global)
            gate, up = y.chunk(2, dim=0)
            outs.append((F.silu(gate) * up).to(torch.bfloat16))
        return torch.stack(outs, dim=0)

    def packed_grouped_gateup() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
        )

    ref = bf16_gateup()
    grouped = packed_grouped_gateup()
    looped = packed_loop_gateup()
    diff_grouped = (grouped.float() - ref.float()).abs()
    diff_looped = (looped.float() - ref.float()).abs()
    timing = {
        "bf16_gateup_ms": _bench(bf16_gateup, args.warmup, args.iters),
        "packed_loop_gateup_ms": _bench(packed_loop_gateup, max(1, args.warmup // 2), max(10, args.iters // 4)),
        "packed_grouped_gateup_ms": _bench(packed_grouped_gateup, args.warmup, args.iters),
    }
    timing["grouped_vs_bf16_ratio"] = timing["bf16_gateup_ms"] / timing["packed_grouped_gateup_ms"]
    timing["grouped_vs_loop_ratio"] = timing["packed_loop_gateup_ms"] / timing["packed_grouped_gateup_ms"]
    result = {
        "schema_version": "lynn-engine-p10f-grouped-packed-gateup-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "kernel_config": {
            "block_inter": args.block_inter,
            "block_hidden": args.block_hidden,
        },
        "diff_grouped_vs_bf16": {
            "max_abs": float(diff_grouped.max().item()),
            "mean_abs": float(diff_grouped.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(grouped.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(grouped.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "diff_grouped_vs_looped_packed": {
            "max_abs": float((grouped.float() - looped.float()).abs().max().item()),
            "mean_abs": float((grouped.float() - looped.float()).abs().mean().item()),
        },
        "diff_looped_vs_bf16": {
            "max_abs": float(diff_looped.max().item()),
            "mean_abs": float(diff_looped.mean().item()),
            "rel_l2": float(torch.linalg.vector_norm(looped.float() - ref.float()).item() / torch.linalg.vector_norm(ref.float()).item()),
            "cosine": float(F.cosine_similarity(looped.float().flatten(), ref.float().flatten(), dim=0).item()),
        },
        "timing_ms": timing,
        "pass": bool(F.cosine_similarity(grouped.float().flatten(), ref.float().flatten(), dim=0).item() > 0.98),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
