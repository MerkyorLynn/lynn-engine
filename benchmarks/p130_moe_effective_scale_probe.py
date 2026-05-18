#!/usr/bin/env python3
"""P130: probe MoE effective-scale repack kernels.

This keeps the active-MoE math contract unchanged at the Python boundary:

    gate/up -> bf16 inter store -> down -> weighted sum

The candidate only precomputes `scale / global_scale` once per loaded layer and
uses Triton kernels that consume those effective scales directly.  It is a
small repack/kernel-boundary step before native grouped FP4 math; it must pass
strict local parity before any generation gate is worth running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.moe_repack_sidecar import load_moe_repack_layer
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_down_weighted_sum_effective_scale,
    nvfp4_grouped_gate_up_silu_fast_decode,
    nvfp4_grouped_gate_up_silu_fast_decode_effective_scale,
)


def _parse_layers(raw: str) -> list[int]:
    out: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(item))
    return sorted(out)


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


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict[str, Any]:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "exact": bool(torch.equal(ref, out)),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _active_ref(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    aliases: dict[str, torch.Tensor],
) -> torch.Tensor:
    inter = nvfp4_grouped_gate_up_silu_fast_decode(
        hidden,
        expert_ids,
        aliases["mlp.experts._gate_up_packed"],
        aliases["mlp.experts._gate_up_scale"],
        aliases["mlp.experts._gate_up_global_scale"],
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )
    return nvfp4_grouped_down_weighted_sum(
        inter,
        expert_ids,
        routing_weights,
        aliases["mlp.experts._down_packed"],
        aliases["mlp.experts._down_scale"],
        aliases["mlp.experts._down_global_scale"],
        block_hidden=8,
        block_inter=512,
        num_warps=8,
    )


def _active_effective(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    aliases: dict[str, torch.Tensor],
    effective: dict[str, torch.Tensor],
) -> torch.Tensor:
    inter = nvfp4_grouped_gate_up_silu_fast_decode_effective_scale(
        hidden,
        expert_ids,
        aliases["mlp.experts._gate_up_packed"],
        effective["gate_up"],
        aliases["mlp.experts._gate_up_global_scale"],
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )
    return nvfp4_grouped_down_weighted_sum_effective_scale(
        inter,
        expert_ids,
        routing_weights,
        aliases["mlp.experts._down_packed"],
        effective["down"],
        aliases["mlp.experts._down_global_scale"],
        block_hidden=8,
        block_inter=512,
        num_warps=8,
    )


def _check_layer(sidecar_dir: Path, layer: int, *, warmup: int, iters: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed + layer)
    device = torch.device("cuda")
    side = load_moe_repack_layer(sidecar_dir, layer, device=device)
    aliases = {key: tensor.contiguous() for key, tensor in side.active_aliases().items()}
    effective = {
        "gate_up": (
            aliases["mlp.experts._gate_up_scale"].float()
            / aliases["mlp.experts._gate_up_global_scale"].float()
        ).contiguous(),
        "down": (
            aliases["mlp.experts._down_scale"].float()
            / aliases["mlp.experts._down_global_scale"].float()
        ).contiguous(),
    }
    hidden = torch.randn(2048, device=device, dtype=torch.bfloat16)
    expert_ids = torch.tensor([0, 7, 31, 63, 95, 127, 191, 255], device=device, dtype=torch.int32)
    routing_weights = torch.softmax(torch.randn(8, device=device, dtype=torch.float32), dim=0).contiguous()

    ref = _active_ref(hidden, expert_ids, routing_weights, aliases)
    eff = _active_effective(hidden, expert_ids, routing_weights, aliases, effective)
    torch.cuda.synchronize()
    ref_ms = _bench(lambda: _active_ref(hidden, expert_ids, routing_weights, aliases), warmup, iters)
    eff_ms = _bench(lambda: _active_effective(hidden, expert_ids, routing_weights, aliases, effective), warmup, iters)
    return {
        "layer": layer,
        "diff_effective_vs_ref": _diff(ref, eff),
        "ref_active_ms": ref_ms,
        "effective_active_ms": eff_ms,
        "speedup_ref_over_effective": ref_ms / max(eff_ms, 1e-9),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    ap.add_argument("--layers", default="0,4,8,16,20,28,32,36,39")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=130)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [
        _check_layer(args.sidecar_dir, layer, warmup=args.warmup, iters=args.iters, seed=args.seed)
        for layer in _parse_layers(args.layers)
    ]
    report = {
        "schema_version": "p130-moe-effective-scale-probe-v1",
        "sidecar_dir": str(args.sidecar_dir),
        "layers": rows,
        "summary": {
            "all_exact": all(row["diff_effective_vs_ref"]["exact"] for row in rows),
            "max_abs": max(row["diff_effective_vs_ref"]["max_abs"] for row in rows),
            "max_rel_l2": max(row["diff_effective_vs_ref"]["rel_l2"] for row in rows),
            "min_cosine": min(row["diff_effective_vs_ref"]["cosine"] for row in rows),
            "mean_ref_active_ms": mean(row["ref_active_ms"] for row in rows),
            "mean_effective_active_ms": mean(row["effective_active_ms"] for row in rows),
            "mean_speedup_ref_over_effective": mean(row["speedup_ref_over_effective"] for row in rows),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["summary"]["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
