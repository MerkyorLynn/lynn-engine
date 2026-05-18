#!/usr/bin/env python3
"""P128: feed the current Triton active-MoE boundary from the MoE sidecar.

This is not a promotion benchmark. It proves the repacked sidecar is a direct
kernel input ABI: active gate/up + down tensors from the sidecar produce the
same output as tensors loaded through the old manifest path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.moe_repack_sidecar import load_moe_repack_layer
from engine.nvfp4_runtime import load_grouped_nvfp4_weight
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
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


def _active_from_aliases(
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


def _to_device_aliases(aliases: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: tensor.to(device).contiguous() for key, tensor in aliases.items()}


def _time_ms(fn, iterations: int) -> dict[str, Any]:
    times: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "iterations": iterations,
    }


def check_layer(
    model_dir: Path,
    sidecar_dir: Path,
    layer: int,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed + layer)
    device = torch.device("cuda")
    hidden = torch.randn(2048, device=device, dtype=torch.bfloat16)
    expert_ids = torch.tensor([0, 7, 31, 63, 95, 127, 191, 255], device=device, dtype=torch.int32)
    routing_weights = torch.softmax(torch.randn(8, device=device, dtype=torch.float32), dim=0).contiguous()

    prefix = f"model.language_model.layers.{layer}"
    gate_packed, gate_scale, gate_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.mlp.experts.gate_up_proj", device="cpu"
    )
    down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.mlp.experts.down_proj", device="cpu"
    )
    manifest_aliases = _to_device_aliases({
        "mlp.experts._gate_up_packed": gate_packed,
        "mlp.experts._gate_up_scale": gate_scale,
        "mlp.experts._gate_up_global_scale": gate_global,
        "mlp.experts._down_packed": down_packed,
        "mlp.experts._down_scale": down_scale,
        "mlp.experts._down_global_scale": down_global,
    }, device)
    side = load_moe_repack_layer(sidecar_dir, layer, device="cpu")
    side_aliases = _to_device_aliases(side.active_aliases(), device)

    # Compile/warm both paths before measuring.
    for _ in range(3):
        _active_from_aliases(hidden, expert_ids, routing_weights, manifest_aliases)
        _active_from_aliases(hidden, expert_ids, routing_weights, side_aliases)
    torch.cuda.synchronize()

    out_manifest = _active_from_aliases(hidden, expert_ids, routing_weights, manifest_aliases)
    out_sidecar = _active_from_aliases(hidden, expert_ids, routing_weights, side_aliases)
    torch.cuda.synchronize()
    diff = (out_manifest.float() - out_sidecar.float()).abs()

    manifest_timing = _time_ms(
        lambda: _active_from_aliases(hidden, expert_ids, routing_weights, manifest_aliases),
        iterations,
    )
    sidecar_timing = _time_ms(
        lambda: _active_from_aliases(hidden, expert_ids, routing_weights, side_aliases),
        iterations,
    )
    return {
        "layer": layer,
        "ok": bool(torch.equal(out_manifest, out_sidecar)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "manifest_timing": manifest_timing,
        "sidecar_timing": sidecar_timing,
        "speed_ratio_manifest_over_sidecar": manifest_timing["mean_ms"] / max(sidecar_timing["mean_ms"], 1e-9),
        "sidecar_strides": {
            "gate_up_packed": list(side_aliases["mlp.experts._gate_up_packed"].stride()),
            "gate_up_scale": list(side_aliases["mlp.experts._gate_up_scale"].stride()),
            "down_packed": list(side_aliases["mlp.experts._down_packed"].stride()),
            "down_scale": list(side_aliases["mlp.experts._down_scale"].stride()),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    ap.add_argument("--layers", default="0,20,39")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--seed", type=int, default=127)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows = [
        check_layer(args.model_dir, args.sidecar_dir, layer, iterations=args.iterations, seed=args.seed)
        for layer in _parse_layers(args.layers)
    ]
    report = {
        "schema_version": "p128-moe-repack-triton-boundary-v1",
        "model_dir": str(args.model_dir),
        "sidecar_dir": str(args.sidecar_dir),
        "layers": rows,
        "ok": all(row["ok"] for row in rows),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
