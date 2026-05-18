#!/usr/bin/env python3
"""P135: isolate packed-NVFP4 native MoE stage drift on p133 fixtures.

The p134 native grouped-per16 candidate is fast but non-exact. This probe splits
the routed active-MoE path into two questions:

1. Does native gate/up produce the same intermediate as Triton gate/up?
2. Given the same Triton intermediate, does native down produce the same output
   as Triton down?

Only candidates that answer these cleanly deserve P37/P25 service gates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.native_cuda import load_lynn_native_extension
from engine.nvfp4_runtime import load_grouped_nvfp4_weight
from triton_kernels.nvfp4_moe import (
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


@dataclass
class StageMetrics:
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int


def _metrics(a: torch.Tensor, b: torch.Tensor) -> StageMetrics:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af).clamp_min(1e-12)
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    max_abs = float(diff.abs().max())
    return StageMetrics(
        max_abs=max_abs,
        mean_abs=float(diff.abs().mean()),
        rel_l2=float(rel),
        cosine=float(cosine),
        exact=1 if max_abs == 0.0 else 0,
    )


def _bench(fn, warmup: int, iters: int) -> float:
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


def _load_active_triplets(model_dir: Path, layer_id: int, device: str) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_id}.mlp.experts"
    gate_packed, gate_scale, gate_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.gate_up_proj", device=device
    )
    down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.down_proj", device=device
    )
    return {
        "gate_packed": gate_packed.contiguous(),
        "gate_scale": gate_scale.contiguous(),
        "gate_global": gate_global.contiguous(),
        "down_packed": down_packed.contiguous(),
        "down_scale": down_scale.contiguous(),
        "down_global": down_global.contiguous(),
    }


def _triton_gate(hidden: torch.Tensor, expert_ids: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    return nvfp4_grouped_gate_up_silu_fast_decode(
        hidden,
        expert_ids,
        w["gate_packed"],
        w["gate_scale"],
        w["gate_global"],
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )


def _triton_down(
    inter: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    w: dict[str, torch.Tensor],
) -> torch.Tensor:
    return nvfp4_grouped_down_weighted_sum(
        inter,
        expert_ids,
        routing_weights,
        w["down_packed"],
        w["down_scale"],
        w["down_global"],
        block_hidden=8,
        block_inter=512,
        num_warps=8,
    )


def run_probe(
    fixtures_dir: Path,
    model_dir: Path,
    *,
    device: str,
    tile_inter: int,
    tile_hidden: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    ext = load_lynn_native_extension(verbose=False)
    layers = sorted(set(entry["layer_id"] for entry in manifest["fixtures"]))
    layer_weights = {
        layer: _load_active_triplets(model_dir, layer, device)
        for layer in layers
    }

    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"]:
        fixture = load_file(str(fixtures_dir / entry["fixture_file"]), device=device)
        hidden = fixture["hidden_in"].view(-1).contiguous().to(torch.bfloat16)
        expert_ids = fixture["expert_ids"].to(torch.int32).contiguous()
        routing_weights = fixture["routing_weights"].to(torch.float32).contiguous()
        expected_routed = fixture["routed_output"].to(torch.bfloat16)
        w = layer_weights[entry["layer_id"]]

        triton_inter = _triton_gate(hidden, expert_ids, w)
        native_inter = ext.gate_up_silu_tile_inter_scalar(
            hidden,
            expert_ids,
            w["gate_packed"],
            w["gate_scale"],
            w["gate_global"],
            tile_inter,
        )
        triton_down_from_triton = _triton_down(triton_inter, expert_ids, routing_weights, w)
        native_down_from_triton = ext.down_weighted_sum_tile_scalar(
            triton_inter,
            expert_ids,
            routing_weights,
            w["down_packed"],
            w["down_scale"],
            w["down_global"],
            tile_hidden,
        ).view(1, -1)
        native_full = ext.active_moe_grouped_per16_nonatomic_reference(
            hidden,
            expert_ids,
            routing_weights,
            w["gate_packed"],
            w["gate_scale"],
            w["gate_global"],
            w["down_packed"],
            w["down_scale"],
            w["down_global"],
            tile_inter,
            tile_hidden,
        ).view(1, -1)

        timings = {
            "triton_gate_ms": _bench(lambda: _triton_gate(hidden, expert_ids, w), warmup, iters),
            "native_gate_ms": _bench(
                lambda: ext.gate_up_silu_tile_inter_scalar(
                    hidden, expert_ids, w["gate_packed"], w["gate_scale"], w["gate_global"], tile_inter
                ),
                warmup,
                iters,
            ),
            "triton_down_ms": _bench(
                lambda: _triton_down(triton_inter, expert_ids, routing_weights, w),
                warmup,
                iters,
            ),
            "native_down_ms": _bench(
                lambda: ext.down_weighted_sum_tile_scalar(
                    triton_inter,
                    expert_ids,
                    routing_weights,
                    w["down_packed"],
                    w["down_scale"],
                    w["down_global"],
                    tile_hidden,
                ),
                warmup,
                iters,
            ),
        }

        rows.append(
            {
                "fixture": entry["fixture_file"],
                "layer_id": entry["layer_id"],
                "prompt_id": entry["prompt_id"],
                "gate_inter_native_vs_triton": asdict(_metrics(triton_inter, native_inter)),
                "down_native_vs_triton_on_triton_inter": asdict(_metrics(triton_down_from_triton, native_down_from_triton)),
                "triton_routed_vs_fixture": asdict(_metrics(expected_routed, triton_down_from_triton)),
                "native_full_vs_fixture": asdict(_metrics(expected_routed, native_full)),
                "timings": timings,
            }
        )

    def collect(path: str) -> list[float]:
        out: list[float] = []
        for row in rows:
            cur: Any = row
            for part in path.split("."):
                cur = cur[part]
            out.append(float(cur))
        return out

    summary = {
        "gate_inter_max_abs_max": max(collect("gate_inter_native_vs_triton.max_abs")),
        "gate_inter_rel_l2_max": max(collect("gate_inter_native_vs_triton.rel_l2")),
        "gate_inter_exact_count": int(sum(collect("gate_inter_native_vs_triton.exact"))),
        "down_on_triton_inter_max_abs_max": max(collect("down_native_vs_triton_on_triton_inter.max_abs")),
        "down_on_triton_inter_rel_l2_max": max(collect("down_native_vs_triton_on_triton_inter.rel_l2")),
        "down_on_triton_inter_exact_count": int(sum(collect("down_native_vs_triton_on_triton_inter.exact"))),
        "native_full_max_abs_max": max(collect("native_full_vs_fixture.max_abs")),
        "native_full_rel_l2_max": max(collect("native_full_vs_fixture.rel_l2")),
        "native_full_exact_count": int(sum(collect("native_full_vs_fixture.exact"))),
        "triton_gate_ms_mean": mean(collect("timings.triton_gate_ms")),
        "native_gate_ms_mean": mean(collect("timings.native_gate_ms")),
        "triton_down_ms_mean": mean(collect("timings.triton_down_ms")),
        "native_down_ms_mean": mean(collect("timings.native_down_ms")),
    }
    return {
        "schema": "p135-moe-native-stage-drift-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixtures_dir": str(fixtures_dir),
        "model_dir": str(model_dir),
        "tile_inter": tile_inter,
        "tile_hidden": tile_hidden,
        "warmup": warmup,
        "iters": iters,
        "total": len(rows),
        "summary": summary,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tile-inter", type=int, default=2)
    ap.add_argument("--tile-hidden", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    report = run_probe(
        args.fixtures,
        args.model_dir,
        device=args.device,
        tile_inter=args.tile_inter,
        tile_hidden=args.tile_hidden,
        warmup=args.warmup,
        iters=args.iters,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
