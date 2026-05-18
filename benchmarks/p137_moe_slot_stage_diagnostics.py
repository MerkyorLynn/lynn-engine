#!/usr/bin/env python3
"""P137 · Native MoE slot stage diagnostics.

Splits the p135 slot-repacked MoE candidate into:
  1. native gate/up -> BF16 inter
  2. native down weighted-sum

This separates native stage drift from slot-vs-unique reference drift before
any resident/P37 promotion work.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_ext():
    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=False)
    required = [
        "moe_slot_output_owned_bf16",
        "moe_slot_gate_up_inter_bf16",
        "moe_slot_down_weighted_sum_bf16",
    ]
    missing = [name for name in required if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"Native extension missing required symbols: {missing}")
    return ext


def metrics(ref: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    rf = ref.float().flatten()
    cf = candidate.float().flatten()
    diff = rf - cf
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12)
        ),
        "cosine": float(
            torch.dot(rf, cf)
            / (
                torch.linalg.vector_norm(rf).clamp_min(1e-12)
                * torch.linalg.vector_norm(cf).clamp_min(1e-12)
            )
        ),
    }


def bench(fn, warmup: int, iters: int) -> float:
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


def torch_slot_inter(hidden: torch.Tensor, slot_gate_up: torch.Tensor) -> torch.Tensor:
    rows = []
    for k in range(slot_gate_up.shape[0]):
        gate_up = F.linear(hidden, slot_gate_up[k])
        gate, up = gate_up.chunk(2, dim=-1)
        rows.append((F.silu(gate) * up).view(-1).to(torch.bfloat16))
    return torch.stack(rows)


def torch_slot_down(
    inter: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_down: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros((1, slot_down.shape[1]), device=inter.device, dtype=torch.bfloat16)
    for k in range(slot_down.shape[0]):
        ffn = F.linear(inter[k : k + 1].to(torch.bfloat16), slot_down[k])
        out += ffn * routing_weights[k].to(out.dtype)
    return out


def run(fixtures_dir: Path, warmup: int, iters: int) -> dict:
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    ext = load_ext()
    results = []

    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        data = load_file(str(fixtures_dir / fixture_file), device="cuda")
        hidden = data["hidden_in"].to(torch.bfloat16)
        routing = data["routing_weights"].to(torch.float32).contiguous()
        slot_gate_up = data["slot_gate_up_weight"].to(torch.bfloat16).contiguous()
        slot_down = data["slot_down_weight"].to(torch.bfloat16).contiguous()
        unique_ref = data["routed_output"].to(torch.bfloat16)

        x = hidden.view(-1).contiguous()
        torch_inter = torch_slot_inter(hidden, slot_gate_up)
        torch_slot = torch_slot_down(torch_inter, routing, slot_down)

        native_inter = ext.moe_slot_gate_up_inter_bf16(x, slot_gate_up)
        native_down_from_torch_inter = ext.moe_slot_down_weighted_sum_bf16(
            torch_inter.contiguous(), routing, slot_down
        ).view(1, -1)
        torch_down_from_native_inter = torch_slot_down(native_inter, routing, slot_down)
        native_full = ext.moe_slot_output_owned_bf16(x, routing, slot_gate_up, slot_down).view(1, -1)
        native_down_from_native_inter = ext.moe_slot_down_weighted_sum_bf16(
            native_inter.contiguous(), routing, slot_down
        ).view(1, -1)

        inter_ms = bench(
            lambda: ext.moe_slot_gate_up_inter_bf16(x, slot_gate_up),
            warmup=warmup,
            iters=iters,
        )
        down_ms = bench(
            lambda: ext.moe_slot_down_weighted_sum_bf16(torch_inter.contiguous(), routing, slot_down),
            warmup=warmup,
            iters=iters,
        )
        full_ms = bench(
            lambda: ext.moe_slot_output_owned_bf16(x, routing, slot_gate_up, slot_down),
            warmup=warmup,
            iters=iters,
        )

        result = {
            "fixture": fixture_file,
            "layer_id": entry["layer_id"],
            "prompt_id": entry["prompt_id"],
            "native_inter_vs_torch_inter": metrics(torch_inter, native_inter),
            "native_down_torch_inter_vs_torch_slot": metrics(
                torch_slot, native_down_from_torch_inter
            ),
            "torch_down_native_inter_vs_torch_slot": metrics(
                torch_slot, torch_down_from_native_inter
            ),
            "native_full_vs_torch_slot": metrics(torch_slot, native_full),
            "native_full_vs_unique": metrics(unique_ref, native_full),
            "native_down_native_inter_vs_native_full": metrics(
                native_full, native_down_from_native_inter
            ),
            "inter_ms": inter_ms,
            "down_ms": down_ms,
            "full_ms": full_ms,
        }
        results.append(result)
        print(
            f"L{entry['layer_id']:02d}/P{entry['prompt_id']:02d} "
            f"inter={result['native_inter_vs_torch_inter']['max_abs']:.2e} "
            f"down={result['native_down_torch_inter_vs_torch_slot']['max_abs']:.2e} "
            f"full={result['native_full_vs_torch_slot']['max_abs']:.2e} "
            f"ms={full_ms:.4f}",
            flush=True,
        )

    def max_metric(name: str, metric: str) -> float:
        return max(r[name][metric] for r in results)

    def min_metric(name: str, metric: str) -> float:
        return min(r[name][metric] for r in results)

    report = {
        "schema": "lynn-moe-slot-stage-diagnostics-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixtures_dir": str(fixtures_dir),
        "num_fixtures": len(results),
        "summary": {
            "native_inter_vs_torch_inter_max_abs": max_metric(
                "native_inter_vs_torch_inter", "max_abs"
            ),
            "native_down_torch_inter_vs_torch_slot_max_abs": max_metric(
                "native_down_torch_inter_vs_torch_slot", "max_abs"
            ),
            "torch_down_native_inter_vs_torch_slot_max_abs": max_metric(
                "torch_down_native_inter_vs_torch_slot", "max_abs"
            ),
            "native_full_vs_torch_slot_max_abs": max_metric(
                "native_full_vs_torch_slot", "max_abs"
            ),
            "native_full_vs_torch_slot_cosine_min": min_metric(
                "native_full_vs_torch_slot", "cosine"
            ),
            "native_full_ms_mean": sum(r["full_ms"] for r in results) / len(results),
            "native_inter_ms_mean": sum(r["inter_ms"] for r in results) / len(results),
            "native_down_ms_mean": sum(r["down_ms"] for r in results) / len(results),
        },
        "results": results,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run(Path(args.fixtures), warmup=args.warmup, iters=args.iters)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"[p137] Report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
