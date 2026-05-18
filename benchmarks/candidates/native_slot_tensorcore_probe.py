#!/usr/bin/env python3
"""P139 candidate: TensorCore batched-bmm slot MoE probe.

2 batched cuBLAS launches instead of 16 sequential mm calls.
Matches slot-order PyTorch semantics (route_w as BF16 multiply).

Usage:
    python benchmarks/candidates/native_slot_tensorcore_probe.py \
        --fixtures /path/to/p135_repacked_fixtures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_ext():
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_slot_tensorcore_probe"):
        raise RuntimeError("Extension missing moe_slot_tensorcore_probe. Rebuild.")
    return ext


def moe_forward_fixture(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_gate_up: torch.Tensor,
    slot_down: torch.Tensor,
) -> torch.Tensor:
    """p136 candidate interface."""
    ext = _load_ext()
    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    rw = routing_weights.to(torch.float32).contiguous()
    sg = slot_gate_up.contiguous().to(torch.bfloat16)
    sd = slot_down.contiguous().to(torch.bfloat16)
    out = ext.moe_slot_tensorcore_probe(x, rw, sg, sd)
    return out.view(1, -1)


def benchmark_kernel(
    hidden_in: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_gate_up: torch.Tensor,
    slot_down: torch.Tensor,
    warmup: int = 50,
    iters: int = 500,
) -> float:
    ext = _load_ext()
    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    rw = routing_weights.to(torch.float32).contiguous()
    sg = slot_gate_up.contiguous().to(torch.bfloat16)
    sd = slot_down.contiguous().to(torch.bfloat16)

    for _ in range(warmup):
        ext.moe_slot_tensorcore_probe(x, rw, sg, sd)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.moe_slot_tensorcore_probe(x, rw, sg, sd)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file

    fixtures_path = Path(args.fixtures)
    with open(fixtures_path / "manifest.json") as f:
        manifest = json.load(f)

    print(f"[tc-probe] TensorCore batched-bmm slot MoE probe")
    print(f"[tc-probe] Fixtures: {fixtures_path} ({manifest['num_fixtures']})")

    ext = _load_ext()
    print(f"[tc-probe] Extension OK")

    results = []
    for entry in manifest["fixtures"]:
        fixture_data = load_file(str(fixtures_path / entry["fixture_file"]), device="cuda")
        hidden_in = fixture_data["hidden_in"].to(torch.bfloat16)
        routing_weights = fixture_data["routing_weights"]
        slot_gate_up = fixture_data["slot_gate_up_weight"].to(torch.bfloat16)
        slot_down = fixture_data["slot_down_weight"].to(torch.bfloat16)
        expected = fixture_data["routed_output"].to(torch.bfloat16)

        candidate_out = moe_forward_fixture(hidden_in, None, routing_weights, slot_gate_up, slot_down)

        rf = expected.float().flatten()
        cf = candidate_out.float().flatten()
        diff = rf - cf
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12))
        cosine = float(torch.dot(rf, cf) / (
            torch.linalg.vector_norm(rf).clamp_min(1e-12) *
            torch.linalg.vector_norm(cf).clamp_min(1e-12)))

        latency_ms = benchmark_kernel(hidden_in, routing_weights, slot_gate_up, slot_down,
                                      warmup=args.warmup, iters=args.iters)

        result = {
            "fixture": entry["fixture_file"],
            "layer_id": entry["layer_id"],
            "prompt_id": entry["prompt_id"],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_l2": rel_l2,
            "cosine": cosine,
            "exact": 1 if max_abs == 0.0 else 0,
            "candidate_ms": latency_ms,
        }
        results.append(result)

        status = "GREEN" if max_abs == 0.0 else ("AMBER" if max_abs <= 1e-3 else "RED")
        print(f"  L{entry['layer_id']:02d}/P{entry['prompt_id']:02d}: "
              f"max_abs={max_abs:.2e} cos={cosine:.10f} "
              f"ms={latency_ms:.4f} {status}")

    # Summary
    all_exact = all(r["exact"] == 1 for r in results)
    max_max_abs = max(r["max_abs"] for r in results)
    min_cosine = min(r["cosine"] for r in results)
    avg_latency = sum(r["candidate_ms"] for r in results) / len(results)
    max_latency = max(r["candidate_ms"] for r in results)

    print(f"\n{'='*70}")
    print(f"TENSORCORE PROBE: batched-bmm slot MoE")
    print(f"  Fixtures:      {len(results)}")
    print(f"  All exact:     {'YES' if all_exact else 'NO'}")
    print(f"  max_abs_max:   {max_max_abs:.6e}")
    print(f"  cos_min:       {min_cosine:.10f}")
    print(f"  Avg latency:   {avg_latency:.4f} ms")
    print(f"  Max latency:   {max_latency:.4f} ms")
    print(f"  Triton active: 0.059 ms")
    speedup = (0.059 - avg_latency) / 0.059 * 100
    print(f"  vs Triton:     {speedup:+.0f}%")
    verdict = "GREEN" if avg_latency < 0.059 else ("AMBER" if avg_latency < 0.075 else "RED")
    print(f"  VERDICT:       {verdict}")
    print(f"{'='*70}")

    out_path = args.out or str(fixtures_path / "native_slot_tensorcore_probe_report.json")
    report = {
        "candidate": "native_slot_tensorcore_probe",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_exact": all_exact,
        "max_max_abs": max_max_abs,
        "min_cosine": min_cosine,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "triton_baseline_ms": 0.059,
        "verdict": verdict,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"[tc-probe] Report: {out_path}")

    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
