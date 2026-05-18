#!/usr/bin/env python3
"""P136 native candidate: slot output-owned BF16.

Eats pre-gathered slot weights directly from p135 repacked fixtures:
  - slot_gate_up_weight: [8, 1024, 2048] BF16
  - slot_down_weight:    [8, 2048, 512] BF16
  - routing_weights:     [8] float32

Calls the native CUDA kernel (moe_slot_output_owned_bf16) which skips
expert_ids indexing entirely — weights are already in dispatch order.

Usage with p136:
    python benchmarks/p136_moe_slot_repack_contract.py \
        --fixtures reports/qwen36_35b/p135_repacked_fixtures \
        --candidate-backend native_slot_output_owned_bf16

Standalone:
    python benchmarks/candidates/native_slot_output_owned_bf16.py \
        --fixtures reports/qwen36_35b/p135_repacked_fixtures
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_native_extension():
    """Load lynn native CUDA extension with slot output-owned BF16 kernel."""
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_slot_output_owned_bf16"):
        raise RuntimeError(
            "Native extension missing moe_slot_output_owned_bf16. "
            "Rebuild with moe_output_owned_bf16.cu (slot variant) included."
        )
    return ext


def moe_forward_fixture(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_gate_up: torch.Tensor,
    slot_down: torch.Tensor,
) -> torch.Tensor:
    """p136 candidate interface: run native slot BF16 MoE.

    Args:
        hidden_in: [1, 2048] BF16
        expert_ids: [8] int32 (unused by kernel, kept for interface compat)
        routing_weights: [8] float32
        slot_gate_up: [8, 1024, 2048] BF16 — pre-gathered gate+up weights
        slot_down: [8, 2048, 512] BF16 — pre-gathered down weights

    Returns:
        [1, 2048] BF16 routed-only output
    """
    ext = _load_native_extension()

    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    routing_w = routing_weights.to(torch.float32).contiguous()
    sg = slot_gate_up.contiguous()  # [8, 1024, 2048]
    sd = slot_down.contiguous()      # [8, 2048, 512]

    out = ext.moe_slot_output_owned_bf16(x, routing_w, sg, sd)
    return out.view(1, -1)


def benchmark_kernel(
    hidden_in: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_gate_up: torch.Tensor,
    slot_down: torch.Tensor,
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """Benchmark native slot kernel latency (ms)."""
    ext = _load_native_extension()

    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    routing_w = routing_weights.to(torch.float32).contiguous()
    sg = slot_gate_up.contiguous()
    sd = slot_down.contiguous()

    for _ in range(warmup):
        ext.moe_slot_output_owned_bf16(x, routing_w, sg, sd)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.moe_slot_output_owned_bf16(x, routing_w, sg, sd)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> int:
    """Standalone: run native slot candidate on all p135 fixtures."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True,
                    help="Path to p135 repacked fixtures dir")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file

    fixtures_path = Path(args.fixtures)
    manifest_path = fixtures_path / "manifest.json"
    if not manifest_path.exists():
        print(f"[slot-candidate] ERROR: manifest.json not found in {fixtures_path}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[slot-candidate] Native slot output-owned BF16 MoE")
    print(f"[slot-candidate] Fixtures: {fixtures_path} ({manifest['num_fixtures']})")

    # Build extension
    print(f"[slot-candidate] Building native extension...")
    ext = _load_native_extension()
    print(f"[slot-candidate] Extension loaded OK")

    results = []

    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        # Load repacked fixture
        fixture_data = load_file(str(fixtures_path / fixture_file), device="cuda")
        hidden_in = fixture_data["hidden_in"].to(torch.bfloat16)
        expert_ids = fixture_data["expert_ids"]
        routing_weights = fixture_data["routing_weights"]
        slot_gate_up = fixture_data["slot_gate_up_weight"].to(torch.bfloat16)
        slot_down = fixture_data["slot_down_weight"].to(torch.bfloat16)
        expected = fixture_data["routed_output"].to(torch.bfloat16)

        # Run candidate
        candidate_out = moe_forward_fixture(
            hidden_in, expert_ids, routing_weights, slot_gate_up, slot_down
        )

        # Metrics
        rf = expected.float().flatten()
        cf = candidate_out.float().flatten()
        diff = rf - cf
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12))
        cosine = float(torch.dot(rf, cf) / (
            torch.linalg.vector_norm(rf).clamp_min(1e-12) *
            torch.linalg.vector_norm(cf).clamp_min(1e-12)
        ))

        # Benchmark
        latency_ms = benchmark_kernel(
            hidden_in, routing_weights, slot_gate_up, slot_down,
            warmup=args.warmup, iters=args.iters
        )

        result = {
            "fixture": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_l2": rel_l2,
            "cosine": cosine,
            "exact": 1 if max_abs == 0.0 else 0,
            "candidate_ms": latency_ms,
        }
        results.append(result)

        status = "GREEN" if max_abs == 0.0 else ("AMBER" if max_abs <= 1e-3 else "RED")
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d}: "
            f"max_abs={max_abs:.2e} cos={cosine:.8f} rel_l2={rel_l2:.2e} "
            f"latency={latency_ms:.4f}ms {status}"
        )

    # Summary
    all_exact = all(r["exact"] == 1 for r in results)
    all_pass = all(r["max_abs"] <= 1e-3 and r["cosine"] >= 0.999999 for r in results)
    avg_latency = sum(r["candidate_ms"] for r in results) / len(results) if results else 0
    max_latency = max(r["candidate_ms"] for r in results) if results else 0
    max_max_abs = max(r["max_abs"] for r in results) if results else 0

    print(f"\n{'='*70}")
    print(f"SLOT CANDIDATE SUMMARY: native_slot_output_owned_bf16")
    print(f"  Fixtures:      {len(results)}")
    print(f"  All exact:     {'YES' if all_exact else 'NO'}")
    print(f"  All pass:      {'YES' if all_pass else 'NO'} (max_abs<=1e-3, cos>=0.999999)")
    print(f"  Max max_abs:   {max_max_abs:.2e}")
    print(f"  Avg latency:   {avg_latency:.4f} ms")
    print(f"  Max latency:   {max_latency:.4f} ms")
    print(f"  Triton active: 0.059 ms (baseline)")
    if avg_latency < 0.059:
        print(f"  VERDICT: FASTER ({(1 - avg_latency/0.059)*100:.0f}% gain)")
    elif avg_latency < 0.075:
        print(f"  VERDICT: AMBER research")
    else:
        print(f"  VERDICT: SLOWER — close")
    print(f"{'='*70}")

    # Write report
    out_path = args.out or str(fixtures_path / "native_slot_output_owned_bf16_report.json")
    report = {
        "candidate": "native_slot_output_owned_bf16",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_exact": all_exact,
        "all_pass_1e3": all_pass,
        "max_max_abs": max_max_abs,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "triton_baseline_ms": 0.059,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[slot-candidate] Report: {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
