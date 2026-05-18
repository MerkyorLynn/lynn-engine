#!/usr/bin/env python3
"""P139b candidate: pretransposed TensorCore slot MoE.

Pre-computes W_fused_T and W_down_T once from p135 fixtures, then benchmarks
the zero-overhead hot path: mm + view + silu*up + bmm + bf16_reduce.

Usage:
    python benchmarks/candidates/native_slot_tensorcore_pretransposed_probe.py \
        --fixtures /root/autodl-tmp/reports/qwen36_35b/p135_repacked_fixtures_official_w4a16_slotorder
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
    if not hasattr(ext, "moe_slot_tensorcore_pretransposed"):
        raise RuntimeError("Extension missing moe_slot_tensorcore_pretransposed. Rebuild.")
    return ext


def pretranspose_weights(slot_gate_up: torch.Tensor, slot_down: torch.Tensor):
    """One-time weight preparation (done at model load, NOT per token).

    Args:
        slot_gate_up: [top_k, 1024, 2048] BF16
        slot_down: [top_k, 2048, 512] BF16

    Returns:
        W_fused_T: [2048, top_k*1024] BF16 contiguous
        W_down_T:  [top_k, 512, 2048] BF16 contiguous
    """
    top_k = slot_gate_up.size(0)
    W_fused_T = slot_gate_up.reshape(top_k * 1024, 2048).t().contiguous()
    W_down_T = slot_down.transpose(1, 2).contiguous()
    return W_fused_T, W_down_T


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

    print(f"[p139b] Pretransposed TensorCore slot MoE probe")
    print(f"[p139b] Fixtures: {fixtures_path} ({manifest['num_fixtures']})")

    ext = _load_ext()
    print(f"[p139b] Extension OK\n")

    results = []
    for entry in manifest["fixtures"]:
        fixture_data = load_file(str(fixtures_path / entry["fixture_file"]), device="cuda")
        hidden_in = fixture_data["hidden_in"].to(torch.bfloat16)
        routing_weights = fixture_data["routing_weights"].to(torch.float32)
        slot_gate_up = fixture_data["slot_gate_up_weight"].to(torch.bfloat16)
        slot_down = fixture_data["slot_down_weight"].to(torch.bfloat16)
        expected = fixture_data["routed_output"].to(torch.bfloat16)

        # Pre-transpose ONCE (simulates model load)
        W_fused_T, W_down_T = pretranspose_weights(slot_gate_up, slot_down)

        x = hidden_in.view(-1).contiguous()
        rw = routing_weights.contiguous()

        # Correctness
        out = ext.moe_slot_tensorcore_pretransposed(x, rw, W_fused_T, W_down_T)
        out_2d = out.view(1, -1)

        rf = expected.view(1, -1).float().flatten()
        cf = out_2d.float().flatten()
        diff = rf - cf
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12))
        cosine = float(torch.dot(rf, cf) / (
            torch.linalg.vector_norm(rf).clamp_min(1e-12) *
            torch.linalg.vector_norm(cf).clamp_min(1e-12)))

        # Benchmark decode-only (pretransposed weights already prepared)
        for _ in range(args.warmup):
            ext.moe_slot_tensorcore_pretransposed(x, rw, W_fused_T, W_down_T)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            ext.moe_slot_tensorcore_pretransposed(x, rw, W_fused_T, W_down_T)
        end.record()
        torch.cuda.synchronize()
        latency_ms = float(start.elapsed_time(end) / args.iters)

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

        status = "GREEN" if max_abs == 0.0 else ("AMBER" if max_abs <= 2e-3 else "RED")
        print(f"  L{entry['layer_id']:02d}/P{entry['prompt_id']:02d}: "
              f"max_abs={max_abs:.2e} cos={cosine:.10f} "
              f"ms={latency_ms:.4f} {status}")

    # Summary
    max_max_abs = max(r["max_abs"] for r in results)
    min_cosine = min(r["cosine"] for r in results)
    avg_latency = sum(r["candidate_ms"] for r in results) / len(results)
    max_latency = max(r["candidate_ms"] for r in results)

    latency_ok = avg_latency <= 0.055
    precision_ok = max_max_abs <= 0.002 and min_cosine >= 0.99998
    verdict = "AMBER_FAST_PRETRANSPOSED" if (latency_ok and precision_ok) else "CLOSED"

    print(f"\n{'='*70}")
    print(f"P139b PRETRANSPOSED TENSORCORE PROBE")
    print(f"  Fixtures:      {len(results)}")
    print(f"  max_abs_max:   {max_max_abs:.6e}  (target <= 0.002)")
    print(f"  cos_min:       {min_cosine:.10f}  (target >= 0.99998)")
    print(f"  Avg latency:   {avg_latency:.4f} ms  (target <= 0.055)")
    print(f"  Max latency:   {max_latency:.4f} ms")
    print(f"  Triton active: 0.059 ms")
    print(f"  vs Triton:     {(1-avg_latency/0.059)*100:+.1f}%")
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*70}")

    out_path = args.out or str(fixtures_path / "p139b_pretransposed_report.json")
    report = {
        "candidate": "native_slot_tensorcore_pretransposed",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_max_abs": max_max_abs,
        "min_cosine": min_cosine,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "triton_baseline_ms": 0.059,
        "latency_ok": latency_ok,
        "precision_ok": precision_ok,
        "verdict": verdict,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[p139b] Report: {out_path}")

    return 0 if verdict == "AMBER_FAST_PRETRANSPOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
