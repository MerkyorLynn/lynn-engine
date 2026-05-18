#!/usr/bin/env python3
"""P142 · Graph-safe pretransposed MoE fixture probe.

Tests the caller-owned-scratch V3 ABI on p138/p135 fixtures.
Preallocates all scratch once, then benchmarks the zero-alloc hot path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_nvfp4(packed, scale, global_scale, device="cuda"):
    """Dequant packed NVFP4 → BF16 (load-time operation)."""
    table = E2M1.to(device)
    low = (packed & 0x0F).int()
    high = ((packed >> 4) & 0x0F).int()
    low_val = table[low & 7] * (1 - 2 * ((low >> 3) & 1).float())
    high_val = table[high & 7] * (1 - 2 * ((high >> 3) & 1).float())
    K = packed.shape[-1] * 2
    result = torch.zeros(*packed.shape[:-1], K, device=device, dtype=torch.float32)
    result[..., 0::2] = low_val
    result[..., 1::2] = high_val
    inv_g = 1.0 / global_scale.float().item()
    se = scale.float().unsqueeze(-1).expand(*scale.shape, 16).reshape(*packed.shape[:-1], K)
    return (result * se * inv_g).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--ref-fixtures", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file
    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.ref_fixtures)

    with open(packed_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"[p142] Graph-safe pretransposed MoE V3 probe")
    print(f"[p142] Packed: {packed_dir} ({manifest['num_fixtures']})")
    print(f"[p142] Ref: {ref_dir}")

    ext = load_lynn_native_extension(verbose=False)
    assert hasattr(ext, "moe_packed_pretransposed_graphsafe_v3"), "Missing V3 kernel"
    print(f"[p142] Extension OK\n")

    # Preallocate scratch (ONE TIME — simulates resident runner)
    device = "cuda"
    gate_up_scratch = torch.empty(1, 8192, device=device, dtype=torch.bfloat16)
    inter_scratch = torch.empty(8, 1, 512, device=device, dtype=torch.bfloat16)
    down_scratch = torch.empty(8, 1, 2048, device=device, dtype=torch.bfloat16)
    out_buf = torch.empty(2048, device=device, dtype=torch.bfloat16)

    results = []
    for entry in manifest["fixtures"]:
        lid = entry["layer_id"]
        pid = entry["prompt_id"]

        pd = load_file(str(packed_dir / entry["fixture_file"]), device=device)
        x = pd["hidden_in"].to(torch.bfloat16).view(1, 2048).contiguous()
        rw = pd["routing_weights"].to(torch.float32).contiguous()

        # Load-time: dequant + pretranspose
        gu_bf16 = dequant_nvfp4(pd["slot_gate_up_packed"].to(device),
                                pd["slot_gate_up_scale"].to(device),
                                pd["slot_gate_up_global_scale"].to(device), device)
        down_bf16 = dequant_nvfp4(pd["slot_down_packed"].to(device),
                                  pd["slot_down_scale"].to(device),
                                  pd["slot_down_global_scale"].to(device), device)
        W_fused_T = gu_bf16.reshape(8 * 1024, 2048).t().contiguous()
        W_down_T = down_bf16.transpose(1, 2).contiguous()

        # Reference
        ref_file = f"layer_{lid:02d}_prompt_{pid:02d}_slots.safetensors"
        ref_path = ref_dir / ref_file
        if not ref_path.exists():
            continue
        rd = load_file(str(ref_path), device=device)
        expected = rd["routed_output"].to(torch.bfloat16).view(1, 2048)

        # Run V3 graph-safe kernel
        ext.moe_packed_pretransposed_graphsafe_v3(
            x, rw, W_fused_T, W_down_T,
            gate_up_scratch, inter_scratch, down_scratch, out_buf)
        out_2d = out_buf.view(1, -1)

        # Metrics
        rf = expected.float().flatten()
        cf = out_2d.float().flatten()
        diff = rf - cf
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        cosine = float(torch.dot(rf, cf) / (
            torch.linalg.vector_norm(rf).clamp_min(1e-12) *
            torch.linalg.vector_norm(cf).clamp_min(1e-12)))

        # Benchmark (graph-safe: no alloc in hot path)
        for _ in range(args.warmup):
            ext.moe_packed_pretransposed_graphsafe_v3(
                x, rw, W_fused_T, W_down_T,
                gate_up_scratch, inter_scratch, down_scratch, out_buf)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            ext.moe_packed_pretransposed_graphsafe_v3(
                x, rw, W_fused_T, W_down_T,
                gate_up_scratch, inter_scratch, down_scratch, out_buf)
        end.record()
        torch.cuda.synchronize()
        ms = float(start.elapsed_time(end) / args.iters)

        result = {"layer_id": lid, "prompt_id": pid,
                  "max_abs": max_abs, "mean_abs": mean_abs,
                  "cosine": cosine, "candidate_ms": ms}
        results.append(result)

        status = "GREEN" if max_abs <= 1e-3 else ("AMBER" if max_abs <= 2e-3 else "RED")
        print(f"  L{lid:02d}/P{pid:02d}: max_abs={max_abs:.2e} cos={cosine:.8f} ms={ms:.4f} {status}")

    if not results:
        print("[p142] No results!")
        return 1

    max_max_abs = max(r["max_abs"] for r in results)
    min_cos = min(r["cosine"] for r in results)
    avg_ms = sum(r["candidate_ms"] for r in results) / len(results)
    max_ms = max(r["candidate_ms"] for r in results)

    precision_ok = max_max_abs <= 0.001953125  # not worse than p141 v2
    latency_ok = avg_ms <= 0.055
    verdict = "AMBER_GRAPHSAFE" if (precision_ok and latency_ok) else "CLOSED"

    print(f"\n{'='*70}")
    print(f"P142 GRAPH-SAFE PRETRANSPOSED V3")
    print(f"  Fixtures:      {len(results)}")
    print(f"  max_abs_max:   {max_max_abs:.6e}")
    print(f"  cos_min:       {min_cos:.8f}")
    print(f"  Avg latency:   {avg_ms:.4f} ms (target <= 0.055)")
    print(f"  Max latency:   {max_ms:.4f} ms")
    print(f"  VERDICT:       {verdict}")
    print(f"{'='*70}")

    out_path = args.out or "reports/qwen36_35b/p142_graphsafe_fixture_report.json"
    report = {
        "candidate": "moe_packed_pretransposed_graphsafe_v3",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_max_abs": max_max_abs,
        "min_cosine": min_cos,
        "avg_latency_ms": avg_ms,
        "max_latency_ms": max_ms,
        "precision_ok": precision_ok,
        "latency_ok": latency_ok,
        "verdict": verdict,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[p142] Report: {out_path}")
    return 0 if verdict != "CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
