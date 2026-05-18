#!/usr/bin/env python3
"""P140 candidate: packed NVFP4 slot MoE probe.

Consumes p138 packed fixtures directly. Compares output against p135 slotorder
routed_output (BF16 reference). Reports max_abs/cosine/latency per fixture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_ext():
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_slot_packed_nvfp4_probe"):
        raise RuntimeError("Extension missing moe_slot_packed_nvfp4_probe. Rebuild.")
    return ext


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-fixtures", required=True,
                    help="p138 packed fixture dir")
    ap.add_argument("--ref-fixtures", required=True,
                    help="p135 slotorder fixture dir (has routed_output)")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.ref_fixtures)

    with open(packed_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"[p140] Packed NVFP4 slot MoE probe")
    print(f"[p140] Packed: {packed_dir} ({manifest['num_fixtures']})")
    print(f"[p140] Ref: {ref_dir}")

    ext = _load_ext()
    print(f"[p140] Extension OK\n")

    results = []
    for entry in manifest["fixtures"]:
        packed_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        # Load packed fixture
        pd = load_file(str(packed_dir / packed_file), device="cuda")
        x = pd["hidden_in"].view(-1).to(torch.bfloat16).contiguous()
        rw = pd["routing_weights"].to(torch.float32).contiguous()
        gu_packed = pd["slot_gate_up_packed"].contiguous()
        gu_scale = pd["slot_gate_up_scale"].to(torch.float16).contiguous()
        gu_global = pd["slot_gate_up_global_scale"].to(torch.float16).contiguous()
        d_packed = pd["slot_down_packed"].contiguous()
        d_scale = pd["slot_down_scale"].to(torch.float16).contiguous()
        d_global = pd["slot_down_global_scale"].to(torch.float16).contiguous()

        # Load BF16 reference (p135 slotorder)
        ref_file = packed_file.replace("_slot_packed", "_slots")
        ref_path = ref_dir / ref_file
        if not ref_path.exists():
            # Try alternative naming
            ref_file = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slots.safetensors"
            ref_path = ref_dir / ref_file
        if not ref_path.exists():
            print(f"  L{layer_id:02d}/P{prompt_id:02d}: SKIP (no ref {ref_file})")
            continue

        rd = load_file(str(ref_path), device="cuda")
        expected = rd["routed_output"].to(torch.bfloat16).view(1, -1)

        # Run native packed kernel
        out = ext.moe_slot_packed_nvfp4_probe(
            x, rw, gu_packed, gu_scale, gu_global, d_packed, d_scale, d_global
        )
        out_2d = out.view(1, -1)

        # Metrics
        rf = expected.float().flatten()
        cf = out_2d.float().flatten()
        diff = rf - cf
        max_abs = float(diff.abs().max())
        mean_abs = float(diff.abs().mean())
        rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12))
        cosine = float(torch.dot(rf, cf) / (
            torch.linalg.vector_norm(rf).clamp_min(1e-12) *
            torch.linalg.vector_norm(cf).clamp_min(1e-12)))

        # Benchmark
        for _ in range(args.warmup):
            ext.moe_slot_packed_nvfp4_probe(
                x, rw, gu_packed, gu_scale, gu_global, d_packed, d_scale, d_global)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            ext.moe_slot_packed_nvfp4_probe(
                x, rw, gu_packed, gu_scale, gu_global, d_packed, d_scale, d_global)
        end.record()
        torch.cuda.synchronize()
        latency_ms = float(start.elapsed_time(end) / args.iters)

        result = {
            "fixture": packed_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rel_l2": rel_l2,
            "cosine": cosine,
            "candidate_ms": latency_ms,
        }
        results.append(result)

        status = "GREEN" if max_abs <= 1e-3 else ("AMBER" if max_abs <= 3e-3 else "RED")
        print(f"  L{layer_id:02d}/P{prompt_id:02d}: "
              f"max_abs={max_abs:.2e} cos={cosine:.8f} "
              f"ms={latency_ms:.4f} {status}")

    if not results:
        print("[p140] No results!")
        return 1

    # Summary
    max_max_abs = max(r["max_abs"] for r in results)
    min_cosine = min(r["cosine"] for r in results)
    avg_latency = sum(r["candidate_ms"] for r in results) / len(results)
    max_latency = max(r["candidate_ms"] for r in results)

    gate_up_pass = max_max_abs <= 3e-3
    strict_pass = max_max_abs <= 1e-3
    cosine_pass = min_cosine >= 0.9999

    if strict_pass and cosine_pass:
        verdict = "GREEN_STAGE"
    elif gate_up_pass and cosine_pass:
        verdict = "AMBER_STAGE"
    else:
        verdict = "CLOSED"

    print(f"\n{'='*70}")
    print(f"P140 PACKED NVFP4 SLOT MOE PROBE")
    print(f"  Fixtures:      {len(results)}")
    print(f"  max_abs_max:   {max_max_abs:.6e}  (GREEN<=1e-3, AMBER<=3e-3)")
    print(f"  cos_min:       {min_cosine:.8f}  (target>=0.9999)")
    print(f"  Avg latency:   {avg_latency:.4f} ms")
    print(f"  Max latency:   {max_latency:.4f} ms")
    print(f"\n  VERDICT: {verdict}")
    print(f"{'='*70}")

    out_path = args.out or str(packed_dir / "p140_packed_nvfp4_probe_report.json")
    report = {
        "candidate": "native_slot_packed_nvfp4_probe",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_max_abs": max_max_abs,
        "min_cosine": min_cosine,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "verdict": verdict,
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[p140] Report: {out_path}")

    return 0 if verdict != "CLOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
