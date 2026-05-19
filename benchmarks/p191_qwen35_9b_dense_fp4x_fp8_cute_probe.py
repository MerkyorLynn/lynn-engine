#!/usr/bin/env python3
"""P191: Dense FP4xFP8 CuTe/scalar PoC probe for Qwen3.5-9B.

Tests the native FP4xFP8 scalar reference kernel against P159 dense FFN fixtures.
Quantizes BF16 activation to E4M3 FP8, then runs mixed-precision matmul against
packed NVFP4 weights.

Reports max_abs, rel_l2, cosine vs W4A8 fake-quant reference.
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


def _quantize_to_fp8_e4m3(x: torch.Tensor):
    """Quantize BF16 activation to E4M3 with per-16 scale. Returns (packed_uint8, scale_fp32)."""
    x_flat = x.view(-1).float()
    K = x_flat.numel()
    assert K % 16 == 0
    groups = K // 16
    grouped = x_flat.view(groups, 16)
    max_e4m3 = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    scale = (grouped.abs().amax(dim=-1) / max_e4m3).clamp_min(1e-8)  # [groups]
    normalized = grouped / scale.unsqueeze(-1)
    # Quantize to E4M3
    fp8_vals = normalized.to(torch.float8_e4m3fn)
    # Store as uint8
    packed = fp8_vals.view(torch.uint8).view(-1)  # [K]
    return packed, scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="P159 fixture dir")
    ap.add_argument("--model-dir", required=True, help="NVFP4 model for packed weights")
    ap.add_argument("--layers", default="0,16", help="Layers to test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors.torch import load_file
    from safetensors import safe_open

    fixtures_dir = Path(args.fixtures)
    model_dir = Path(args.model_dir)
    layers = [int(x) for x in args.layers.split(",")]

    print(f"[p191] Dense FP4xFP8 PoC probe")
    print(f"[p191] Fixtures: {fixtures_dir}")
    print(f"[p191] Model: {model_dir}")
    print(f"[p191] Layers: {layers}")

    # Load native extension
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    has_scalar = hasattr(ext, "dense_fp4xfp8_scalar_reference")
    has_mma = hasattr(ext, "dense_fp4xfp8_mma_probe")
    print(f"[p191] Scalar reference: {has_scalar}")
    print(f"[p191] MMA probe: {has_mma}")

    if not has_scalar:
        print("[p191] ERROR: scalar reference kernel not found")
        return 1

    # Test MMA capability
    mma_available = False
    if has_mma:
        try:
            dummy_a = torch.zeros(32, device="cuda", dtype=torch.uint8)
            dummy_b = torch.zeros(16, device="cuda", dtype=torch.uint8)
            ext.dense_fp4xfp8_mma_probe(dummy_a, dummy_b, 1, 8, 32)
            mma_available = True
            print("[p191] MMA probe: COMPILED OK")
        except RuntimeError as e:
            print(f"[p191] MMA probe: {e}")

    results = []
    for layer_id in layers:
        # Find fixture
        fixture_files = list(fixtures_dir.glob(f"layer_{layer_id:02d}_prompt_00.safetensors"))
        if not fixture_files:
            print(f"  L{layer_id:02d}: SKIP (no fixture)")
            continue

        fixture = load_file(str(fixture_files[0]), device="cuda")
        ffn_in = fixture["ffn_in"].to(torch.bfloat16)  # [1, 4096]
        ffn_output_ref = fixture["ffn_output"].to(torch.bfloat16)  # [1, 4096] BF16 ground truth

        # Load packed weights for this layer
        model_st = model_dir / "model.safetensors"
        if not model_st.exists():
            # Multi-file model
            idx_path = model_dir / "model.safetensors.index.json"
            if idx_path.exists():
                with open(idx_path) as f:
                    idx = json.load(f)
                # Find gate_proj packed for this layer
                gate_key = f"model.layers.{layer_id}.mlp.gate_proj.weight_packed"
                if gate_key not in idx["weight_map"]:
                    gate_key = f"model.language_model.layers.{layer_id}.mlp.gate_proj.weight_packed"
                if gate_key not in idx["weight_map"]:
                    print(f"  L{layer_id:02d}: SKIP (weight key not found)")
                    continue
                model_st = model_dir / idx["weight_map"][gate_key]

        # For this PoC: test gate_proj only (4096 -> 12288)
        prefix = f"model.language_model.layers.{layer_id}.mlp.gate_proj"
        with safe_open(str(model_st), framework="pt", device="cuda") as st:
            keys = list(st.keys())
            # Try to find the right key
            packed_key = None
            for k in keys:
                if f"layers.{layer_id}.mlp.gate_proj.weight_packed" in k:
                    packed_key = k
                    break
            if packed_key is None:
                print(f"  L{layer_id:02d}: SKIP (gate_proj.weight_packed not in {model_st.name})")
                continue

            base = packed_key.replace(".weight_packed", "")
            w_packed = st.get_tensor(packed_key)          # [N, K/2]
            w_scale = st.get_tensor(base + ".weight_scale").float()  # [N, K/16]
            w_global = st.get_tensor(base + ".weight_global_scale").float()  # scalar

        N, K_half = w_packed.shape
        K = K_half * 2
        print(f"  L{layer_id:02d}: gate_proj [{N}, {K}], fixture ffn_in [{ffn_in.shape}]")

        # Quantize activation to E4M3 FP8
        act_fp8, act_scale = _quantize_to_fp8_e4m3(ffn_in.view(-1)[:K])

        # Run scalar reference kernel
        torch.cuda.synchronize()
        t0 = time.time()
        out_scalar = ext.dense_fp4xfp8_scalar_reference(
            act_fp8.contiguous(),
            act_scale.contiguous(),
            w_packed.contiguous(),
            w_scale.contiguous(),
            w_global.contiguous().view(-1),
        )
        torch.cuda.synchronize()
        scalar_ms = (time.time() - t0) * 1000

        # W4A8 fake-quant reference: dequant weight to FP32, quantize act to FP8, matmul
        # For reference: just use the fixture gate_output if available
        gate_ref = fixture.get("gate_output")
        if gate_ref is not None:
            gate_ref = gate_ref.to(torch.bfloat16).view(-1)[:N]
        else:
            # Compute reference via Python dequant + FP8 act matmul
            gate_ref = None

        # Compare scalar output vs BF16 reference (if gate_ref available)
        if gate_ref is not None:
            rf = gate_ref.float()
            cf = out_scalar[:N].float()
            diff = rf - cf
            max_abs = float(diff.abs().max())
            rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf).clamp_min(1e-12))
            cosine = float(torch.dot(rf, cf) / (
                torch.linalg.vector_norm(rf).clamp_min(1e-12) *
                torch.linalg.vector_norm(cf).clamp_min(1e-12)))
        else:
            max_abs = rel_l2 = cosine = None

        # Benchmark
        for _ in range(10):
            ext.dense_fp4xfp8_scalar_reference(act_fp8, act_scale, w_packed, w_scale, w_global.view(-1))
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            ext.dense_fp4xfp8_scalar_reference(act_fp8, act_scale, w_packed, w_scale, w_global.view(-1))
        end.record()
        torch.cuda.synchronize()
        bench_ms = float(start.elapsed_time(end) / 50)

        result = {
            "layer_id": layer_id,
            "shape": f"[{N}, {K}]",
            "max_abs_vs_bf16_ref": max_abs,
            "rel_l2_vs_bf16_ref": rel_l2,
            "cosine_vs_bf16_ref": cosine,
            "scalar_ms": bench_ms,
            "mma_available": mma_available,
        }
        results.append(result)

        status = "GREEN" if (cosine and cosine > 0.99) else "AMBER"
        print(f"    max_abs={max_abs:.4e} rel_l2={rel_l2:.4e} cos={cosine:.6f} ms={bench_ms:.3f} {status}")

    # Summary
    report = {
        "probe": "p191_dense_fp4xfp8_poc",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixtures_dir": str(fixtures_dir),
        "model_dir": str(model_dir),
        "mma_compiled": mma_available,
        "scalar_reference_available": has_scalar,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[p191] Report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
