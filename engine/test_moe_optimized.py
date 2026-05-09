"""
Lynn Engine · Phase 3.2 · MoE optimization correctness test.

Compares moe_forward_decode_optimized + moe_forward_decode_bmm against the
baseline _moe_forward (full 256-iteration loop). For decode T=1, all three
should produce identical output (bit-exact within FP rounding).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, "/work")
    from engine.loader import load_qwen36_layer
    from engine.full_forward import _moe_forward
    from engine.moe_optimized import (
        moe_forward_decode_optimized,
        moe_forward_decode_bmm,
    )

    print(f"Loading layer {args.layer} ...", flush=True)
    weights, _ = load_qwen36_layer(args.model, args.layer, device=args.device,
                                   dequant_dtype=torch.bfloat16)

    cfg = {"num_experts": 256, "num_experts_per_tok": 8}

    torch.manual_seed(42)
    h = torch.randn(1, 1, 2048, device=args.device, dtype=torch.bfloat16)

    # Baseline (256-iter loop)
    print("\nRunning baseline (256-iter loop) ...", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_base = _moe_forward(h, weights, cfg)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_base = (time.time() - t0) * 1000

    # Phase 3.2.1: active-experts loop
    print("Running Phase 3.2.1 (active-experts loop) ...", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_p321 = moe_forward_decode_optimized(h, weights, cfg)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_p321 = (time.time() - t0) * 1000

    # Phase 3.2.2: bmm batched matmul
    print("Running Phase 3.2.2 (bmm) ...", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    out_p322 = moe_forward_decode_bmm(h, weights, cfg)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t_p322 = (time.time() - t0) * 1000

    # Warm up + measure 5 runs each for stable timing
    print("\nWarmup + 5-run timing average ...", flush=True)
    for _ in range(2):
        _ = _moe_forward(h, weights, cfg)
        _ = moe_forward_decode_optimized(h, weights, cfg)
        _ = moe_forward_decode_bmm(h, weights, cfg)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    def time_fn(fn, n=5):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            _ = fn(h, weights, cfg)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000

    t_base_avg = time_fn(_moe_forward)
    t_p321_avg = time_fn(moe_forward_decode_optimized)
    t_p322_avg = time_fn(moe_forward_decode_bmm)

    # Correctness
    diff_p321 = (out_p321 - out_base).float().abs()
    diff_p322 = (out_p322 - out_base).float().abs()
    base_mag = out_base.float().abs().mean().item()

    print("\n" + "=" * 60)
    print(f"MoE forward — layer {args.layer}, decode T=1")
    print("=" * 60)
    print(f"  baseline (256-iter):   {t_base_avg:6.1f} ms/call  (5-run avg)")
    print(f"  P3.2.1 active-experts: {t_p321_avg:6.1f} ms/call   "
          f"speedup {t_base_avg/t_p321_avg:.2f}x")
    print(f"  P3.2.2 bmm:            {t_p322_avg:6.1f} ms/call   "
          f"speedup {t_base_avg/t_p322_avg:.2f}x")
    print()
    print(f"Correctness vs baseline (ref_mag={base_mag:.4f}):")
    print(f"  P3.2.1: max_diff={diff_p321.max().item():.3e}  "
          f"rel={diff_p321.max().item()/max(base_mag,1e-8)*100:.3f}%")
    print(f"  P3.2.2: max_diff={diff_p322.max().item():.3e}  "
          f"rel={diff_p322.max().item()/max(base_mag,1e-8)*100:.3f}%")

    ok_p321 = diff_p321.max().item() / max(base_mag, 1e-8) < 0.05   # 5% rel
    ok_p322 = diff_p322.max().item() / max(base_mag, 1e-8) < 0.05
    print(f"\n  P3.2.1 correctness: {'✅ PASS' if ok_p321 else '❌ FAIL'}")
    print(f"  P3.2.2 correctness: {'✅ PASS' if ok_p322 else '❌ FAIL'}")

    sys.exit(0 if ok_p321 and ok_p322 else 1)


if __name__ == "__main__":
    main()
