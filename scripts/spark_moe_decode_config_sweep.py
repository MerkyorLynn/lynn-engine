#!/usr/bin/env python3
"""Spark decode-kernel config sweep for the NVFP4 MoE gate_up + down GEMVs.

The production config (BLOCK_*/num_warps) is LOCKED to the R6000-best profile
(the LYNN_MOE_FAST_FIXED guard). Spark (GB10 sm_121) has a different SM count /
memory system, so the R6000 config may be suboptimal. This sweeps the real
kernels at the real 35B-A3B shapes with random valid-shaped packed weights (no
model load -> timing only; values irrelevant for latency) and reports the best
Spark config vs the R6000-locked one. A stackable decode multiplier toward 60.

Run on Spark:  python scripts/spark_moe_decode_config_sweep.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_down_weighted_sum,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
)

E, K = 256, 8
HID, INTER = HIDDEN_SIZE, INTERMEDIATE_SIZE
GU = 2 * INTER


def _bench(fn, iters=60, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def _mk_scale(shape, dev):
    for dt in (torch.float8_e4m3fn, torch.bfloat16, torch.float32):
        try:
            return (torch.rand(shape, device=dev) * 0.1 + 0.01).to(dt), dt
        except Exception:
            continue
    raise RuntimeError("no scale dtype worked")


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    cap = torch.cuda.get_device_capability()
    torch.manual_seed(0)
    print(f"device sm_{cap[0]}{cap[1]}  HID={HID} INTER={INTER} GU={GU} E={E} K={K}")

    x = (torch.randn(HID, device=dev) * 0.4).bfloat16()
    eids = torch.randint(0, E, (K,), device=dev, dtype=torch.int32)
    rw = torch.softmax(torch.randn(K, device=dev), dim=-1).bfloat16()
    gu_packed = torch.randint(0, 256, (E, GU, HID // 2), device=dev, dtype=torch.uint8)
    d_packed = torch.randint(0, 256, (E, HID, INTER // 2), device=dev, dtype=torch.uint8)
    gu_scale, sdt = _mk_scale((E, GU, HID // 16), dev)
    d_scale, _ = _mk_scale((E, HID, INTER // 16), dev)
    gu_g = torch.tensor(0.02, device=dev)
    d_g = torch.tensor(0.02, device=dev)
    print(f"scale dtype = {sdt}  weights ~= {(gu_packed.numel()+d_packed.numel())/1e9:.2f} GB")

    # ---- gate_up sweep ----
    print("\n=== gate_up (R6000-locked: block_inter=8 block_hidden=256 num_warps=4) ===")
    best = None
    locked_t = None
    for bi, bh, nw in itertools.product((8, 16, 32, 64), (64, 128, 256, 512), (2, 4, 8)):
        try:
            t = _bench(lambda bi=bi, bh=bh, nw=nw: nvfp4_grouped_gate_up_silu(
                x, eids, gu_packed, gu_scale, gu_g,
                block_inter=bi, block_hidden=bh, num_warps=nw))
        except Exception:
            continue
        if (bi, bh, nw) == (8, 256, 4):
            locked_t = t
        if best is None or t < best[0]:
            best = (t, bi, bh, nw)
    if locked_t and best:
        print(f"  locked    : {locked_t:7.1f} us")
        print(f"  BEST      : {best[0]:7.1f} us  (bi={best[1]} bh={best[2]} nw={best[3]})  "
              f"{locked_t/best[0]:.2f}x vs locked")

    # ---- down sweep ----
    inter = torch.randn(K, INTER, device=dev).bfloat16()
    print("\n=== down (R6000-locked: block_hidden=8 block_inter=512 num_warps=8) ===")
    best = None
    locked_t = None
    for bh, bi, nw in itertools.product((4, 8, 16, 32), (128, 256, 512), (4, 8)):
        try:
            t = _bench(lambda bh=bh, bi=bi, nw=nw: nvfp4_grouped_down_weighted_sum(
                inter, eids, rw, d_packed, d_scale, d_g,
                block_hidden=bh, block_inter=bi, num_warps=nw))
        except Exception:
            continue
        if (bh, bi, nw) == (8, 512, 8):
            locked_t = t
        if best is None or t < best[0]:
            best = (t, bh, bi, nw)
    if locked_t and best:
        print(f"  locked    : {locked_t:7.1f} us")
        print(f"  BEST      : {best[0]:7.1f} us  (bh={best[1]} bi={best[2]} nw={best[3]})  "
              f"{locked_t/best[0]:.2f}x vs locked")


if __name__ == "__main__":
    main()
