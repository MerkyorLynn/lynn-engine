#!/usr/bin/env python3
"""Spark MoE spec-verify probe: per-position vs grouped small-M (no model load).

The MTP smoke's spec_k2 slowdown is the per-position T=1 MoE verify. FlashRT's
small-M 16x is a DENSE win (all rows share one weight). MoE routed experts have
DIVERSE routing (top-8 of 256) -> at M=K_draft+1 the (pos,expert) pairs are
mostly UNIQUE -> little weight-read amortization. So the real MoE-verify win is
bandwidth/launch efficiency: collapse the M*K tiny latency-bound expert GEMVs
(current per-position path) into ONE grouped op + amortize the dense shared
expert across M rows.

This probe measures that win at the real 35B-A3B shapes (E=256, K=8,
hidden=2048, gate_up=1024, inter=512), realistic random routing, BF16 (W4A16),
on Spark, with no checkpoint. It tells us whether grouping flips MTP to a win
and by how much, BEFORE the token-exact production wiring.

Run on Spark:  python scripts/spark_moe_verify_grouped_probe.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

E, K = 256, 8           # experts, top-k
HID, GU, INTER = 2048, 1024, 512  # hidden, gate_up out (=2*inter), inter


def _bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def per_position(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn):
    # h [M,HID]; eids [M,K]; current path: M independent full-MoE calls.
    M = h.shape[0]
    out = torch.zeros_like(h)
    for m in range(M):
        acc = torch.zeros(HID, device=h.device, dtype=h.dtype)
        hm = h[m]
        for j in range(K):
            e = int(eids[m, j])
            t = hm @ gu_w[e].t()              # [GU]
            g, u = t[:INTER], t[INTER:]
            a = F.silu(g) * u                  # [INTER]
            acc += rw[m, j] * (a @ dn_w[e].t())  # [HID]
        # dense shared expert (per position)
        ts = hm @ sh_gu.t(); gs, us = ts[:INTER], ts[INTER:]
        acc += (F.silu(gs) * us) @ sh_dn.t()
        out[m] = acc
    return out


def grouped(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn):
    # gather the M*K (pos,expert) rows, two batched bmms, scatter-add. Plus the
    # shared expert as ONE dense small-M GEMM over all M rows.
    M = h.shape[0]
    rows = torch.arange(M, device=h.device).repeat_interleave(K)  # [M*K]
    ef = eids.reshape(-1)                                          # [M*K]
    wf = rw.reshape(-1)                                            # [M*K]
    hr = h[rows]                                                   # [M*K, HID]
    GUg = gu_w[ef]                                                 # [M*K, GU, HID]
    t = torch.bmm(hr.unsqueeze(1), GUg.transpose(1, 2)).squeeze(1)  # [M*K, GU]
    a = F.silu(t[:, :INTER]) * t[:, INTER:]                        # [M*K, INTER]
    DNg = dn_w[ef]                                                 # [M*K, HID, INTER]
    o = torch.bmm(a.unsqueeze(1), DNg.transpose(1, 2)).squeeze(1)  # [M*K, HID]
    o = o * wf.unsqueeze(1)
    out = torch.zeros_like(h)
    out.index_add_(0, rows, o)
    # shared expert: dense, all M rows in one small-M GEMM (the FlashRT win)
    ts = h @ sh_gu.t()
    out = out + (F.silu(ts[:, :INTER]) * ts[:, INTER:]) @ sh_dn.t()
    return out


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    cap = torch.cuda.get_device_capability()
    torch.manual_seed(0)
    print(f"device sm_{cap[0]}{cap[1]}  E={E} K={K} HID={HID} GU={GU} INTER={INTER}")
    gu_w = (torch.randn(E, GU, HID, device=dev) * 0.02).bfloat16()
    dn_w = (torch.randn(E, HID, INTER, device=dev) * 0.02).bfloat16()
    sh_gu = (torch.randn(GU, HID, device=dev) * 0.02).bfloat16()
    sh_dn = (torch.randn(HID, INTER, device=dev) * 0.02).bfloat16()
    print(f"weights resident ~= {(gu_w.numel()+dn_w.numel())*2/1e9:.2f} GB")

    for M in (3, 5, 9):  # k2 -> M=3, k4 -> M=5, k8 -> M=9
        h = (torch.randn(M, HID, device=dev) * 0.5).bfloat16()
        eids = torch.randint(0, E, (M, K), device=dev)
        rw = torch.softmax(torch.randn(M, K, device=dev), dim=-1).bfloat16()
        uniq = torch.unique(eids).numel()
        op = per_position(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn)
        og = grouped(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn)
        cos = F.cosine_similarity(op.float().flatten(), og.float().flatten(), dim=0).item()
        tp = _bench(lambda: per_position(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn))
        tg = _bench(lambda: grouped(h, eids, rw, gu_w, dn_w, sh_gu, sh_dn))
        print(f"M={M} (uniq experts {uniq}/{M*K}): "
              f"per-position {tp*1000:7.1f}us  grouped {tg*1000:7.1f}us  "
              f"{tp/tg:5.2f}x  cos={cos:.5f}")


if __name__ == "__main__":
    main()
