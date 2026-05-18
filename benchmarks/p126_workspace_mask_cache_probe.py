#!/usr/bin/env python3
"""P126: decode-workspace + mask-cache scaffold probe.

Companion to P123 (full-attn strict cache probe) and P124 (linear-core
boundary probe). P126 covers the two remaining Stream B owned-module
scaffolds:

* ``triton_kernels/full_attn_decode_workspace.py`` (scratch pos/token tensors)
* ``triton_kernels/full_attn_mask_cache.py`` (causal mask buffer)

Both are scaffold-only on this commit — not wired into the safe-default
decode path. The probe exists so any future "wire-in" commit has a
measurable baseline:

* prewarm cost (first-call vs steady-state);
* per-call cost of ``set_position`` / ``set_token`` vs the ad-hoc
  ``torch.tensor([[…]])`` allocation that the fallback path uses today;
* per-call cost of ``mask_cache.lookup(seq_len)`` view slice vs the
  ad-hoc ``torch.tril(torch.ones(...))`` baseline;
* strict numerical parity:
  - workspace ``set_position(p)`` output equals
    ``torch.tensor([[p]], device, long)`` byte-for-byte;
  - mask cache ``lookup(seq_len)`` content equals
    ``torch.tril(torch.ones(seq_len, seq_len, bool))``.

Stream B promotion bar reminder (2026-05-18 hand-off):

* DEFAULT promote requires P37 3/3 exact + structured 40/40 + P25 512 ≥ 108 TPS.
* AMBER allows P37 drift but requires structured 70/70 + P25 512 ≥ 118 TPS.
* Sprint target 118 TPS.
* Latency in this probe alone is **not** a promotion signal — never report
  "microbench faster" without the three gate fields together.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from triton_kernels.full_attn_decode_workspace import (  # noqa: E402
    FullAttnDecodeWorkspace,
    get_global_workspace,
    reset_global_workspace,
)
from triton_kernels.full_attn_mask_cache import (  # noqa: E402
    FullAttnMaskCache,
    get_global_mask_cache,
)


def _bench(fn: Callable[[], Any], warmup: int, iters: int) -> float:
    """Mean per-call latency in ms (CUDA event timed when on cuda)."""
    is_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / iters)
    else:
        import time

        t0 = time.time()
        for _ in range(iters):
            fn()
        return float((time.time() - t0) * 1000.0 / iters)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.dtype == torch.bool:
        return float((a != b).long().sum())
    return float((a.to(torch.float64) - b.to(torch.float64)).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seq", type=int, default=4096, help="mask cache table max_seq")
    ap.add_argument("--mask-seq-len", type=int, default=512, help="lookup window for mask")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[p126] cuda requested but unavailable; falling back to cpu", file=sys.stderr)
        device = "cpu"

    # ─── workspace ──────────────────────────────────────────────────────
    reset_global_workspace()
    workspace = FullAttnDecodeWorkspace(device=device, dtype=torch.long).prewarm()

    # parity: workspace.set_position(p) value vs torch.tensor([[p]])
    p_val = 17
    ws_pos = workspace.set_position(p_val).clone()
    ref_pos = torch.tensor([[p_val]], device=device, dtype=torch.long)
    pos_max_abs = _max_abs(ws_pos, ref_pos)

    t_val = 248064 - 1  # near end of Qwen3.6 vocab
    ws_tok = workspace.set_token(t_val).clone()
    ref_tok = torch.tensor([[t_val]], device=device, dtype=torch.long)
    tok_max_abs = _max_abs(ws_tok, ref_tok)

    # Bench: workspace set vs ad-hoc alloc
    pos_counter = [0]

    def ws_set_position():
        pos_counter[0] += 1
        return workspace.set_position(pos_counter[0])

    def adhoc_pos_alloc():
        return torch.tensor([[pos_counter[0]]], device=device, dtype=torch.long)

    tok_counter = [0]

    def ws_set_token():
        tok_counter[0] += 1
        return workspace.set_token(tok_counter[0])

    def adhoc_tok_alloc():
        return torch.tensor([[tok_counter[0]]], device=device, dtype=torch.long)

    workspace_timing = {
        "ws_set_position": _bench(ws_set_position, args.warmup, args.iters),
        "adhoc_pos_alloc": _bench(adhoc_pos_alloc, args.warmup, args.iters),
        "ws_set_token": _bench(ws_set_token, args.warmup, args.iters),
        "adhoc_tok_alloc": _bench(adhoc_tok_alloc, args.warmup, args.iters),
    }
    ws_info = workspace.info()

    # singleton getter sanity
    global_ws = get_global_workspace(device=device)
    singleton_ok = global_ws is not workspace and isinstance(
        global_ws, FullAttnDecodeWorkspace
    )

    # ─── mask cache ─────────────────────────────────────────────────────
    mask_cache = FullAttnMaskCache()
    # prewarm cost
    t_prewarm = _bench(
        lambda: mask_cache.prewarm(device, torch.bool, args.max_seq),
        max(5, args.warmup // 10),
        max(20, args.iters // 50),
    )

    # parity: mask_cache.lookup(seq_len) vs torch.tril
    ref_mask = torch.tril(
        torch.ones(args.mask_seq_len, args.mask_seq_len, device=device, dtype=torch.bool)
    )
    cache_mask = mask_cache.lookup(args.mask_seq_len, device, torch.bool, args.max_seq)
    mask_parity_max_abs = _max_abs(cache_mask, ref_mask)
    mask_parity_shape = (
        list(cache_mask.shape) == list(ref_mask.shape)
        and bool((cache_mask == ref_mask).all())
    )

    def cache_lookup():
        return mask_cache.lookup(args.mask_seq_len, device, torch.bool, args.max_seq)

    def adhoc_tril():
        return torch.tril(
            torch.ones(args.mask_seq_len, args.mask_seq_len, device=device, dtype=torch.bool)
        )

    mask_timing = {
        "mask_cache_prewarm": t_prewarm,
        "mask_cache_lookup_view": _bench(cache_lookup, args.warmup, args.iters),
        "adhoc_tril_alloc": _bench(adhoc_tril, max(5, args.warmup // 5), max(20, args.iters // 20)),
    }

    mask_info_list = [info.__dict__ for info in mask_cache.info()]

    # ─── env snapshot ───────────────────────────────────────────────────
    env_snapshot = {
        name: os.environ.get(name)
        for name in (
            "LYNN_FULL_ATTN_MASK_CACHE",
            "LYNN_FULL_ATTN_MASK_CACHE_MAX_SEQ",
            "LYNN_FULL_ATTN_ROPE_CACHE",
            "LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ",
        )
    }

    result = {
        "schema_version": "lynn-engine-p126-workspace-mask-cache-probe-v1",
        "device": (
            torch.cuda.get_device_name(device) if device.startswith("cuda") else device
        ),
        "env": env_snapshot,
        "workspace_timing_ms": workspace_timing,
        "workspace_parity": {
            "pos_max_abs": pos_max_abs,
            "tok_max_abs": tok_max_abs,
            "pos_exact": pos_max_abs == 0.0,
            "tok_exact": tok_max_abs == 0.0,
        },
        "workspace_info": ws_info.__dict__,
        "workspace_singleton_ok": singleton_ok,
        "mask_cache_timing_ms": mask_timing,
        "mask_cache_parity": {
            "max_abs_mismatches": mask_parity_max_abs,
            "shape_and_content_exact": mask_parity_shape,
        },
        "mask_cache_info": mask_info_list,
        "promotion_bar_reminder": {
            "DEFAULT": "P37 3/3 + structured 40/40 + P25 512 >= 108 TPS",
            "AMBER": "P37 drift ok + structured 70/70 + P25 512 >= 118 TPS",
            "SPRINT": "118 TPS",
            "report_rule": "every status must carry P37 exact + structured pass + P25 512 decode together",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
