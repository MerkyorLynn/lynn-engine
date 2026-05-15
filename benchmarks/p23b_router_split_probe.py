#!/usr/bin/env python3
"""P23-B: split router latency into linear / top-k / softmax pieces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _manual_softmax(x: torch.Tensor) -> torch.Tensor:
    # Small-vector explicit softmax for comparison only. This is not expected to
    # beat fused torch softmax because it launches multiple pointwise ops.
    x32 = x.float()
    y = torch.exp(x32 - x32.max(dim=-1, keepdim=True).values)
    return y / y.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    top_k = int(cfg["num_experts_per_tok"])

    logits = F.linear(h_flat, w["mlp.gate.weight"])
    values, indices = torch.topk(logits, top_k, dim=-1, sorted=False)
    logits_1d = logits[0].contiguous()
    values_1d, indices_1d = torch.topk(logits_1d, top_k, dim=0, sorted=False)
    sm = F.softmax(values, dim=-1, dtype=torch.float32)
    sm_1d = F.softmax(values_1d, dim=0, dtype=torch.float32)
    manual_sm = _manual_softmax(values)

    def linear_only() -> torch.Tensor:
        return F.linear(h_flat, w["mlp.gate.weight"])

    def topk_only() -> torch.Tensor:
        v, i = torch.topk(logits, top_k, dim=-1, sorted=False)
        return v + i.float() * 0.0

    def topk_1d_only() -> torch.Tensor:
        v, i = torch.topk(logits_1d, top_k, dim=0, sorted=False)
        return v + i.float() * 0.0

    def softmax_only() -> torch.Tensor:
        return F.softmax(values, dim=-1, dtype=torch.float32)

    def softmax_1d_only() -> torch.Tensor:
        return F.softmax(values_1d, dim=0, dtype=torch.float32)

    def manual_softmax_only() -> torch.Tensor:
        return _manual_softmax(values)

    def full_router() -> torch.Tensor:
        local_logits = F.linear(h_flat, w["mlp.gate.weight"])
        local_values, local_indices = torch.topk(local_logits, top_k, dim=-1, sorted=False)
        return F.softmax(local_values, dim=-1, dtype=torch.float32) + local_indices.float() * 0.0

    def full_router_1d() -> torch.Tensor:
        local_logits = F.linear(h_flat, w["mlp.gate.weight"])[0].contiguous()
        local_values, local_indices = torch.topk(local_logits, top_k, dim=0, sorted=False)
        return F.softmax(local_values, dim=0, dtype=torch.float32) + local_indices.float() * 0.0

    result = {
        "schema_version": "lynn-engine-p23b-router-split-probe-v1",
        "model": args.model,
        "layer": args.layer,
        "top_k": top_k,
        "expert_count": int(w["mlp.gate.weight"].shape[0]),
        "expert_indices": [int(x) for x in indices[0].tolist()],
        "expert_indices_1d": [int(x) for x in indices_1d.tolist()],
        "topk_same_set": sorted(int(x) for x in indices[0].tolist()) == sorted(int(x) for x in indices_1d.tolist()),
        "softmax_diff_manual_max_abs": float((sm - manual_sm).abs().max().item()),
        "softmax_diff_2d_1d_max_abs": float((sm[0].sort().values - sm_1d.sort().values).abs().max().item()),
        "timing_ms": {
            "linear_only_ms": _bench(linear_only, args.warmup, args.iters),
            "topk_only_ms": _bench(topk_only, args.warmup, args.iters),
            "topk_1d_only_ms": _bench(topk_1d_only, args.warmup, args.iters),
            "softmax_only_ms": _bench(softmax_only, args.warmup, args.iters),
            "softmax_1d_only_ms": _bench(softmax_1d_only, args.warmup, args.iters),
            "manual_softmax_only_ms": _bench(manual_softmax_only, args.warmup, args.iters),
            "full_router_ms": _bench(full_router, args.warmup, args.iters),
            "full_router_1d_ms": _bench(full_router_1d, args.warmup, args.iters),
        },
        "notes": [
            "topk_only reuses cached logits; full_router is the production-shaped path.",
            "manual softmax is only a probe and is expected to launch more kernels.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
