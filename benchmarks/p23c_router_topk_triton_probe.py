#!/usr/bin/env python3
"""P23-C: Triton fused top-k+softmax router probe."""
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

from benchmarks.p23_active_moe_layer_sweep import _collect_layer_inputs  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.router_topk import router_topk_softmax_triton  # noqa: E402


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


def _sets_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return sorted(int(x) for x in a.tolist()) == sorted(int(x) for x in b.tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--layers", default="all")
    args = ap.parse_args()

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    layer_inputs = _collect_layer_inputs(runner, args.prompt)
    if args.layers == "all":
        layer_ids = list(range(runner.n_layers))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]

    rows = []
    for layer_idx in layer_ids:
        w = runner.layer_weights[layer_idx]
        cfg = runner.layer_cfgs[layer_idx]
        h_moe = _rms_norm(layer_inputs[layer_idx], w["post_attention_layernorm.weight"])
        h_flat = h_moe.reshape(-1, h_moe.shape[-1])
        top_k = int(cfg["num_experts_per_tok"])
        logits = F.linear(h_flat, w["mlp.gate.weight"])
        ref_values, ref_indices = torch.topk(logits, top_k, dim=-1, sorted=False)
        ref_weights = F.softmax(ref_values, dim=-1, dtype=torch.float32)[0]
        tri_weights, tri_indices = router_topk_softmax_triton(logits[0].contiguous(), top_k)

        def torch_router() -> torch.Tensor:
            local_logits = F.linear(h_flat, w["mlp.gate.weight"])
            local_values, local_indices = torch.topk(local_logits, top_k, dim=-1, sorted=False)
            return F.softmax(local_values, dim=-1, dtype=torch.float32) + local_indices.float() * 0.0

        def triton_router_cached_logits() -> torch.Tensor:
            local_weights, local_indices = router_topk_softmax_triton(logits[0].contiguous(), top_k)
            return local_weights + local_indices.float() * 0.0

        def triton_router_full() -> torch.Tensor:
            local_logits = F.linear(h_flat, w["mlp.gate.weight"])
            local_weights, local_indices = router_topk_softmax_triton(local_logits[0].contiguous(), top_k)
            return local_weights + local_indices.float() * 0.0

        # Compare unordered weights by mapping index -> probability.
        ref_map = {int(i): float(v) for i, v in zip(ref_indices[0].tolist(), ref_weights.tolist())}
        tri_map = {int(i): float(v) for i, v in zip(tri_indices.tolist(), tri_weights.tolist())}
        common = sorted(set(ref_map) & set(tri_map))
        weight_max_abs = max((abs(ref_map[i] - tri_map[i]) for i in common), default=float("inf"))
        rows.append({
            "layer": layer_idx,
            "expert_count": int(w["mlp.gate.weight"].shape[0]),
            "topk_same_set": _sets_equal(ref_indices[0], tri_indices),
            "ref_indices": [int(x) for x in ref_indices[0].tolist()],
            "triton_indices": [int(x) for x in tri_indices.tolist()],
            "weight_common_count": len(common),
            "weight_max_abs_on_common": float(weight_max_abs),
            "torch_router_ms": _bench(torch_router, args.warmup, args.iters),
            "triton_cached_logits_ms": _bench(triton_router_cached_logits, args.warmup, args.iters),
            "triton_full_router_ms": _bench(triton_router_full, args.warmup, args.iters),
        })

    result = {
        "schema_version": "lynn-engine-p23c-router-topk-triton-probe-v1",
        "model": args.model,
        "layers": rows,
        "all_same_set": all(r["topk_same_set"] for r in rows),
        "max_weight_abs_on_common": max(r["weight_max_abs_on_common"] for r in rows),
        "mean_torch_router_ms": sum(r["torch_router_ms"] for r in rows) / len(rows),
        "mean_triton_cached_logits_ms": sum(r["triton_cached_logits_ms"] for r in rows) / len(rows),
        "mean_triton_full_router_ms": sum(r["triton_full_router_ms"] for r in rows) / len(rows),
        "top_mismatches": [r for r in rows if not r["topk_same_set"]][:10],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
