#!/usr/bin/env python3
"""Stage 6 Phase 2-D: router/shared-inclusive one-layer MoE hybrid PoC.

P2-C measured the active routed expert path with precomputed routes and without
the shared expert. This harness puts the expensive reality back in:

* router linear + top-k + softmax + eager grouping are timed inside the hybrid;
* active routed experts read packed NVFP4 directly, after BF16 active shadows
  are deleted;
* shared expert stays on the existing BF16 prefill path for this gate.

It is still a one-layer PoC, not a serving integration path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _moe_forward  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)
from scripts.spark_stage6_p2c_active_moe_lower_bound_poc import (  # noqa: E402
    _bf16_active_moe,
    _load_grouped_pair,
    _packed_active_moe,
    _route,
)


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _tensor_bytes(items: list[torch.Tensor | None]) -> int:
    return sum(_nbytes(t) for t in items if isinstance(t, torch.Tensor))


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def _peak_once(fn: Callable[[], torch.Tensor]) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = _cuda_mem_gib()
    y = fn()
    torch.cuda.synchronize()
    after = _cuda_mem_gib()
    peak = float(torch.cuda.max_memory_allocated() / GIB)
    del y
    torch.cuda.empty_cache()
    return {"before_gib": before, "after_gib": after, "peak_gib": peak}


def _shared_bf16(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return torch.zeros_like(h_flat)
    gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
    up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
        shared = shared * shared_gate
    return shared


def _hybrid_one_layer_moe(
    h: torch.Tensor,
    w: dict[str, Any],
    packed: dict[str, torch.Tensor],
    *,
    top_k: int,
    block_t: int,
    block_inter: int,
    block_hidden: int,
    down_block_hidden: int,
    down_block_inter: int,
) -> torch.Tensor:
    h_flat = h.reshape(-1, h.shape[-1])
    expert_indices, routing_weights, groups = _route(h_flat, w["mlp.gate.weight"], top_k)
    active = _packed_active_moe(
        h_flat,
        expert_indices,
        routing_weights,
        groups,
        packed,
        top_k=top_k,
        block_t=block_t,
        block_inter=block_inter,
        block_hidden=block_hidden,
        down_block_hidden=down_block_hidden,
        down_block_inter=down_block_inter,
    )
    shared = _shared_bf16(h_flat, w)
    return (active + shared).to(h.dtype).reshape_as(h)


def _route_only(h: torch.Tensor, w: dict[str, Any], *, top_k: int) -> tuple[torch.Tensor, torch.Tensor, Any]:
    h_flat = h.reshape(-1, h.shape[-1])
    return _route(h_flat, w["mlp.gate.weight"], top_k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--block-t", type=int, default=32)
    ap.add_argument("--block-inter", type=int, default=16)
    ap.add_argument("--block-hidden", type=int, default=128)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    batches = _parse_batches(args.batches)
    torch.manual_seed(args.seed)
    os.environ.setdefault("LYNN_ROUTER_TOPK_SORTED", "0")

    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))
    w, layer_cfg = load_qwen36_layer(
        str(model_dir),
        args.layer,
        num_experts=num_experts,
        device=device,
        dequant_dtype=torch.bfloat16,
    )
    layer_cfg["num_experts"] = int(layer_cfg.get("num_experts", num_experts))
    layer_cfg["num_experts_per_tok"] = top_k
    layer_cfg["is_moe"] = True
    packed = _load_grouped_pair(model_dir, args.layer, device=device)

    hidden = int(w["mlp.gate.weight"].shape[1])
    active_bf16_bytes = _tensor_bytes([
        w.get("mlp.experts.gate_up_proj"),
        w.get("mlp.experts.down_proj"),
    ])
    shared_bf16_bytes = _tensor_bytes([
        w.get("mlp.shared_expert.gate_proj.weight"),
        w.get("mlp.shared_expert.up_proj.weight"),
        w.get("mlp.shared_expert.down_proj.weight"),
        w.get("mlp.shared_expert_gate.weight"),
    ])
    packed_active_bytes = sum(_nbytes(t) for t in packed.values())
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((1, b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    torch.cuda.synchronize()

    print("=============== STAGE 6 PHASE 2-D ONE-LAYER MOE HYBRID POC ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layer        : {args.layer}", flush=True)
    print(f"batches      : {batches}", flush=True)
    print(f"shape        : hidden={hidden} experts={num_experts} top_k={top_k}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"BF16 shared  : {shared_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_active_bytes / GIB:.3f} GiB", flush=True)

    refs: dict[int, torch.Tensor] = {}
    bench_bf16_full: dict[str, Any] = {}
    for b, h in xs.items():
        refs[b] = _moe_forward(h, w, layer_cfg).detach()
        bench_bf16_full[str(b)] = _bench_cuda(
            lambda h=h: _moe_forward(h, w, layer_cfg),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    # Component references are collected before deleting BF16 active shadows so
    # the report can explain the hybrid cost. The timed hybrid below cannot read
    # these active shadows because they are deleted before it runs.
    routes: dict[int, dict[str, Any]] = {}
    active_numeric: dict[str, Any] = {}
    for b, h in xs.items():
        h_flat = h.reshape(-1, h.shape[-1])
        expert_indices, routing_weights, groups = _route(h_flat, w["mlp.gate.weight"], top_k)
        routes[b] = {
            "expert_indices": expert_indices,
            "routing_weights": routing_weights,
            "groups": groups,
            "unique_experts": len(groups),
            "top_group_counts": [
                {"expert": g.expert, "rows": int(g.token_idx.numel())}
                for g in groups[:12]
            ],
        }
        active_ref = _bf16_active_moe(
            h_flat,
            groups,
            routing_weights,
            w["mlp.experts.gate_up_proj"],
            w["mlp.experts.down_proj"],
        ).detach()
        active_y = _packed_active_moe(
            h_flat,
            expert_indices,
            routing_weights,
            groups,
            packed,
            top_k=top_k,
            block_t=args.block_t,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
        ).detach()
        active_numeric[str(b)] = _diff_stats(active_y, active_ref)
        del active_ref, active_y

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    numeric: dict[str, Any] = {}
    bench_hybrid: dict[str, Any] = {}
    bench_route: dict[str, Any] = {}
    bench_shared: dict[str, Any] = {}
    bench_active_precomputed: dict[str, Any] = {}
    peak_hybrid: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, h in xs.items():
        h_flat = h.reshape(-1, h.shape[-1])
        route = routes[b]
        hybrid_fn = lambda h=h: _hybrid_one_layer_moe(
            h,
            w,
            packed,
            top_k=top_k,
            block_t=args.block_t,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
        )
        y = hybrid_fn().detach()
        numeric[str(b)] = _diff_stats(y, refs[b])
        del y
        peak_hybrid[str(b)] = _peak_once(hybrid_fn)
        bench_hybrid[str(b)] = _bench_cuda(
            hybrid_fn,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bench_route[str(b)] = _bench_cuda(
            lambda h=h: _route_only(h, w, top_k=top_k),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bench_shared[str(b)] = _bench_cuda(
            lambda h_flat=h_flat: _shared_bf16(h_flat, w),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bench_active_precomputed[str(b)] = _bench_cuda(
            lambda h_flat=h_flat, route=route: _packed_active_moe(
                h_flat,
                route["expert_indices"],
                route["routing_weights"],
                route["groups"],
                packed,
                top_k=top_k,
                block_t=args.block_t,
                block_inter=args.block_inter,
                block_hidden=args.block_hidden,
                down_block_hidden=args.down_block_hidden,
                down_block_inter=args.down_block_inter,
            ),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bf = bench_bf16_full[str(b)]["median_us"]
        hy = bench_hybrid[str(b)]["median_us"]
        rt = bench_route[str(b)]["median_us"]
        sh = bench_shared[str(b)]["median_us"]
        ac = bench_active_precomputed[str(b)]["median_us"]
        rows.append({
            "batch": b,
            "unique_experts": route["unique_experts"],
            "bf16_full_median_us": bf,
            "hybrid_median_us": hy,
            "route_median_us": rt,
            "shared_bf16_median_us": sh,
            "active_precomputed_packed_median_us": ac,
            "hybrid_vs_bf16": bf / hy if hy else math.nan,
            "hybrid_us_per_token": hy / b,
            "bf16_us_per_token": bf / b,
        })
        print(
            f"[M={b} unique={route['unique_experts']}] "
            f"hybrid={hy:.2f}us bf16={bf:.2f}us speedup={bf / hy if hy else math.nan:.3f}x "
            f"route={rt:.2f}us active_pre={ac:.2f}us shared={sh:.2f}us "
            f"cos={numeric[str(b)]['cosine']:.9f}",
            flush=True,
        )

    numeric_pass = all(
        numeric[str(b)]["cosine"] > 0.999 and numeric[str(b)]["argmax_match"]
        for b in batches
    )
    active_numeric_pass = all(
        active_numeric[str(b)]["cosine"] > 0.999 and active_numeric[str(b)]["argmax_match"]
        for b in batches
    )
    no_active_shadow_pass = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
    result = {
        "schema": "lynn-stage6-p2d-one-layer-moe-hybrid-poc-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "tile": {
            "gateup_block_t": args.block_t,
            "gateup_block_inter": args.block_inter,
            "gateup_block_hidden": args.block_hidden,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
        },
        "shape": {
            "hidden": hidden,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "bytes": {
            "bf16_layer_active_experts": active_bf16_bytes,
            "bf16_layer_shared_expert": shared_bf16_bytes,
            "packed_layer_active_experts": packed_active_bytes,
            "active_bf16_to_packed_ratio": active_bf16_bytes / packed_active_bytes if packed_active_bytes else None,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "routes": {
            str(b): {
                "unique_experts": routes[b]["unique_experts"],
                "top_group_counts": routes[b]["top_group_counts"],
            }
            for b in batches
        },
        "numeric": {
            "hybrid_vs_full_bf16": numeric,
            "active_packed_vs_active_bf16": active_numeric,
        },
        "bench": {
            "rows": rows,
            "bf16_full_moe": bench_bf16_full,
            "hybrid_router_packed_active_bf16_shared": bench_hybrid,
            "component_router": bench_route,
            "component_packed_active_precomputed": bench_active_precomputed,
            "component_shared_bf16": bench_shared,
        },
        "memory": {
            "hybrid_peak": peak_hybrid,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "active_numeric": bool(active_numeric_pass),
            "no_bf16_active_shadow_for_hybrid_bench": bool(no_active_shadow_pass),
            "perf_speedup_vs_bf16_all_batches": all(r["hybrid_vs_bf16"] >= 1.0 for r in rows),
            "all": bool(
                numeric_pass
                and active_numeric_pass
                and no_active_shadow_pass
                and all(r["hybrid_vs_bf16"] >= 1.0 for r in rows)
            ),
        },
        "notes": [
            "Hybrid includes router linear/topk/softmax/eager grouping inside the timed path.",
            "Hybrid active experts read packed NVFP4 directly after deleting BF16 active shadows.",
            "Shared expert remains BF16 for this gate; packed shared prefill is a separate follow-up if needed.",
            "This is a one-layer prefill PoC, not an end-to-end serving integration path.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
