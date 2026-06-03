#!/usr/bin/env python3
"""Stage 6 Phase 2-E: grouped scheduler / packed-active retune probe.

P2-D exposed two measured blockers:

* route/top-k/grouping is several milliseconds at M=64;
* packed active MoE is memory-clean but still slower than resident BF16.

This probe tests low-risk runtime changes before touching production code:

* current `unique + mask.nonzero` grouping vs `sort + unique_consecutive`;
* baseline packed active allocation behavior vs scratch-buffer reuse;
* a small tile sweep around the packed gate/up prefill kernel.
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
    ExpertGroup,
    _load_grouped_pair,
    _packed_active_moe,
    _route,
)
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_prefill_gate_up_silu_one_expert,
)


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


def _parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_tile_sweep(text: str) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        cfg = {"block_t": 32, "block_inter": 16, "block_hidden": 128, "num_warps": 4}
        for item in raw.split(","):
            key, value = item.split("=")
            cfg[key.strip()] = int(value)
        configs.append(cfg)
    return configs


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


def _route_sort_groups(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[ExpertGroup]]:
    logits = F.linear(x, gate_weight)
    routing_logits, expert_indices = torch.topk(logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_logits, dim=-1, dtype=torch.float32)

    flat_experts = expert_indices.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[order]
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    token_base = torch.arange(x.shape[0], device=x.device, dtype=torch.long).repeat_interleave(top_k)
    slot_base = torch.arange(top_k, device=x.device, dtype=torch.long).repeat(x.shape[0])
    sorted_tokens = token_base[order]
    sorted_slots = slot_base[order]

    groups: list[ExpertGroup] = []
    offset = 0
    for expert, count in zip(unique.tolist(), counts.tolist(), strict=True):
        end = offset + int(count)
        groups.append(
            ExpertGroup(
                int(expert),
                sorted_tokens[offset:end].contiguous(),
                sorted_slots[offset:end].contiguous(),
            )
        )
        offset = end
    return expert_indices.to(torch.int32).contiguous(), routing_weights.contiguous(), groups


def _shared_bf16(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return torch.zeros_like(h_flat)
    gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
    up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _packed_active_scratch(
    x: torch.Tensor,
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    groups: list[ExpertGroup],
    packed: dict[str, torch.Tensor],
    *,
    top_k: int,
    block_t: int,
    block_inter: int,
    block_hidden: int,
    num_warps: int,
    down_block_hidden: int,
    down_block_inter: int,
    down_num_warps: int,
) -> torch.Tensor:
    inter = torch.empty((x.shape[0], top_k, 512), device=x.device, dtype=torch.bfloat16)
    max_rows = max((int(g.token_idx.numel()) for g in groups), default=1)
    gate_scratch = torch.empty((max_rows, 512), device=x.device, dtype=torch.bfloat16)
    for group in groups:
        rows = int(group.token_idx.numel())
        y = nvfp4_prefill_gate_up_silu_one_expert(
            x[group.token_idx],
            group.expert,
            packed["gate_up_packed"],
            packed["gate_up_scale"],
            packed["gate_up_global_scale"],
            block_t=block_t,
            block_inter=block_inter,
            block_hidden=block_hidden,
            num_warps=num_warps,
            out=gate_scratch[:rows],
        )
        inter[group.token_idx, group.slot_idx] = y
    out = torch.empty_like(x)
    for token in range(x.shape[0]):
        nvfp4_grouped_down_weighted_sum(
            inter[token],
            expert_indices[token],
            routing_weights[token],
            packed["down_packed"],
            packed["down_scale"],
            packed["down_global_scale"],
            block_hidden=down_block_hidden,
            block_inter=down_block_inter,
            num_warps=down_num_warps,
            out=out[token],
        )
    return out


def _hybrid_scratch(
    h: torch.Tensor,
    w: dict[str, Any],
    packed: dict[str, torch.Tensor],
    *,
    route_mode: str,
    top_k: int,
    tile: dict[str, int],
    down_block_hidden: int,
    down_block_inter: int,
    down_num_warps: int,
) -> torch.Tensor:
    h_flat = h.reshape(-1, h.shape[-1])
    if route_mode == "current":
        expert_indices, routing_weights, groups = _route(h_flat, w["mlp.gate.weight"], top_k)
    elif route_mode == "sort":
        expert_indices, routing_weights, groups = _route_sort_groups(h_flat, w["mlp.gate.weight"], top_k)
    else:
        raise ValueError(f"unknown route mode {route_mode!r}")
    active = _packed_active_scratch(
        h_flat,
        expert_indices,
        routing_weights,
        groups,
        packed,
        top_k=top_k,
        block_t=tile["block_t"],
        block_inter=tile["block_inter"],
        block_hidden=tile["block_hidden"],
        num_warps=tile["num_warps"],
        down_block_hidden=down_block_hidden,
        down_block_inter=down_block_inter,
        down_num_warps=down_num_warps,
    )
    return (active + _shared_bf16(h_flat, w)).to(h.dtype).reshape_as(h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument(
        "--tile-sweep",
        default=(
            "block_t=32,block_inter=16,block_hidden=128,num_warps=4;"
            "block_t=16,block_inter=16,block_hidden=128,num_warps=4;"
            "block_t=64,block_inter=16,block_hidden=128,num_warps=4;"
            "block_t=32,block_inter=8,block_hidden=128,num_warps=4;"
            "block_t=32,block_inter=32,block_hidden=128,num_warps=4;"
            "block_t=32,block_inter=16,block_hidden=256,num_warps=4;"
            "block_t=32,block_inter=16,block_hidden=128,num_warps=8"
        ),
    )
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--down-num-warps", type=int, default=8)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = "cuda"
    model_dir = Path(args.model)
    batches = _parse_batches(args.batches)
    tiles = _parse_tile_sweep(args.tile_sweep)
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
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((1, b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    refs = {b: _moe_forward(h, w, layer_cfg).detach() for b, h in xs.items()}
    bf16_full = {
        str(b): _bench_cuda(lambda h=h: _moe_forward(h, w, layer_cfg), warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        for b, h in xs.items()
    }
    active_bf16_bytes = _tensor_bytes([w.get("mlp.experts.gate_up_proj"), w.get("mlp.experts.down_proj")])
    packed_active_bytes = sum(_nbytes(t) for t in packed.values())

    routes: dict[str, dict[int, dict[str, Any]]] = {"current": {}, "sort": {}}
    route_bench: dict[str, dict[str, Any]] = {"current": {}, "sort": {}}
    for b, h in xs.items():
        h_flat = h.reshape(-1, h.shape[-1])
        for mode in ("current", "sort"):
            route_fn = _route if mode == "current" else _route_sort_groups
            expert_indices, routing_weights, groups = route_fn(h_flat, w["mlp.gate.weight"], top_k)
            routes[mode][b] = {
                "expert_indices": expert_indices,
                "routing_weights": routing_weights,
                "groups": groups,
                "unique_experts": len(groups),
                "top_group_counts": [
                    {"expert": g.expert, "rows": int(g.token_idx.numel())}
                    for g in sorted(groups, key=lambda g: int(g.token_idx.numel()), reverse=True)[:12]
                ],
            }
            route_bench[mode][str(b)] = _bench_cuda(
                lambda h_flat=h_flat, route_fn=route_fn: route_fn(h_flat, w["mlp.gate.weight"], top_k),
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    baseline_tile = tiles[0]
    active_rows: list[dict[str, Any]] = []
    active_bench: dict[str, Any] = {}
    numeric: dict[str, Any] = {}
    for b, h in xs.items():
        h_flat = h.reshape(-1, h.shape[-1])
        route = routes["current"][b]
        alloc_fn = lambda h_flat=h_flat, route=route: _packed_active_moe(
            h_flat,
            route["expert_indices"],
            route["routing_weights"],
            route["groups"],
            packed,
            top_k=top_k,
            block_t=baseline_tile["block_t"],
            block_inter=baseline_tile["block_inter"],
            block_hidden=baseline_tile["block_hidden"],
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
        )
        scratch_fn = lambda h_flat=h_flat, route=route: _packed_active_scratch(
            h_flat,
            route["expert_indices"],
            route["routing_weights"],
            route["groups"],
            packed,
            top_k=top_k,
            block_t=baseline_tile["block_t"],
            block_inter=baseline_tile["block_inter"],
            block_hidden=baseline_tile["block_hidden"],
            num_warps=baseline_tile["num_warps"],
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
            down_num_warps=args.down_num_warps,
        )
        y = scratch_fn().detach()
        # Compare against the full BF16 MoE after adding the shared path in the
        # hybrid gate below; active-only numeric is covered by P2-C. Here we
        # validate scratch equivalence against the allocation baseline.
        numeric[f"scratch_vs_alloc_M{b}"] = _diff_stats(y, alloc_fn().detach())
        del y
        active_bench[f"alloc_M{b}"] = _bench_cuda(alloc_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        active_bench[f"scratch_M{b}"] = _bench_cuda(scratch_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        alloc_us = active_bench[f"alloc_M{b}"]["median_us"]
        scratch_us = active_bench[f"scratch_M{b}"]["median_us"]
        active_rows.append({
            "batch": b,
            "mode": "scratch_vs_alloc",
            "alloc_us": alloc_us,
            "scratch_us": scratch_us,
            "scratch_speedup": alloc_us / scratch_us if scratch_us else math.nan,
        })

    tile_rows: list[dict[str, Any]] = []
    tile_bench: dict[str, Any] = {}
    max_batch = max(batches)
    h_flat_max = xs[max_batch].reshape(-1, hidden)
    route_max = routes["current"][max_batch]
    for i, tile in enumerate(tiles):
        key = f"tile_{i}"
        fn = lambda tile=tile: _packed_active_scratch(
            h_flat_max,
            route_max["expert_indices"],
            route_max["routing_weights"],
            route_max["groups"],
            packed,
            top_k=top_k,
            block_t=tile["block_t"],
            block_inter=tile["block_inter"],
            block_hidden=tile["block_hidden"],
            num_warps=tile["num_warps"],
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
            down_num_warps=args.down_num_warps,
        )
        try:
            tile_bench[key] = _bench_cuda(fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        except Exception as exc:
            tile_bench[key] = {"error": repr(exc)}
            tile_rows.append({
                "key": key,
                "batch": max_batch,
                **tile,
                "median_us": math.inf,
                "error": repr(exc),
            })
            print(
                f"[tile {key}] M={max_batch} bt={tile['block_t']} bi={tile['block_inter']} "
                f"bh={tile['block_hidden']} nw={tile['num_warps']} ERROR={exc!r}",
                flush=True,
            )
            continue
        row = {
            "key": key,
            "batch": max_batch,
            **tile,
            "median_us": tile_bench[key]["median_us"],
        }
        tile_rows.append(row)
        print(
            f"[tile {key}] M={max_batch} bt={tile['block_t']} bi={tile['block_inter']} "
            f"bh={tile['block_hidden']} nw={tile['num_warps']} median={row['median_us']:.2f}us",
            flush=True,
        )
    best_tile = min(tile_rows, key=lambda r: r["median_us"])
    best_tile_cfg = {
        "block_t": int(best_tile["block_t"]),
        "block_inter": int(best_tile["block_inter"]),
        "block_hidden": int(best_tile["block_hidden"]),
        "num_warps": int(best_tile["num_warps"]),
    }

    hybrid_rows: list[dict[str, Any]] = []
    hybrid_bench: dict[str, Any] = {}
    hybrid_peak: dict[str, Any] = {}
    for b, h in xs.items():
        for route_mode in ("current", "sort"):
            key = f"{route_mode}_M{b}"
            fn = lambda h=h, route_mode=route_mode: _hybrid_scratch(
                h,
                w,
                packed,
                route_mode=route_mode,
                top_k=top_k,
                tile=best_tile_cfg,
                down_block_hidden=args.down_block_hidden,
                down_block_inter=args.down_block_inter,
                down_num_warps=args.down_num_warps,
            )
            y = fn().detach()
            numeric[f"hybrid_{key}_vs_bf16"] = _diff_stats(y, refs[b])
            del y
            hybrid_peak[key] = _peak_once(fn)
            hybrid_bench[key] = _bench_cuda(fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
            hy = hybrid_bench[key]["median_us"]
            bf = bf16_full[str(b)]["median_us"]
            hybrid_rows.append({
                "batch": b,
                "route_mode": route_mode,
                "hybrid_us": hy,
                "bf16_full_us": bf,
                "hybrid_vs_bf16": bf / hy if hy else math.nan,
                "route_us": route_bench[route_mode][str(b)]["median_us"],
                "unique_experts": routes[route_mode][b]["unique_experts"],
            })
            print(
                f"[hybrid {route_mode} M={b}] hybrid={hy:.2f}us bf16={bf:.2f}us "
                f"speedup={bf / hy if hy else math.nan:.3f}x route={route_bench[route_mode][str(b)]['median_us']:.2f}us",
                flush=True,
            )

    numeric_pass = all(v["cosine"] > 0.999 and v["argmax_match"] for v in numeric.values())
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
    best_hybrid = min(hybrid_rows, key=lambda r: r["hybrid_us"])
    sort_hybrid_speedup_all = all(
        r["hybrid_vs_bf16"] >= 1.0 for r in hybrid_rows if r["route_mode"] == "sort"
    )
    result = {
        "schema": "lynn-stage6-p2e-scheduler-active-retune-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "shape": {"hidden": hidden, "num_experts": num_experts, "top_k": top_k},
        "bytes": {
            "bf16_layer_active_experts": active_bf16_bytes,
            "packed_layer_active_experts": packed_active_bytes,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "routes": {
            mode: {
                str(b): {
                    "unique_experts": routes[mode][b]["unique_experts"],
                    "top_group_counts": routes[mode][b]["top_group_counts"],
                }
                for b in batches
            }
            for mode in ("current", "sort")
        },
        "numeric": numeric,
        "bench": {
            "bf16_full_moe": bf16_full,
            "route": route_bench,
            "active_alloc_vs_scratch": active_bench,
            "active_rows": active_rows,
            "tile_sweep": tile_bench,
            "tile_rows": tile_rows,
            "best_tile": best_tile_cfg,
            "hybrid": hybrid_bench,
            "hybrid_rows": hybrid_rows,
            "best_hybrid": best_hybrid,
        },
        "memory": {"hybrid_peak": hybrid_peak},
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_active_shadow": bool(no_shadow_pass),
            "scratch_improves_alloc_all_batches": all(r["scratch_speedup"] > 1.0 for r in active_rows),
            "best_hybrid_speedup_vs_bf16": bool(best_hybrid["hybrid_vs_bf16"] >= 1.0),
            "sort_hybrid_speedup_vs_bf16_all_batches": bool(sort_hybrid_speedup_all),
            "all": bool(numeric_pass and no_shadow_pass and sort_hybrid_speedup_all),
        },
        "notes": [
            "This is a retune/profiling probe only; it does not edit resident_runner.",
            "The sort route mode still builds Python ExpertGroup objects, but replaces per-expert mask scans with one sort and unique_consecutive.",
            "Scratch mode reuses gate/up and down output buffers where the existing wrappers allow it.",
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
