#!/usr/bin/env python3
"""Stage 6 Phase 2-C: active routed MoE lower-bound from packed NVFP4.

This composes the P2-B routed gate/up grouping with the existing packed down
weighted-sum kernel. Routes are precomputed and shared expert is out of scope,
so this is a lower-bound for the active routed expert path, not a serving path.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402
from engine.nvfp4_runtime import load_grouped_nvfp4_weight  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_prefill_gate_up_silu_one_expert,
)


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
GIB = 1024**3


@dataclass(frozen=True)
class ExpertGroup:
    expert: int
    token_idx: torch.Tensor
    slot_idx: torch.Tensor


def _parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _model_cfg(model_dir: Path) -> dict[str, Any]:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("text_config", cfg)


def _load_grouped_pair(model_dir: Path, layer_idx: int, *, device: str) -> dict[str, torch.Tensor]:
    base = f"model.language_model.layers.{layer_idx}.mlp.experts"
    gu_packed, gu_scale, gu_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{base}.gate_up_proj",
        device=device,
    )
    d_packed, d_scale, d_global = load_grouped_nvfp4_weight(
        model_dir,
        f"{base}.down_proj",
        device=device,
    )
    return {
        "gate_up_packed": gu_packed,
        "gate_up_scale": gu_scale,
        "gate_up_global_scale": gu_global,
        "down_packed": d_packed,
        "down_scale": d_scale,
        "down_global_scale": d_global,
    }


def _route(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor, list[ExpertGroup]]:
    logits = F.linear(x, gate_weight)
    routing_logits, expert_indices = torch.topk(logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_logits, dim=-1, dtype=torch.float32)
    groups: list[ExpertGroup] = []
    for e in torch.unique(expert_indices).tolist():
        mask = expert_indices == int(e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        groups.append(ExpertGroup(int(e), token_idx.contiguous(), slot_idx.contiguous()))
    groups.sort(key=lambda g: len(g.token_idx), reverse=True)
    return expert_indices.to(torch.int32).contiguous(), routing_weights.contiguous(), groups


def _bf16_active_moe(
    x: torch.Tensor,
    groups: list[ExpertGroup],
    routing_weights: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    down_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for group in groups:
        x_e = x[group.token_idx]
        gate_up = F.linear(x_e, gate_up_bf16[group.expert])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate) * up
        down = F.linear(inter, down_bf16[group.expert])
        weights = routing_weights[group.token_idx, group.slot_idx].to(x.dtype).unsqueeze(-1)
        out.index_add_(0, group.token_idx, down * weights)
    return out


def _packed_gateup_inter(
    x: torch.Tensor,
    groups: list[ExpertGroup],
    packed: dict[str, torch.Tensor],
    *,
    top_k: int,
    block_t: int,
    block_inter: int,
    block_hidden: int,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], top_k, 512), device=x.device, dtype=torch.bfloat16)
    for group in groups:
        y = nvfp4_prefill_gate_up_silu_one_expert(
            x[group.token_idx],
            group.expert,
            packed["gate_up_packed"],
            packed["gate_up_scale"],
            packed["gate_up_global_scale"],
            block_t=block_t,
            block_inter=block_inter,
            block_hidden=block_hidden,
        )
        out[group.token_idx, group.slot_idx] = y
    return out


def _packed_active_moe(
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
    down_block_hidden: int,
    down_block_inter: int,
) -> torch.Tensor:
    inter = _packed_gateup_inter(
        x,
        groups,
        packed,
        top_k=top_k,
        block_t=block_t,
        block_inter=block_inter,
        block_hidden=block_hidden,
    )
    out = torch.empty_like(x)
    for token in range(x.shape[0]):
        out[token] = nvfp4_grouped_down_weighted_sum(
            inter[token],
            expert_indices[token],
            routing_weights[token],
            packed["down_packed"],
            packed["down_scale"],
            packed["down_global_scale"],
            block_hidden=down_block_hidden,
            block_inter=down_block_inter,
        )
    return out


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
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

    text_cfg = _model_cfg(model_dir)
    num_experts = int(text_cfg.get("num_experts", 256))
    top_k = int(text_cfg.get("num_experts_per_tok", 8))
    w, _ = load_qwen36_layer(
        str(model_dir),
        args.layer,
        num_experts=num_experts,
        device=device,
        dequant_dtype=torch.bfloat16,
    )
    gate_weight = w["mlp.gate.weight"]
    gate_up_bf16 = w["mlp.experts.gate_up_proj"]
    down_bf16 = w["mlp.experts.down_proj"]
    packed = _load_grouped_pair(model_dir, args.layer, device=device)

    hidden = int(gate_weight.shape[1])
    intermediate = int(gate_up_bf16.shape[1] // 2)
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    routes: dict[int, dict[str, Any]] = {}
    for b, x in xs.items():
        expert_indices, routing_weights, groups = _route(x, gate_weight, top_k)
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

    bf16_shadow_bytes = _nbytes(gate_up_bf16) + _nbytes(down_bf16)
    packed_bytes = sum(_nbytes(t) for t in packed.values())
    print("=============== STAGE 6 PHASE 2-C ACTIVE MOE LOWER-BOUND POC ===============", flush=True)
    print(f"model       : {model_dir}", flush=True)
    print(f"layer       : {args.layer}", flush=True)
    print(f"batches     : {batches}", flush=True)
    print(f"shape       : hidden={hidden} intermediate={intermediate} top_k={top_k}", flush=True)
    print(f"BF16 active : {bf16_shadow_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed      : {packed_bytes / GIB:.3f} GiB", flush=True)

    numeric: dict[str, Any] = {}
    for b, x in xs.items():
        route = routes[b]
        ref = _bf16_active_moe(
            x,
            route["groups"],
            route["routing_weights"],
            gate_up_bf16,
            down_bf16,
        ).detach()
        y = _packed_active_moe(
            x,
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
        ).detach()
        numeric[str(b)] = _diff_stats(y, ref)
        print(
            f"[numeric M={b} unique={route['unique_experts']}] "
            f"cos={numeric[str(b)]['cosine']:.9f} rel_l2={numeric[str(b)]['rel_l2']:.3e} "
            f"argmax={numeric[str(b)]['argmax_match']}",
            flush=True,
        )

    bench_bf16: dict[str, Any] = {}
    for b, x in xs.items():
        route = routes[b]
        bench_bf16[str(b)] = _bench_cuda(
            lambda x=x, route=route: _bf16_active_moe(
                x,
                route["groups"],
                route["routing_weights"],
                gate_up_bf16,
                down_bf16,
            ),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"], gate_up_bf16, down_bf16
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    bench_packed: dict[str, Any] = {}
    peak_packed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, x in xs.items():
        route = routes[b]
        fn = lambda x=x, route=route: _packed_active_moe(
            x,
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
        )
        peak_packed[str(b)] = _peak_once(fn)
        bench_packed[str(b)] = _bench_cuda(
            fn,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bp = bench_packed[str(b)]
        bb = bench_bf16[str(b)]
        speedup = bb["median_us"] / bp["median_us"] if bp["median_us"] else math.nan
        rows.append({
            "batch": b,
            "unique_experts": route["unique_experts"],
            "packed_median_us": bp["median_us"],
            "bf16_median_us": bb["median_us"],
            "speedup_vs_bf16": speedup,
            "packed_us_per_token": bp["median_us"] / b,
            "bf16_us_per_token": bb["median_us"] / b,
        })
        print(
            f"[bench M={b} unique={route['unique_experts']}] packed={bp['median_us']:.2f}us "
            f"bf16={bb['median_us']:.2f}us speedup={speedup:.3f}x",
            flush=True,
        )

    numeric_pass = all(
        numeric[str(b)]["cosine"] > 0.999 and numeric[str(b)]["argmax_match"]
        for b in batches
    )
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w
    result = {
        "schema": "lynn-stage6-p2c-active-moe-lower-bound-poc-v1",
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
            "expert_intermediate": intermediate,
            "top_k": top_k,
            "num_experts": num_experts,
        },
        "bytes": {
            "bf16_layer_active_experts": bf16_shadow_bytes,
            "packed_layer_active_experts": packed_bytes,
            "bf16_to_packed_ratio": bf16_shadow_bytes / packed_bytes if packed_bytes else None,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "routes": {
            str(b): {
                "unique_experts": routes[b]["unique_experts"],
                "top_group_counts": routes[b]["top_group_counts"],
            }
            for b in batches
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_active_moe": bench_bf16,
            "packed_active_moe": bench_packed,
        },
        "memory": {
            "packed_peak": peak_packed,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_active_shadow_for_packed_bench": bool(no_shadow_pass),
            "perf_speedup_vs_bf16_all_batches": all(r["speedup_vs_bf16"] >= 1.0 for r in rows),
            "all": bool(numeric_pass and no_shadow_pass and all(r["speedup_vs_bf16"] >= 1.0 for r in rows)),
        },
        "notes": [
            "Routes are precomputed; this is a lower-bound active routed expert path.",
            "Shared expert, router latency, residual, and layernorm are out of scope.",
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
