#!/usr/bin/env python3
"""Stage 6 Phase 2-B: routed gate/up grouping lower-bound.

P2-A proved a single expert's packed gate/up component is memory-clean but slow
against BF16. P2-B measures the real routed shape: keep the current prefill
router, pre-group token/slot rows by unique expert, then call the P2-A packed
kernel once per unique expert. Down projection, route weighting, index_add, and
shared expert remain out of scope.
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
from triton_kernels.nvfp4_moe import nvfp4_prefill_gate_up_silu_one_expert  # noqa: E402


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


def _attach_gate_up_packed(model_dir: Path, layer_idx: int, *, device: str) -> dict[str, torch.Tensor]:
    base = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
    packed, scale, global_scale = load_grouped_nvfp4_weight(model_dir, base, device=device)
    return {
        "packed": packed,
        "scale": scale,
        "global_scale": global_scale,
    }


def _silu_gate_up_ref(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _route_groups(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> tuple[torch.Tensor, list[ExpertGroup]]:
    logits = F.linear(x, gate_weight)
    _, expert_indices = torch.topk(logits, top_k, dim=-1)
    groups: list[ExpertGroup] = []
    for e in torch.unique(expert_indices).tolist():
        mask = expert_indices == int(e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        groups.append(ExpertGroup(int(e), token_idx.contiguous(), slot_idx.contiguous()))
    groups.sort(key=lambda g: len(g.token_idx), reverse=True)
    return expert_indices, groups


def _grouped_bf16_gateup(
    x: torch.Tensor,
    groups: list[ExpertGroup],
    gate_up_bf16: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], top_k, gate_up_bf16.shape[1] // 2), device=x.device, dtype=torch.bfloat16)
    for group in groups:
        y = _silu_gate_up_ref(x[group.token_idx], gate_up_bf16[group.expert])
        out[group.token_idx, group.slot_idx] = y
    return out


def _grouped_packed_gateup(
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
            packed["packed"],
            packed["scale"],
            packed["global_scale"],
            block_t=block_t,
            block_inter=block_inter,
            block_hidden=block_hidden,
        )
        out[group.token_idx, group.slot_idx] = y
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
    packed = _attach_gate_up_packed(model_dir, args.layer, device=device)
    hidden = int(gate_weight.shape[1])
    intermediate = int(gate_up_bf16.shape[1] // 2)

    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((b, hidden), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }
    routes: dict[int, dict[str, Any]] = {}
    for b, x in xs.items():
        expert_indices, groups = _route_groups(x, gate_weight, top_k)
        routes[b] = {
            "expert_indices": expert_indices,
            "groups": groups,
            "unique_experts": len(groups),
            "top_group_counts": [
                {"expert": g.expert, "rows": int(g.token_idx.numel())}
                for g in groups[:12]
            ],
        }

    bf16_shadow_bytes = _nbytes(gate_up_bf16)
    packed_bytes = _nbytes(packed["packed"]) + _nbytes(packed["scale"]) + _nbytes(packed["global_scale"])
    print("=============== STAGE 6 PHASE 2-B ROUTED GATE/UP GROUPING POC ===============", flush=True)
    print(f"model       : {model_dir}", flush=True)
    print(f"layer       : {args.layer}", flush=True)
    print(f"batches     : {batches}", flush=True)
    print(f"shape       : hidden={hidden} intermediate={intermediate} top_k={top_k}", flush=True)
    print(f"BF16 gateup : {bf16_shadow_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed      : {packed_bytes / GIB:.3f} GiB", flush=True)

    numeric: dict[str, Any] = {}
    for b, x in xs.items():
        groups = routes[b]["groups"]
        ref = _grouped_bf16_gateup(x, groups, gate_up_bf16, top_k=top_k).detach()
        y = _grouped_packed_gateup(
            x,
            groups,
            packed,
            top_k=top_k,
            block_t=args.block_t,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
        ).detach()
        numeric[str(b)] = _diff_stats(y, ref)
        print(
            f"[numeric M={b} unique={routes[b]['unique_experts']}] "
            f"cos={numeric[str(b)]['cosine']:.9f} rel_l2={numeric[str(b)]['rel_l2']:.3e} "
            f"argmax={numeric[str(b)]['argmax_match']}",
            flush=True,
        )

    bench_bf16: dict[str, Any] = {}
    for b, x in xs.items():
        groups = routes[b]["groups"]
        bench_bf16[str(b)] = _bench_cuda(
            lambda x=x, groups=groups: _grouped_bf16_gateup(x, groups, gate_up_bf16, top_k=top_k),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    del w["mlp.experts.gate_up_proj"], gate_up_bf16
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()

    bench_packed: dict[str, Any] = {}
    peak_packed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, x in xs.items():
        groups = routes[b]["groups"]
        peak_packed[str(b)] = _peak_once(
            lambda x=x, groups=groups: _grouped_packed_gateup(
                x,
                groups,
                packed,
                top_k=top_k,
                block_t=args.block_t,
                block_inter=args.block_inter,
                block_hidden=args.block_hidden,
            )
        )
        bench_packed[str(b)] = _bench_cuda(
            lambda x=x, groups=groups: _grouped_packed_gateup(
                x,
                groups,
                packed,
                top_k=top_k,
                block_t=args.block_t,
                block_inter=args.block_inter,
                block_hidden=args.block_hidden,
            ),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )
        bp = bench_packed[str(b)]
        bb = bench_bf16[str(b)]
        speedup = bb["median_us"] / bp["median_us"] if bp["median_us"] else math.nan
        rows.append({
            "batch": b,
            "unique_experts": routes[b]["unique_experts"],
            "packed_median_us": bp["median_us"],
            "bf16_median_us": bb["median_us"],
            "speedup_vs_bf16": speedup,
            "packed_us_per_token": bp["median_us"] / b,
            "bf16_us_per_token": bb["median_us"] / b,
        })
        print(
            f"[bench M={b} unique={routes[b]['unique_experts']}] packed={bp['median_us']:.2f}us "
            f"bf16={bb['median_us']:.2f}us speedup={speedup:.3f}x",
            flush=True,
        )

    numeric_pass = all(
        numeric[str(b)]["cosine"] > 0.999 and numeric[str(b)]["argmax_match"]
        for b in batches
    )
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w
    result = {
        "schema": "lynn-stage6-p2b-routed-gateup-grouping-poc-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "tile": {
            "block_t": args.block_t,
            "block_inter": args.block_inter,
            "block_hidden": args.block_hidden,
        },
        "shape": {
            "hidden": hidden,
            "expert_intermediate": intermediate,
            "top_k": top_k,
            "num_experts": num_experts,
        },
        "bytes": {
            "bf16_layer_gate_up": bf16_shadow_bytes,
            "packed_layer_gate_up": packed_bytes,
            "bf16_to_packed_ratio": bf16_shadow_bytes / packed_bytes if packed_bytes else None,
            "mem_after_deleting_bf16_gate_up_gib": mem_after_delete,
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
            "bf16_grouped_gateup": bench_bf16,
            "packed_grouped_gateup": bench_packed,
        },
        "memory": {
            "packed_peak": peak_packed,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_gate_up_shadow_for_packed_bench": bool(no_shadow_pass),
            "perf_speedup_vs_bf16_all_batches": all(r["speedup_vs_bf16"] >= 1.0 for r in rows),
            "all": bool(numeric_pass and no_shadow_pass and all(r["speedup_vs_bf16"] >= 1.0 for r in rows)),
        },
        "notes": [
            "Routes and groups are precomputed; this is a lower-bound gate/up grouping cost.",
            "Down projection, routing-weight multiply, index_add, and shared expert are out of scope.",
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
