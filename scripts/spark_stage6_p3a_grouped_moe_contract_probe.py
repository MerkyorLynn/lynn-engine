#!/usr/bin/env python3
"""Stage 6 Phase 3-A: grouped active-MoE contract probe.

P3-A is the bridge from the P2-E packed-prefill composition to the eventual
fused grouped-MoE kernel. This probe is active-expert only: router inputs are
computed once, shared expert is excluded, and the candidate receives exactly the
P3 contract tensors.

It intentionally does not claim a banked fused kernel. A PASS only means the
contract-shaped callable is numerically aligned, reads packed active weights
after BF16 active shadows are removed, and reports honest temporary/speed data.
"""
from __future__ import annotations

import argparse
import json
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
from engine.moe_packed_nvfp4 import active_moe_grouped_prefill_p3a  # noqa: E402
from scripts.spark_stage6_p1_dense_projection_poc import (  # noqa: E402
    _bench_cuda,
    _diff_stats,
    _nbytes,
)
from scripts.spark_stage6_p2_grouped_moe_prefill_census import _attach_packed_moe  # noqa: E402


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


def _route(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[ExpertGroup]]:
    logits = F.linear(hidden, gate_weight)
    routing_logits, expert_ids = torch.topk(logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_logits, dim=-1, dtype=torch.float32).contiguous()
    flat_experts = expert_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_experts)
    sorted_experts = flat_experts[order]
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    token_base = torch.arange(hidden.shape[0], device=hidden.device, dtype=torch.long).repeat_interleave(top_k)
    slot_base = torch.arange(top_k, device=hidden.device, dtype=torch.long).repeat(hidden.shape[0])
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
    return expert_ids.to(torch.int32).contiguous(), routing_weights, groups


def _bf16_active_moe(
    hidden: torch.Tensor,
    groups: list[ExpertGroup],
    routing_weights: torch.Tensor,
    gate_up_bf16: torch.Tensor,
    down_bf16: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(hidden)
    for group in groups:
        x_e = hidden[group.token_idx]
        gate_up = F.linear(x_e, gate_up_bf16[group.expert])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate) * up
        down = F.linear(inter, down_bf16[group.expert])
        weights = routing_weights[group.token_idx, group.slot_idx].to(hidden.dtype).unsqueeze(-1)
        out.index_add_(0, group.token_idx, down * weights)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="1,16,64")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--block-t", type=int, default=32)
    ap.add_argument("--block-inter", type=int, default=8)
    ap.add_argument("--block-hidden", type=int, default=128)
    ap.add_argument("--num-warps", type=int, default=4)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--down-num-warps", type=int, default=8)
    ap.add_argument("--min-cosine", type=float, default=0.999)
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
    layer_cfg["layer_idx"] = int(args.layer)
    packed = _attach_packed_moe(model_dir, args.layer, w, device=device)

    gate_weight = w["mlp.gate.weight"]
    gate_up_bf16 = w["mlp.experts.gate_up_proj"]
    down_bf16 = w["mlp.experts.down_proj"]
    hidden_size = int(gate_weight.shape[1])
    intermediate = int(packed["gate_up_packed"].shape[1] // 2)
    xs: dict[int, torch.Tensor] = {
        b: (torch.randn((b, hidden_size), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
        for b in batches
    }

    active_bf16_bytes = _tensor_bytes([gate_up_bf16, down_bf16])
    packed_active_bytes = _tensor_bytes(list(packed.values()))
    max_inter_scratch_bytes = max(batches) * top_k * intermediate * torch.empty((), dtype=torch.bfloat16).element_size()

    print("=============== STAGE 6 PHASE 3-A GROUPED MOE CONTRACT PROBE ===============", flush=True)
    print(f"model        : {model_dir}", flush=True)
    print(f"layer        : {args.layer}", flush=True)
    print(f"batches      : {batches}", flush=True)
    print(f"shape        : hidden={hidden_size} intermediate={intermediate} experts={num_experts} top_k={top_k}", flush=True)
    print(f"BF16 active  : {active_bf16_bytes / GIB:.3f} GiB", flush=True)
    print(f"packed active: {packed_active_bytes / GIB:.3f} GiB", flush=True)

    routes: dict[int, dict[str, Any]] = {}
    refs: dict[int, torch.Tensor] = {}
    bench_bf16: dict[str, Any] = {}
    for b, hidden in xs.items():
        expert_ids, routing_weights, groups = _route(hidden, gate_weight, top_k)
        routes[b] = {
            "expert_ids": expert_ids,
            "routing_weights": routing_weights,
            "groups": groups,
            "unique_experts": len(groups),
        }
        ref_fn = lambda hidden=hidden, groups=groups, routing_weights=routing_weights: _bf16_active_moe(
            hidden,
            groups,
            routing_weights,
            gate_up_bf16,
            down_bf16,
        )
        refs[b] = ref_fn().detach()
        bench_bf16[str(b)] = _bench_cuda(ref_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

    del w["mlp.experts.gate_up_proj"], w["mlp.experts.down_proj"]
    del gate_up_bf16, down_bf16
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()
    shadow_absent = "mlp.experts.gate_up_proj" not in w and "mlp.experts.down_proj" not in w

    numeric: dict[str, Any] = {}
    bench_candidate: dict[str, Any] = {}
    peak_candidate: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, hidden in xs.items():
        route = routes[b]
        candidate_fn = lambda hidden=hidden, route=route: active_moe_grouped_prefill_p3a(
            hidden,
            route["expert_ids"],
            route["routing_weights"],
            packed["gate_up_packed"],
            packed["gate_up_scale"],
            packed["gate_up_global_scale"],
            packed["down_packed"],
            packed["down_scale"],
            packed["down_global_scale"],
            block_t=args.block_t,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
            num_warps=args.num_warps,
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
            down_num_warps=args.down_num_warps,
        )
        candidate = candidate_fn().detach()
        stats = _diff_stats(candidate, refs[b])
        numeric[str(b)] = stats
        peak_candidate[str(b)] = _peak_once(candidate_fn)
        bench_candidate[str(b)] = _bench_cuda(candidate_fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
        bf = bench_bf16[str(b)]["median_us"]
        cand = bench_candidate[str(b)]["median_us"]
        row = {
            "batch": b,
            "unique_experts": route["unique_experts"],
            "bf16_active_us": bf,
            "p3a_contract_us": cand,
            "p3a_vs_bf16": bf / cand if cand else None,
            "cosine": stats["cosine"],
            "argmax_match": stats["argmax_match"],
        }
        rows.append(row)
        print(
            f"[M={b}] experts={route['unique_experts']} bf16={bf:.2f}us "
            f"p3a={cand:.2f}us speed={row['p3a_vs_bf16']:.3f}x "
            f"cos={stats['cosine']:.9f} argmax={stats['argmax_match']}",
            flush=True,
        )

    numeric_pass = all(
        item["cosine"] >= args.min_cosine and item["argmax_match"]
        for item in numeric.values()
    )
    result = {
        "schema": "lynn-stage6-p3a-grouped-moe-contract-probe-v1",
        "verdict": "PASS" if numeric_pass and shadow_absent else "FAIL",
        "banked_fused_kernel": False,
        "model": str(model_dir),
        "layer": args.layer,
        "seed": args.seed,
        "batches": batches,
        "shape": {
            "hidden": hidden_size,
            "intermediate": intermediate,
            "num_experts": num_experts,
            "top_k": top_k,
        },
        "tiles": {
            "block_t": args.block_t,
            "block_inter": args.block_inter,
            "block_hidden": args.block_hidden,
            "num_warps": args.num_warps,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
            "down_num_warps": args.down_num_warps,
        },
        "bytes": {
            "bf16_layer_active_experts": active_bf16_bytes,
            "packed_layer_active_experts": packed_active_bytes,
            "max_inter_scratch_estimate": max_inter_scratch_bytes,
            "mem_after_deleting_bf16_active_gib": mem_after_delete,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_active": bench_bf16,
            "p3a_contract": bench_candidate,
        },
        "memory": {"p3a_candidate_peak": peak_candidate},
        "passes": {
            "numeric": bool(numeric_pass),
            "shadow_absent_at_candidate_start": bool(shadow_absent),
            "all": bool(numeric_pass and shadow_absent),
        },
        "notes": [
            "Active MoE only: shared expert and router are excluded from the P3-A contract.",
            "The candidate consumes packed active weights after BF16 active shadows are deleted.",
            "Speed is reported, but this probe does not bank a fused P3 kernel.",
        ],
    }
    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passes"]["all"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
