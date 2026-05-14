#!/usr/bin/env python3
"""P3-H probe: multi-seed layer decode gate for packed NVFP4 bridge.

P3-G proves one full layer decode path works. P3-H repeats the same layer-level
bridge over multiple hidden-state seeds to catch router sensitivity. This is the
cheap regression gate before replacing scalar bridge kernels with tensor-core
FP4 GEMM internals.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer

from benchmarks.p3_nvfp4_layer_decode_packed_probe import (
    LINEAR_ATTN_WEIGHT_NAMES,
    PackedExpert,
    _cfg_from,
    _compare,
    _layer_decode_packed,
    _layer_decode_resident,
    _load_packed_linear,
    _make_inputs,
    _packed_moe_decode,
    _route,
)


def _route_margin(h: torch.Tensor, w: dict, cfg: dict) -> float:
    """Return top-k boundary margin: logit[k-1] - logit[k]."""
    h_flat = h.view(-1, h.shape[-1])
    logits = torch.nn.functional.linear(h_flat, w["mlp.gate.weight"])
    values, _ = torch.topk(logits, cfg["num_experts_per_tok"] + 1, dim=-1)
    margin = values[0, cfg["num_experts_per_tok"] - 1] - values[0, cfg["num_experts_per_tok"]]
    return float(margin.float())


def _run_one(
    *,
    seed: int,
    v8_dir: Path,
    layer: int,
    resident_w: dict,
    packed_w: dict,
    cfg: dict,
    expert_cache: dict[int, PackedExpert],
    device: str,
    dtype: torch.dtype,
    cosine_threshold: float,
    rel_l2_threshold: float,
) -> dict[str, Any]:
    h_new, recurrent_state, conv_state = _make_inputs(device, dtype, seed)

    resident = _layer_decode_resident(
        h_new,
        resident_w,
        cfg,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    _, resident_route = _route(resident["h_moe_norm"], resident_w, cfg)

    packed_prefix = _layer_decode_resident(
        h_new,
        packed_w,
        cfg,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    _, packed_route = _route(packed_prefix["h_moe_norm"], resident_w, cfg)
    active_ids = sorted(
        set(int(x) for x in resident_route[0].tolist())
        | set(int(x) for x in packed_route[0].tolist())
    )
    for expert_id in active_ids:
        if expert_id not in expert_cache:
            expert_cache[expert_id] = PackedExpert.from_safetensors(v8_dir, layer, expert_id, device)

    packed, packed_trace = _layer_decode_packed(
        h_new,
        packed_w,
        cfg,
        expert_cache,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    packed_moe_same_input, same_input_trace = _packed_moe_decode(
        resident["h_moe_norm"],
        resident_w,
        cfg,
        expert_cache,
    )
    torch.cuda.synchronize()

    final_cmp = _compare(packed["h_out"], resident["h_out"])
    same_input_cmp = _compare(packed_moe_same_input, resident["moe_out"])
    resident_ids = [int(x) for x in resident_route[0].tolist()]
    packed_ids = packed_trace["expert_ids"]
    topk_exact = resident_ids == packed_ids
    topk_set_match = set(resident_ids) == set(packed_ids)
    pass_one = (
        final_cmp["cosine"] >= cosine_threshold
        and final_cmp["rel_l2"] <= rel_l2_threshold
        and same_input_cmp["cosine"] >= cosine_threshold
        and topk_set_match
    )
    route_status = "exact" if topk_exact else ("order_only_mismatch" if topk_set_match else "set_mismatch")
    return {
        "seed": seed,
        "verdict": "PASS" if pass_one else "FAIL",
        "topk_exact_match": topk_exact,
        "topk_set_match": topk_set_match,
        "route_status": route_status,
        "resident_expert_ids": resident_ids,
        "packed_expert_ids": packed_ids,
        "same_input_expert_ids": same_input_trace["expert_ids"],
        "route_boundary_margin": _route_margin(resident["h_moe_norm"], resident_w, cfg),
        "active_union_size": len(active_ids),
        "comparisons": {
            "attn_out": _compare(packed["attn_out"], resident["attn_out"]),
            "moe_out_same_h_moe_norm": same_input_cmp,
            "moe_out_full_layer": _compare(packed["moe_out"], resident["moe_out"]),
            "final_layer_output": final_cmp,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True, help="NVFP4 v8-RTN checkpoint dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--start-seed", type=int, default=20260514)
    ap.add_argument("--seed-count", type=int, default=8)
    ap.add_argument("--cosine-threshold", type=float, default=0.999)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.02)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)
    cfg, layer_types = _cfg_from(v8_dir)
    layer_type = layer_types[args.layer]
    if layer_type != "linear_attention":
        raise ValueError(f"P3-H currently targets a linear_attention layer, got {layer_type!r}")

    resident_w, _ = load_qwen36_layer(
        str(v8_dir),
        args.layer,
        num_experts=cfg["num_experts"],
        device=args.device,
        dequant_dtype=dtype,
    )
    packed_w = copy.copy(resident_w)
    for name in LINEAR_ATTN_WEIGHT_NAMES:
        packed_w[name] = _load_packed_linear(v8_dir, args.layer, name, args.device)

    expert_cache: dict[int, PackedExpert] = {}
    seeds = [args.start_seed + i for i in range(args.seed_count)]
    cases = [
        _run_one(
            seed=seed,
            v8_dir=v8_dir,
            layer=args.layer,
            resident_w=resident_w,
            packed_w=packed_w,
            cfg=cfg,
            expert_cache=expert_cache,
            device=args.device,
            dtype=dtype,
            cosine_threshold=args.cosine_threshold,
            rel_l2_threshold=args.rel_l2_threshold,
        )
        for seed in seeds
    ]

    final_cosines = [case["comparisons"]["final_layer_output"]["cosine"] for case in cases]
    final_rel_l2 = [case["comparisons"]["final_layer_output"]["rel_l2"] for case in cases]
    route_margins = [case["route_boundary_margin"] for case in cases]
    pass_count = sum(case["verdict"] == "PASS" for case in cases)
    topk_exact_count = sum(bool(case["topk_exact_match"]) for case in cases)
    topk_set_count = sum(bool(case["topk_set_match"]) for case in cases)
    order_only_mismatch_count = sum(case["route_status"] == "order_only_mismatch" for case in cases)

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-layer-decode-multiseed-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "layer_type": layer_type,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "thresholds": {
            "cosine": args.cosine_threshold,
            "rel_l2": args.rel_l2_threshold,
            "topk_set_match_required": True,
            "topk_order_mismatch": "WARN",
        },
        "summary": {
            "seed_count": len(seeds),
            "pass_count": pass_count,
            "topk_exact_count": topk_exact_count,
            "topk_set_count": topk_set_count,
            "order_only_mismatch_count": order_only_mismatch_count,
            "cached_packed_experts": len(expert_cache),
            "min_final_cosine": min(final_cosines),
            "avg_final_cosine": sum(final_cosines) / len(final_cosines),
            "max_final_rel_l2": max(final_rel_l2),
            "min_route_boundary_margin": min(route_margins),
            "avg_route_boundary_margin": sum(route_margins) / len(route_margins),
        },
        "cases": cases,
        "notes": [
            "P3-H repeats P3-G over multiple deterministic hidden-state seeds.",
            "A top-k set mismatch is FAIL. An order-only mismatch is WARN because MoE expert summation is order-invariant.",
            "This is still the scalar bridge path, not tensor-core FP4 GEMM.",
        ],
    }
    result["verdict"] = "PASS" if pass_count == len(seeds) else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
