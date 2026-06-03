#!/usr/bin/env python3
"""Stage 6 Phase 2-A: single-expert batched packed gate/up PoC.

P2 census locked the next real target to routed grouped MoE prefill. This PoC
tests the smallest useful slice: one expert, M>1 hidden rows, packed NVFP4
gate/up weights, fused SwiGLU intermediate output. It deliberately excludes
down projection, route weighting, index_add, and shared expert.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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


def _choose_hot_expert(x: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> tuple[int, dict[str, Any]]:
    logits = F.linear(x, gate_weight)
    _, expert_indices = torch.topk(logits, top_k, dim=-1)
    flat = expert_indices.flatten()
    counts = torch.bincount(flat, minlength=gate_weight.shape[0])
    expert = int(torch.argmax(counts).item())
    return expert, {
        "counts_top8": [
            {"expert": int(i), "count": int(counts[i].item())}
            for i in torch.topk(counts, min(8, counts.numel())).indices.tolist()
        ],
        "selected_count": int(counts[expert].item()),
    }


def _cuda_mem_gib() -> float:
    return float(torch.cuda.memory_allocated() / GIB)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batches", default="1,4,16,64")
    ap.add_argument("--expert-id", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block-t", type=int, default=16)
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
    max_batch = max(batches)
    if args.expert_id >= 0:
        expert_id = int(args.expert_id)
        expert_info = {"selected_count": None, "counts_top8": []}
    else:
        expert_id, expert_info = _choose_hot_expert(xs[max_batch], gate_weight, top_k)

    bf16_shadow_bytes = _nbytes(gate_up_bf16[expert_id])
    packed_expert_bytes = (
        _nbytes(packed["packed"][expert_id])
        + _nbytes(packed["scale"][expert_id])
        + _nbytes(packed["global_scale"])
    )
    print("=============== STAGE 6 PHASE 2-A GATE/UP PREFILL POC ===============", flush=True)
    print(f"model       : {model_dir}", flush=True)
    print(f"layer       : {args.layer}", flush=True)
    print(f"expert      : {expert_id} info={expert_info}", flush=True)
    print(f"batches     : {batches}", flush=True)
    print(f"shape       : hidden={hidden} intermediate={intermediate}", flush=True)
    print(f"BF16 expert : {bf16_shadow_bytes / 1024**2:.2f} MiB", flush=True)
    print(f"packed exp  : {packed_expert_bytes / 1024**2:.2f} MiB", flush=True)

    numeric: dict[str, Any] = {}
    refs: dict[int, torch.Tensor] = {}
    for b, x in xs.items():
        ref = _silu_gate_up_ref(x, gate_up_bf16[expert_id]).detach()
        y = nvfp4_prefill_gate_up_silu_one_expert(
            x,
            expert_id,
            packed["packed"],
            packed["scale"],
            packed["global_scale"],
            block_t=args.block_t,
            block_inter=args.block_inter,
            block_hidden=args.block_hidden,
        ).detach()
        refs[b] = ref
        numeric[str(b)] = _diff_stats(y, ref)
        print(
            f"[numeric M={b}] cos={numeric[str(b)]['cosine']:.9f} "
            f"rel_l2={numeric[str(b)]['rel_l2']:.3e} argmax={numeric[str(b)]['argmax_match']}",
            flush=True,
        )

    bench_bf16: dict[str, Any] = {}
    for b, x in xs.items():
        bench_bf16[str(b)] = _bench_cuda(
            lambda x=x: _silu_gate_up_ref(x, gate_up_bf16[expert_id]),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        )

    del w["mlp.experts.gate_up_proj"], gate_up_bf16, refs
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mem_after_delete = _cuda_mem_gib()
    torch.cuda.reset_peak_memory_stats()

    bench_packed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for b, x in xs.items():
        bench_packed[str(b)] = _bench_cuda(
            lambda x=x: nvfp4_prefill_gate_up_silu_one_expert(
                x,
                expert_id,
                packed["packed"],
                packed["scale"],
                packed["global_scale"],
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
            "packed_median_us": bp["median_us"],
            "bf16_median_us": bb["median_us"],
            "speedup_vs_bf16": speedup,
            "packed_us_per_token": bp["median_us"] / b,
            "bf16_us_per_token": bb["median_us"] / b,
        })
        print(
            f"[bench M={b}] packed={bp['median_us']:.2f}us "
            f"bf16={bb['median_us']:.2f}us speedup={speedup:.3f}x",
            flush=True,
        )

    peak_gib = float(torch.cuda.max_memory_allocated() / GIB)
    numeric_pass = all(
        numeric[str(b)]["cosine"] > 0.999
        and numeric[str(b)]["argmax_match"]
        for b in batches
    )
    perf_pass = all(r["speedup_vs_bf16"] >= 1.0 for r in rows)
    no_shadow_pass = "mlp.experts.gate_up_proj" not in w
    result = {
        "schema": "lynn-stage6-p2a-gateup-prefill-poc-v1",
        "model": str(model_dir),
        "layer": args.layer,
        "expert_id": expert_id,
        "expert_selection": expert_info,
        "seed": args.seed,
        "batches": batches,
        "tile": {
            "block_t": args.block_t,
            "block_inter": args.block_inter,
            "block_hidden": args.block_hidden,
        },
        "shape": {"hidden": hidden, "expert_intermediate": intermediate},
        "bytes": {
            "bf16_one_expert_gate_up": bf16_shadow_bytes,
            "packed_one_expert_gate_up": packed_expert_bytes,
            "bf16_to_packed_ratio": bf16_shadow_bytes / packed_expert_bytes if packed_expert_bytes else None,
        },
        "numeric": numeric,
        "bench": {
            "rows": rows,
            "bf16_gate_up_silu": bench_bf16,
            "packed_gate_up_silu": bench_packed,
        },
        "memory_after_deleting_bf16_gate_up": {
            "allocated_gib": mem_after_delete,
            "peak_packed_bench_gib": peak_gib,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_gate_up_shadow_for_packed_bench": bool(no_shadow_pass),
            "perf_speedup_vs_bf16_all_batches": bool(perf_pass),
            "all": bool(numeric_pass and no_shadow_pass and perf_pass),
        },
        "notes": [
            "This PoC covers one expert's gate/up+SwiGLU only.",
            "It does not include down projection, route weighting, index_add, or shared expert.",
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
