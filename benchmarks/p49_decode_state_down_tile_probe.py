#!/usr/bin/env python3
"""P49: true-decode-state down-tile parity probe.

P48 proved that the non-atomic tile-hidden down kernel can beat the current
Triton down kernel on a prefill-derived hidden sample.  Runtime greedy gates
still drifted, including single-layer allowlists, so P49 repeats the comparison
on the actual incremental-decode MoE input:

    prompt prefill -> choose next token -> run decode through previous layers
    -> stop at target layer after attention -> compare down kernels

If P49 is clean while runtime gates drift, the next target is graph/state
interaction or greedy sensitivity, not the isolated down kernel.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p15_moe_packed_segment_profile import (  # noqa: E402
    _bench,
    _prefill,
    _prepare_layer_moe_input,
)
from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV, PACKED_MOE_KEYS  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.native_cuda import load_lynn_native_extension  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _router_topk_decode(h_flat: torch.Tensor, w: dict, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=False,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0].contiguous()
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    return expert_ids, routing_weights


def _run_layer(
    runner: LynnIncrementalRunner,
    base_state: LynnInferenceState,
    ext,
    *,
    token_id: int,
    decode_position: int,
    layer: int,
    tile: int,
    warmup: int,
    iters: int,
) -> dict:
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    runner._restore_state(state, runner._snapshot_state(base_state))
    h_moe = _prepare_layer_moe_input(runner, state, token_id, decode_position, layer)
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    missing = [key for key in PACKED_MOE_KEYS if key not in w]
    if missing:
        return {
            "layer": layer,
            "layer_type": LAYER_TYPES[layer],
            "skipped": True,
            "skip_reason": "packed_nvfp4_moe_aliases_missing",
            "missing_keys": missing,
        }

    expert_ids, routing_weights = _router_topk_decode(h_flat, w, cfg)
    hidden = h_flat[0].contiguous()
    inter = nvfp4_grouped_gate_up_silu(
        hidden,
        expert_ids,
        w["mlp.experts._gate_up_packed"],
        w["mlp.experts._gate_up_scale"],
        w["mlp.experts._gate_up_global_scale"],
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )

    def triton_down() -> torch.Tensor:
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=8,
            block_inter=512,
            num_warps=8,
        )

    def tile_down() -> torch.Tensor:
        return ext.down_weighted_sum_tile_scalar(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            tile,
        )

    ref = triton_down()
    cand = tile_down()
    triton_ms = _bench(triton_down, warmup, iters)
    tile_ms = _bench(tile_down, warmup, iters)
    return {
        "layer": layer,
        "layer_type": LAYER_TYPES[layer],
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "timings_ms": {
            "triton_down_ms": triton_ms,
            "cuda_tile_down_ms": tile_ms,
            "tile_vs_triton_speedup": triton_ms / tile_ms,
        },
        "diff_vs_triton": _diff(ref, cand),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--tile", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rel-l2-threshold", type=float, default=1e-3)
    ap.add_argument("--cosine-threshold", type=float, default=0.99999)
    args = ap.parse_args()

    for key, value in BEST_R6000_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["LYNN_NATIVE_DOWN_BACKEND"] = "triton"
    os.environ["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = "triton"
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p49_decode_down_tile")

    ext = load_lynn_native_extension(verbose=False)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    base_state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    token_id, decode_position = _prefill(runner, base_state, args.prompt)
    cases = [
        _run_layer(
            runner,
            base_state,
            ext,
            token_id=token_id,
            decode_position=decode_position,
            layer=layer,
            tile=args.tile,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]
    active_cases = [c for c in cases if not c.get("skipped")]
    mean_triton = sum(c["timings_ms"]["triton_down_ms"] for c in active_cases) / len(active_cases)
    mean_tile = sum(c["timings_ms"]["cuda_tile_down_ms"] for c in active_cases) / len(active_cases)
    max_rel_l2 = max(c["diff_vs_triton"]["rel_l2"] for c in active_cases)
    min_cosine = min(c["diff_vs_triton"]["cosine"] for c in active_cases)
    result = {
        "schema_version": "lynn-engine-p49-decode-state-down-tile-probe-v1",
        "model": args.model,
        "prompt": args.prompt,
        "decode_position": decode_position,
        "first_decode_token_id": token_id,
        "tile": args.tile,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_triton_down_ms": mean_triton,
            "mean_cuda_tile_down_ms": mean_tile,
            "mean_tile_vs_triton_speedup": mean_triton / mean_tile,
            "max_rel_l2_vs_triton": max_rel_l2,
            "min_cosine_vs_triton": min_cosine,
            "decode_state_parity_pass": bool(
                max_rel_l2 <= args.rel_l2_threshold
                and min_cosine >= args.cosine_threshold
            ),
            "interpretation": (
                "down_tile_matches_true_decode_state; runtime drift likely graph/state or greedy-sensitivity"
                if max_rel_l2 <= args.rel_l2_threshold and min_cosine >= args.cosine_threshold
                else "down_tile_diff_amplifies_on_true_decode_state; keep kernel research-only"
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
