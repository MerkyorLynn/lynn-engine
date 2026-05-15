#!/usr/bin/env python3
"""P39: split active routed MoE into gate/up and down segments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p15_moe_packed_segment_profile import (  # noqa: E402
    _active_packed,
    _bench,
    _diff,
    _prepare_layer_moe_input,
    _prefill,
    _router_topk,
    _shared_bf16,
)
from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV, PACKED_MOE_KEYS  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _profile_layer(
    runner: LynnIncrementalRunner,
    base_state: LynnInferenceState,
    token_id: int,
    decode_position: int,
    layer_idx: int,
    *,
    warmup: int,
    iters: int,
    gate_block_inter: int,
    gate_block_hidden: int,
    down_block_hidden: int,
    down_block_inter: int,
) -> dict[str, Any]:
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    runner._restore_state(state, runner._snapshot_state(base_state))
    h_moe = _prepare_layer_moe_input(runner, state, token_id, decode_position, layer_idx)
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    w = runner.layer_weights[layer_idx]
    cfg = runner.layer_cfgs[layer_idx]
    missing = [key for key in PACKED_MOE_KEYS if key not in w]
    if missing:
        return {
            "layer": layer_idx,
            "layer_type": LAYER_TYPES[layer_idx],
            "skipped": True,
            "skip_reason": "packed_nvfp4_moe_aliases_missing",
            "missing_keys": missing,
        }

    expert_ids, routing_weights = _router_topk(h_flat, w, cfg)
    expert_ids = expert_ids.to(torch.int32).contiguous()
    hidden = h_flat[0]

    def gate_up():
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            w["mlp.experts._gate_up_packed"],
            w["mlp.experts._gate_up_scale"],
            w["mlp.experts._gate_up_global_scale"],
            block_inter=gate_block_inter,
            block_hidden=gate_block_hidden,
            num_warps=int(os.environ.get("LYNN_MOE_GATE_NUM_WARPS", "4")),
        )

    inter = gate_up()

    def down():
        return nvfp4_grouped_down_weighted_sum(
            inter,
            expert_ids,
            routing_weights,
            w["mlp.experts._down_packed"],
            w["mlp.experts._down_scale"],
            w["mlp.experts._down_global_scale"],
            block_hidden=down_block_hidden,
            block_inter=down_block_inter,
            num_warps=int(os.environ.get("LYNN_MOE_DOWN_NUM_WARPS", "8")),
        )

    split_out = down().reshape_as(h_flat)
    combined_out = _active_packed(
        h_flat,
        w,
        cfg,
        gate_block_inter=gate_block_inter,
        gate_block_hidden=gate_block_hidden,
        down_block_hidden=down_block_hidden,
        down_block_inter=down_block_inter,
    )
    shared = _shared_bf16(h_flat, w)

    timings = {
        "router_topk_ms": _bench(lambda: _router_topk(h_flat, w, cfg), warmup, iters),
        "gate_up_ms": _bench(gate_up, warmup, iters),
        "down_ms": _bench(down, warmup, iters),
        "active_combined_ms": _bench(
            lambda: _active_packed(
                h_flat,
                w,
                cfg,
                gate_block_inter=gate_block_inter,
                gate_block_hidden=gate_block_hidden,
                down_block_hidden=down_block_hidden,
                down_block_inter=down_block_inter,
            ),
            warmup,
            iters,
        ),
    }
    if shared is not None:
        timings["shared_bf16_ms"] = _bench(lambda: _shared_bf16(h_flat, w), warmup, iters)

    return {
        "layer": layer_idx,
        "layer_type": LAYER_TYPES[layer_idx],
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.float().tolist()],
        "timings_ms": timings,
        "timings_tps_equiv": {k.replace("_ms", "_tps"): 1000.0 / v for k, v in timings.items()},
        "split_sum_ms": timings["gate_up_ms"] + timings["down_ms"],
        "split_vs_combined": _diff(split_out, combined_out),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row["timings_ms"][key] for row in rows if not row.get("skipped") and key in row["timings_ms"]]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="2,8,14,20,28,36")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--no-best-r6000-env", action="store_true")
    args = ap.parse_args()

    applied_env: dict[str, str] = {}
    if not args.no_best_r6000_env:
        for key, value in BEST_R6000_ENV.items():
            os.environ.setdefault(key, value)
            applied_env[key] = os.environ[key]

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    base_state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    token_id, decode_position = _prefill(runner, base_state, args.prompt)
    rows = [
        _profile_layer(
            runner,
            base_state,
            token_id,
            decode_position,
            layer,
            warmup=args.warmup,
            iters=args.iters,
            gate_block_inter=args.gate_block_inter,
            gate_block_hidden=args.gate_block_hidden,
            down_block_hidden=args.down_block_hidden,
            down_block_inter=args.down_block_inter,
        )
        for layer in layers
    ]
    summary = {
        "router_topk_ms_mean": _mean(rows, "router_topk_ms"),
        "gate_up_ms_mean": _mean(rows, "gate_up_ms"),
        "down_ms_mean": _mean(rows, "down_ms"),
        "active_combined_ms_mean": _mean(rows, "active_combined_ms"),
        "shared_bf16_ms_mean": _mean(rows, "shared_bf16_ms"),
        "sampled_layers": layers,
        "profiled_layer_count": sum(1 for row in rows if not row.get("skipped")),
        "skipped_layer_count": sum(1 for row in rows if row.get("skipped")),
    }
    result = {
        "schema_version": "lynn-engine-p39-active-moe-inner-profile-v1",
        "model": args.model,
        "prompt": args.prompt,
        "decode_position": decode_position,
        "moe_impl": runner.moe_impl,
        "applied_env": applied_env,
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
        },
        "summary": summary,
        "layers": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
