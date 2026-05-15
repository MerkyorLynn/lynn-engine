#!/usr/bin/env python3
"""P38: multi-layer MoE segment profile.

P37 closed the broad block-retune line on one representative layer. P38 loads
the resident model once and samples several layers with the same MoE segment
probe so the next kernel work is based on a cross-layer bottleneck picture.
"""
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
    _shared_packed_native_fast_2d,
    _shared_packed_scalar_bridge,
)
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402

BEST_R6000_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_MOE_GATE_BLOCK_INTER": "8",
    "LYNN_MOE_GATE_BLOCK_HIDDEN": "256",
    "LYNN_MOE_DOWN_BLOCK_HIDDEN": "8",
    "LYNN_MOE_DOWN_BLOCK_INTER": "512",
    "LYNN_MOE_GATE_NUM_WARPS": "4",
    "LYNN_MOE_DOWN_NUM_WARPS": "8",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_LINEAR_BLOCK_GRAPH": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
}

PACKED_MOE_KEYS = (
    "mlp.experts._gate_up_packed",
    "mlp.experts._gate_up_scale",
    "mlp.experts._gate_up_global_scale",
    "mlp.experts._down_packed",
    "mlp.experts._down_scale",
    "mlp.experts._down_global_scale",
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

    current = moe_forward_decode_packed_nvfp4(h_moe, w, cfg).reshape_as(h_flat)
    active = _active_packed(
        h_flat,
        w,
        cfg,
        gate_block_inter=gate_block_inter,
        gate_block_hidden=gate_block_hidden,
        down_block_hidden=down_block_hidden,
        down_block_inter=down_block_inter,
    )
    shared_bf16 = _shared_bf16(h_flat, w)
    shared_scalar = _shared_packed_scalar_bridge(h_flat, w)
    shared_native = _shared_packed_native_fast_2d(h_flat, w)
    expert_ids, routing_weights = _router_topk(h_flat, w, cfg)

    timings: dict[str, float] = {}
    timings["router_topk_ms"] = _bench(lambda: _router_topk(h_flat, w, cfg), warmup, iters)
    timings["active_packed_ms"] = _bench(
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
    )
    if shared_bf16 is not None:
        timings["shared_bf16_ms"] = _bench(lambda: _shared_bf16(h_flat, w), warmup, iters)
    if shared_scalar is not None:
        timings["shared_packed_scalar_bridge_ms"] = _bench(lambda: _shared_packed_scalar_bridge(h_flat, w), warmup, iters)
    if shared_native is not None:
        timings["shared_packed_native_fast_2d_ms"] = _bench(lambda: _shared_packed_native_fast_2d(h_flat, w), warmup, iters)
    timings["current_full_moe_ms"] = _bench(lambda: moe_forward_decode_packed_nvfp4(h_moe, w, cfg), warmup, iters)

    expected_bf16 = active + shared_bf16 if shared_bf16 is not None else active
    expected_scalar = active + shared_scalar if shared_scalar is not None else None
    expected_native = active + shared_native if shared_native is not None else None
    total_measured = sum(
        timings.get(k, 0.0)
        for k in ("router_topk_ms", "active_packed_ms", "shared_bf16_ms")
    )

    return {
        "layer": layer_idx,
        "layer_type": LAYER_TYPES[layer_idx],
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.float().tolist()],
        "timings_ms": timings,
        "timings_tps_equiv": {k.replace("_ms", "_tps"): 1000.0 / v for k, v in timings.items()},
        "router_active_shared_sum_ms": total_measured,
        "current_vs_active_plus_shared_bf16": _diff(current, expected_bf16),
        "current_vs_active_plus_shared_scalar": _diff(current, expected_scalar),
        "current_vs_active_plus_shared_native": _diff(current, expected_native),
        "shared_scalar_vs_bf16": _diff(shared_scalar, shared_bf16),
        "shared_native_vs_bf16": _diff(shared_native, shared_bf16),
        "shared_native_vs_scalar": _diff(shared_native, shared_scalar),
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row["timings_ms"][key] for row in rows if not row.get("skipped") and key in row["timings_ms"]]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="2,8,14,20,28,36")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument(
        "--no-best-r6000-env",
        action="store_true",
        help="Do not install the current best R6000 decode env before constructing the runner.",
    )
    args = ap.parse_args()

    applied_env: dict[str, str] = {}
    if not args.no_best_r6000_env:
        for key, value in BEST_R6000_ENV.items():
            os.environ.setdefault(key, value)
            applied_env[key] = os.environ[key]

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    for layer in layers:
        if not (0 <= layer < runner.n_layers):
            raise ValueError(f"layer {layer} out of range [0, {runner.n_layers})")

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
        "router_topk_ms_mean": _avg(rows, "router_topk_ms"),
        "active_packed_ms_mean": _avg(rows, "active_packed_ms"),
        "shared_bf16_ms_mean": _avg(rows, "shared_bf16_ms"),
        "current_full_moe_ms_mean": _avg(rows, "current_full_moe_ms"),
        "profiled_layer_count": sum(1 for row in rows if not row.get("skipped")),
        "skipped_layer_count": sum(1 for row in rows if row.get("skipped")),
        "sampled_layers": layers,
    }
    result = {
        "schema_version": "lynn-engine-p38-moe-multilayer-profile-v1",
        "model": args.model,
        "prompt": args.prompt,
        "decode_position": decode_position,
        "native_prepared": runner.packed_decode_native_prepared,
        "packed_decode_backend": runner.packed_decode_backend,
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
        "note": "Profiles MoE subsegments across sampled layers; attention/setup is outside timed regions.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
