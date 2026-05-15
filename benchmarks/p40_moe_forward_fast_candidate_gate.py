#!/usr/bin/env python3
"""P40: exact fast MoE forward candidate.

P39 showed that the active expert gate/up+down kernels are much cheaper than
the end-to-end MoE wall suggests. This gate compares the current production
`moe_forward_decode_packed_nvfp4` against a fixed-config candidate that keeps
the same math but removes optional branches/env lookups from the timed path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p15_moe_packed_segment_profile import (  # noqa: E402
    _bench,
    _diff,
    _prepare_layer_moe_input,
    _prefill,
)
from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV, PACKED_MOE_KEYS  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu,
)


def _candidate_fixed_moe(h: torch.Tensor, w: dict, cfg: dict) -> torch.Tensor:
    h_flat = h.reshape(-1, h.shape[-1])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(
        router_logits,
        int(cfg["num_experts_per_tok"]),
        dim=-1,
        sorted=False,
    )
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    hidden = h_flat[0]
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
    moe_out = nvfp4_grouped_down_weighted_sum(
        inter,
        expert_ids,
        routing_weights,
        w["mlp.experts._down_packed"],
        w["mlp.experts._down_scale"],
        w["mlp.experts._down_global_scale"],
        block_hidden=8,
        block_inter=512,
        num_warps=8,
    ).reshape_as(h_flat)
    if "mlp.shared_expert._gate_up_proj.weight" in w:
        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate_s, up_s = gate_up_s.chunk(2, dim=-1)
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    else:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
        shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return (moe_out + shared).to(h.dtype).reshape_as(h)


def _profile_layer(
    runner: LynnIncrementalRunner,
    base_state: LynnInferenceState,
    token_id: int,
    decode_position: int,
    layer_idx: int,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
    runner._restore_state(state, runner._snapshot_state(base_state))
    h_moe = _prepare_layer_moe_input(runner, state, token_id, decode_position, layer_idx)
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
    reference = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
    candidate = _candidate_fixed_moe(h_moe, w, cfg)
    timings = {
        "reference_ms": _bench(lambda: moe_forward_decode_packed_nvfp4(h_moe, w, cfg), warmup, iters),
        "candidate_fixed_ms": _bench(lambda: _candidate_fixed_moe(h_moe, w, cfg), warmup, iters),
    }
    return {
        "layer": layer_idx,
        "layer_type": LAYER_TYPES[layer_idx],
        "timings_ms": timings,
        "speedup": timings["reference_ms"] / timings["candidate_fixed_ms"],
        "diff": _diff(reference, candidate),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row["timings_ms"][key] for row in rows if not row.get("skipped")]
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="2,8,14,20,28,36")
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
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
        _profile_layer(runner, base_state, token_id, decode_position, layer, warmup=args.warmup, iters=args.iters)
        for layer in layers
    ]
    ref_mean = _mean(rows, "reference_ms")
    cand_mean = _mean(rows, "candidate_fixed_ms")
    result = {
        "schema_version": "lynn-engine-p40-moe-forward-fast-candidate-gate-v1",
        "model": args.model,
        "decode_position": decode_position,
        "moe_impl": runner.moe_impl,
        "applied_env": applied_env,
        "summary": {
            "reference_ms_mean": ref_mean,
            "candidate_fixed_ms_mean": cand_mean,
            "mean_speedup": (ref_mean / cand_mean) if ref_mean and cand_mean else None,
            "all_exact_or_trivial": all(
                (row.get("skipped") or row["diff"]["max_abs"] <= 6.2e-05 and row["diff"]["cosine"] >= 0.99999)
                for row in rows
            ),
            "sampled_layers": layers,
        },
        "layers": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["all_exact_or_trivial"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
