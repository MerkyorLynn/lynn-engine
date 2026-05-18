#!/usr/bin/env python3
"""P43: split the BF16 shared expert path."""
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

from benchmarks.p15_moe_packed_segment_profile import _bench, _diff, _prepare_layer_moe_input, _prefill  # noqa: E402
from benchmarks.p38_moe_multilayer_profile import BEST_R6000_ENV  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.moe_packed_nvfp4 import _apply_shared_expert_gate  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _shared_separate(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
    up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return shared


def _shared_fused(h_flat: torch.Tensor, w: dict) -> torch.Tensor:
    gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
    gate_s, up_s = gate_up_s.chunk(2, dim=-1)
    shared = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    return _apply_shared_expert_gate(h_flat, shared, w)


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
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    w = runner.layer_weights[layer_idx]
    if "mlp.shared_expert.gate_proj.weight" not in w:
        return {"layer": layer_idx, "layer_type": LAYER_TYPES[layer_idx], "skipped": True, "skip_reason": "no_shared_expert"}

    gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
    gate_s, up_s = gate_up_s.chunk(2, dim=-1)
    inter = F.silu(gate_s) * up_s
    down = F.linear(inter, w["mlp.shared_expert.down_proj.weight"])

    timings = {
        "separate_total_ms": _bench(lambda: _shared_separate(h_flat, w), warmup, iters),
        "fused_total_ms": _bench(lambda: _shared_fused(h_flat, w), warmup, iters),
        "fused_gate_up_ms": _bench(lambda: F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"]), warmup, iters),
        "down_ms": _bench(lambda: F.linear(inter, w["mlp.shared_expert.down_proj.weight"]), warmup, iters),
    }
    if "mlp.shared_expert_gate.weight" in w:
        timings["shared_gate_ms"] = _bench(
            lambda: torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])),
            warmup,
            iters,
        )
        timings["down_times_gate_ms"] = _bench(
            lambda: down * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"])),
            warmup,
            iters,
        )

    separate = _shared_separate(h_flat, w)
    fused = _shared_fused(h_flat, w)
    return {
        "layer": layer_idx,
        "layer_type": LAYER_TYPES[layer_idx],
        "timings_ms": timings,
        "fused_speedup_vs_separate": timings["separate_total_ms"] / timings["fused_total_ms"],
        "fused_vs_separate": _diff(fused, separate),
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
    summary = {
        "separate_total_ms_mean": _mean(rows, "separate_total_ms"),
        "fused_total_ms_mean": _mean(rows, "fused_total_ms"),
        "fused_gate_up_ms_mean": _mean(rows, "fused_gate_up_ms"),
        "down_ms_mean": _mean(rows, "down_ms"),
        "shared_gate_ms_mean": _mean(rows, "shared_gate_ms"),
        "sampled_layers": layers,
        "profiled_layer_count": sum(1 for row in rows if not row.get("skipped")),
    }
    if summary["separate_total_ms_mean"] and summary["fused_total_ms_mean"]:
        summary["fused_speedup_vs_separate_mean"] = summary["separate_total_ms_mean"] / summary["fused_total_ms_mean"]

    result = {
        "schema_version": "lynn-engine-p43-shared-expert-inner-profile-v1",
        "model": args.model,
        "decode_position": decode_position,
        "applied_env": applied_env,
        "shared_expert_gate_backend": os.environ.get("LYNN_SHARED_EXPERT_GATE_BACKEND", "torch"),
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
