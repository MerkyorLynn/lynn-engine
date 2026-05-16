#!/usr/bin/env python3
"""P60: fail-loud runtime guard for the future grouped per-16 backend.

This does not claim a speedup.  It creates the stable backend name and verifies
that `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16` cannot silently fall back to
the rejected scalar/tile bridges.
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

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    args = ap.parse_args()

    best_env = {
        "LYNN_MOE_IMPL": "packed_nvfp4",
        "LYNN_MOE_GATE_BLOCK_INTER": "8",
        "LYNN_MOE_GATE_BLOCK_HIDDEN": "256",
        "LYNN_MOE_DOWN_BLOCK_HIDDEN": "8",
        "LYNN_MOE_DOWN_BLOCK_INTER": "512",
        "LYNN_MOE_GATE_NUM_WARPS": "4",
        "LYNN_MOE_DOWN_NUM_WARPS": "8",
    }
    for key, value in best_env.items():
        os.environ.setdefault(key, value)
    os.environ["LYNN_MOE_FAST_FIXED"] = "0"

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])

    baseline = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
    baseline_norm = float(baseline.float().norm().item())

    grouped_error: dict[str, Any] | None = None
    old = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": "grouped_per16"})
    try:
        try:
            _ = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
        except Exception as exc:  # expected until the real kernel lands
            grouped_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
    finally:
        _restore_env(old)

    ok = (
        baseline_norm > 0
        and grouped_error is not None
        and grouped_error["type"] == "NotImplementedError"
        and "true grouped per-16 native-FP4" in grouped_error["message"]
    )
    result = {
        "schema_version": "lynn-engine-p60-grouped-per16-backend-guard-v1",
        "model": args.model,
        "layer": args.layer,
        "baseline_triton_norm": baseline_norm,
        "grouped_per16_error": grouped_error,
        "pass": ok,
        "decision": (
            "The grouped_per16 backend name is now reserved and fail-loud. "
            "The next implementation must replace this guard with the real "
            "grouped per-16 CUDA/CUTLASS kernel, not a scalar bridge."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
