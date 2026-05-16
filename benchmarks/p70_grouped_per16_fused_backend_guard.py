#!/usr/bin/env python3
"""P70: fail-loud guard for the true fused grouped per-16 backend.

`grouped_per16` was reserved in P60. P68 then showed a two-stage tiled
reference is only a kernel signal. P70 creates a more explicit runtime name for
the real target:

  LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_fused

This backend must pass shape/layout checks and then fail loudly until the
one-boundary CUDA/CUTLASS kernel lands.
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
from benchmarks.p37_moe_config_generate_gate import BASE_ENV  # noqa: E402
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

    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["LYNN_MOE_FAST_FIXED"] = "0"
    os.environ.setdefault("LYNN_NATIVE_CUDA_BUILD_DIR", "/tmp/lynn_engine_native_build/p70_grouped_fused_guard")

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    h_layer, _ = _prefill_to_layer_input(runner, args.layer, args.prompt)
    w = runner.layer_weights[args.layer]
    cfg = runner.layer_cfgs[args.layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])

    baseline = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
    baseline_norm = float(baseline.float().norm().item())

    fused_error: dict[str, Any] | None = None
    old = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": "grouped_per16_fused"})
    try:
        try:
            _ = moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
        except Exception as exc:  # expected until the real fused kernel lands
            fused_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
    finally:
        _restore_env(old)

    ok = (
        baseline_norm > 0
        and fused_error is not None
        # Native extension TORCH_CHECK failures surface as RuntimeError while
        # Python fail-loud guards surface as NotImplementedError. The message
        # is the contract; the exception wrapper is a PyTorch implementation
        # detail.
        and fused_error["type"] in {"RuntimeError", "NotImplementedError"}
        and "fused grouped per-16" in fused_error["message"]
        and "two-stage intermediate tensor" in fused_error["message"]
    )
    result = {
        "schema_version": "lynn-engine-p70-grouped-per16-fused-backend-guard-v1",
        "model": args.model,
        "layer": args.layer,
        "baseline_triton_norm": baseline_norm,
        "grouped_per16_fused_error": fused_error,
        "pass": ok,
        "decision": (
            "The grouped_per16_fused backend name is now reserved for the true "
            "one-boundary active expert kernel. It must not fall back to P68's "
            "two-stage tile reference path."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
