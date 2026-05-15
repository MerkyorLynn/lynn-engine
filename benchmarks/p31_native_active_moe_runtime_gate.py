#!/usr/bin/env python3
"""P31: runtime opt-in gate for native active-MoE CUDA scalar backend.

P30 proved the native active-MoE contract in isolation. P31 wires that backend
behind `LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar` in the production
`moe_forward_decode_packed_nvfp4` path and verifies it matches the default
Triton backend on representative layers.
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

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _diff(ref: torch.Tensor, out: torch.Tensor) -> dict:
    rf = ref.float().reshape(-1)
    of = out.float().reshape(-1)
    delta = of - rf
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(delta).item() / torch.linalg.vector_norm(rf).item()),
        "cosine": float(F.cosine_similarity(rf, of, dim=0).item()),
    }


def _run_layer(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompt: str,
    warmup: int,
    iters: int,
) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])

    def run_with_backend(backend: str) -> torch.Tensor:
        old = os.environ.get("LYNN_NATIVE_ACTIVE_MOE_BACKEND")
        os.environ["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = backend
        try:
            return moe_forward_decode_packed_nvfp4(h_moe, w, cfg)
        finally:
            if old is None:
                os.environ.pop("LYNN_NATIVE_ACTIVE_MOE_BACKEND", None)
            else:
                os.environ["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = old

    def triton_backend() -> torch.Tensor:
        return run_with_backend("triton")

    def cuda_scalar_backend() -> torch.Tensor:
        return run_with_backend("cuda_scalar")

    ref = triton_backend()
    out = cuda_scalar_backend()
    return {
        "layer": layer,
        "diff_vs_triton_backend": _diff(ref, out),
        "triton_backend_ms": _bench(triton_backend, warmup, iters),
        "cuda_scalar_backend_ms": _bench(cuda_scalar_backend, warmup, iters),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[28])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(runner, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    for case in cases:
        case["cuda_vs_triton_speedup"] = case["triton_backend_ms"] / case["cuda_scalar_backend_ms"]
    promote = all(case["diff_vs_triton_backend"]["cosine"] >= 0.999999 for case in cases)
    result = {
        "schema_version": "lynn-engine-p31-native-active-moe-runtime-gate-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "pass": promote,
        "promote_default": False,
        "notes": [
            "cuda_scalar is an opt-in runtime backend and intentionally not the default.",
            "This gate proves the production MoE function can dispatch through Lynn native CUDA without numeric drift.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if promote else 1


if __name__ == "__main__":
    raise SystemExit(main())
