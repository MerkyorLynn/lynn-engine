#!/usr/bin/env python3
"""P53-LUT: lightweight E2M1 decode-expression probe."""
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
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_fast_decode,
)


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


def _run_layer(runner: LynnIncrementalRunner, *, layer: int, prompt: str, warmup: int, iters: int) -> dict:
    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, int(cfg["num_experts_per_tok"]), dim=-1, sorted=False)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    gate_up_packed = w["mlp.experts._gate_up_packed"]
    gate_up_scale = w["mlp.experts._gate_up_scale"]
    gate_up_global = w["mlp.experts._gate_up_global_scale"]

    def ref() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    def fast() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu_fast_decode(
            hidden,
            expert_ids,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    ref_out = ref()
    fast_out = fast()
    ref_ms = _bench(ref, warmup, iters)
    fast_ms = _bench(fast, warmup, iters)
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_indices[0].tolist()],
        "diff_fast_vs_ref": _diff(ref_out, fast_out),
        "ref_ms": ref_ms,
        "fast_ms": fast_ms,
        "speedup": ref_ms / fast_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28, 36])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [_run_layer(runner, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters) for layer in args.layers]
    result = {
        "schema_version": "lynn-engine-p53-lut-fast-decode-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_ref_ms": sum(c["ref_ms"] for c in cases) / len(cases),
            "mean_fast_ms": sum(c["fast_ms"] for c in cases) / len(cases),
            "mean_speedup": sum(c["speedup"] for c in cases) / len(cases),
            "min_cosine": min(c["diff_fast_vs_ref"]["cosine"] for c in cases),
            "max_rel_l2": max(c["diff_fast_vs_ref"]["rel_l2"] for c in cases),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
