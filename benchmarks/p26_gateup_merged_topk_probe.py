#!/usr/bin/env python3
"""P26: merged-top-k gate/up probe for packed NVFP4 MoE.

The production gate/up kernel launches one Triton program per
`(expert_slot, inter_block)`. This probe tests the opposite scheduling shape:
one program per `inter_block`, looping over all active top-k experts inside the
program. It keeps exactly the same per-16 scalar math contract as the production
kernel, so this is a scheduling experiment, not a quantization experiment.

Promotion criteria:
  - cosine >= 0.999999 against the production scalar packed gate/up reference;
  - latency faster than production gate/up on representative layers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p10e_packed_active_expert_probe import _load_grouped, _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_gate_up_silu,
    nvfp4_grouped_gate_up_silu_merged_topk,
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


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return {
        "max_abs": float((af - bf).abs().max().item()),
        "mean_abs": float((af - bf).abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(af - bf).item() / torch.linalg.vector_norm(af).item()),
        "cosine": float(F.cosine_similarity(af, bf, dim=0).item()),
    }


def _run_layer(
    runner: LynnIncrementalRunner,
    model_dir: Path,
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
    h_flat = h_moe.view(-1, h_moe.shape[-1])
    hidden = h_flat[0]
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    expert_ids = expert_indices[0].to(torch.int32).contiguous()
    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    def ref_fn() -> torch.Tensor:
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

    ref = ref_fn()
    rows = []
    for block_inter in (4, 8, 16):
        for block_hidden in (64, 128, 256):
            for num_warps in (4, 8):
                def cand(
                    block_inter=block_inter,
                    block_hidden=block_hidden,
                    num_warps=num_warps,
                ) -> torch.Tensor:
                    return nvfp4_grouped_gate_up_silu_merged_topk(
                        hidden,
                        expert_ids,
                        gate_up_packed,
                        gate_up_scale,
                        gate_up_global,
                        block_inter=block_inter,
                        block_hidden=block_hidden,
                        num_warps=num_warps,
                    )

                out = cand()
                rows.append({
                    "block_inter": block_inter,
                    "block_hidden": block_hidden,
                    "num_warps": num_warps,
                    "ms": _bench(cand, warmup, iters),
                    "diff_vs_reference": _diff(ref, out),
                })

    ref_ms = _bench(ref_fn, warmup, iters)
    best = sorted(rows, key=lambda r: r["ms"])[:5]
    passing = [
        r for r in sorted(rows, key=lambda r: r["ms"])
        if r["diff_vs_reference"]["cosine"] >= 0.999999
    ][:5]
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "reference_ms": ref_ms,
        "candidates": rows,
        "best_by_speed": best,
        "best_passing_cosine_999999": passing,
        "best_speedup_vs_reference": ref_ms / best[0]["ms"] if best else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[28])
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    model_dir = Path(args.model)
    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    cases = [
        _run_layer(
            runner,
            model_dir,
            layer=layer,
            prompt=args.prompt,
            warmup=args.warmup,
            iters=args.iters,
        )
        for layer in args.layers
    ]
    result = {
        "schema_version": "lynn-engine-p26-gateup-merged-topk-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "promote": all(
            case["best_by_speed"]
            and case["best_by_speed"][0]["ms"] < case["reference_ms"]
            and case["best_by_speed"][0]["diff_vs_reference"]["cosine"] >= 0.999999
            for case in cases
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

