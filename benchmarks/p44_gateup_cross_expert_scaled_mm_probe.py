#!/usr/bin/env python3
"""P44: native FP4 cross-expert gate/up probe.

This is a deliberately shaped *probe*, not a production path.  The current
Triton gate/up kernel computes only the 8 active expert blocks.  A plain
`torch._scaled_mm` cannot express that block diagonal directly, so this probe
does the wasteful version:

  A: [top_k, hidden]          repeated activation rows
  B: [top_k * 1024, hidden]   selected gate/up rows for all active experts

The matmul computes every cross expert pair, then we keep the diagonal
1024-wide block for each row.  That pays roughly top_k extra work, but it tells
us whether Blackwell native FP4 tensor cores are strong enough to justify
writing the real grouped/block-diagonal kernel.
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
from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_linear import quantize_fp4_m1_native  # noqa: E402
from triton_kernels.nvfp4_moe import nvfp4_grouped_gate_up_silu  # noqa: E402


def _repeat_single_row_native_quant(hidden: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one decode hidden row, then repeat it for cross-expert matmul.

    The existing runtime quantizer is intentionally M=1.  Decode MoE uses the
    same activation row for every active expert, so repeating the packed row is
    exactly the shape this probe needs.
    """
    packed_one, _ = quantize_fp4_m1_native(hidden.reshape(1, -1))
    k = hidden.numel()
    groups = k // 16
    compact = (hidden.float().reshape(1, groups, 16).abs().amax(dim=-1) / 6.0).clamp_min(1.0e-8)
    compact = compact.expand(top_k, -1).contiguous()
    scale_a = _compact_scale_to_swizzled_fp8(compact, outer_dim=top_k, k=k)
    return packed_one.expand(top_k, -1).contiguous(), scale_a


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


def _diagonal_gateup_blocks(mm_out: torch.Tensor, *, top_k: int, gate_up_rows: int) -> torch.Tensor:
    pieces = []
    for slot in range(top_k):
        start = slot * gate_up_rows
        pieces.append(mm_out[slot, start : start + gate_up_rows])
    return torch.stack(pieces, dim=0)


def _run_layer(
    runner: LynnIncrementalRunner,
    model_dir: Path,
    *,
    layer: int,
    prompt: str,
    warmup: int,
    iters: int,
) -> dict:
    if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("P44 requires torch.float4_e2m1fn_x2 and torch._scaled_mm")

    h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
    h_flat = h_moe.reshape(-1, h_moe.shape[-1])
    hidden = h_flat[0].contiguous()
    top_k = int(cfg["num_experts_per_tok"])
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    _, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    expert_ids_i64 = expert_indices[0].to(torch.long).contiguous()
    expert_ids_i32 = expert_indices[0].to(torch.int32).contiguous()

    gate_up_packed, gate_up_scale, gate_up_global = _load_grouped(
        model_dir,
        f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj",
        runner.device,
    )

    def reference() -> torch.Tensor:
        return nvfp4_grouped_gate_up_silu(
            hidden,
            expert_ids_i32,
            gate_up_packed,
            gate_up_scale,
            gate_up_global,
            block_inter=8,
            block_hidden=256,
            num_warps=4,
        )

    selected_packed = gate_up_packed.index_select(0, expert_ids_i64).reshape(top_k * 1024, 1024).contiguous()
    selected_scale = gate_up_scale.index_select(0, expert_ids_i64).reshape(top_k * 1024, 128).contiguous()
    effective_scale = selected_scale.float() / gate_up_global.to(selected_scale.device).float()
    scale_b = _compact_scale_to_swizzled_fp8(effective_scale, outer_dim=top_k * 1024, k=2048)
    weight_t = selected_packed.view(torch.float4_e2m1fn_x2).t()
    x_rows = hidden.reshape(1, -1).expand(top_k, -1).contiguous()
    act_packed_once, scale_a_once = _repeat_single_row_native_quant(hidden, top_k)

    def native_mm_only() -> torch.Tensor:
        mm = torch._scaled_mm(
            act_packed_once.view(torch.float4_e2m1fn_x2),
            weight_t,
            scale_a=scale_a_once,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )
        gate_up = _diagonal_gateup_blocks(mm, top_k=top_k, gate_up_rows=1024)
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

    def native_quant_plus_mm() -> torch.Tensor:
        act_packed, scale_a = _repeat_single_row_native_quant(hidden, top_k)
        mm = torch._scaled_mm(
            act_packed.view(torch.float4_e2m1fn_x2),
            weight_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float16,
        )
        gate_up = _diagonal_gateup_blocks(mm, top_k=top_k, gate_up_rows=1024)
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

    ref = reference()
    native = native_quant_plus_mm()
    timings = {
        "reference_triton_gateup_ms": _bench(reference, warmup, iters),
        "native_cross_mm_only_ms": _bench(native_mm_only, warmup, iters),
        "native_cross_quant_plus_mm_ms": _bench(native_quant_plus_mm, warmup, iters),
        "activation_quant_repeat_topk_rows_ms": _bench(
            lambda: _repeat_single_row_native_quant(hidden, top_k)[0],
            warmup,
            iters,
        ),
    }
    timings["native_mm_only_speedup_vs_reference"] = timings["reference_triton_gateup_ms"] / timings["native_cross_mm_only_ms"]
    timings["native_quant_plus_mm_speedup_vs_reference"] = (
        timings["reference_triton_gateup_ms"] / timings["native_cross_quant_plus_mm_ms"]
    )
    return {
        "layer": layer,
        "expert_ids": [int(x) for x in expert_ids_i64.tolist()],
        "matrix_shapes": {
            "activation_rows": list(x_rows.shape),
            "selected_packed": list(selected_packed.shape),
            "scale_b": list(scale_b.shape),
            "dense_mm_output": [top_k, top_k * 1024],
            "kept_diagonal": [top_k, 1024],
            "wasted_cross_expert_factor": top_k,
        },
        "timings_ms": timings,
        "diff_vs_reference": _diff(ref, native),
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
        _run_layer(runner, model_dir, layer=layer, prompt=args.prompt, warmup=args.warmup, iters=args.iters)
        for layer in args.layers
    ]
    result = {
        "schema_version": "lynn-engine-p44-gateup-cross-expert-scaled-mm-probe-v1",
        "model": args.model,
        "layers": args.layers,
        "cases": cases,
        "summary": {
            "mean_reference_triton_gateup_ms": sum(c["timings_ms"]["reference_triton_gateup_ms"] for c in cases)
            / len(cases),
            "mean_native_mm_only_ms": sum(c["timings_ms"]["native_cross_mm_only_ms"] for c in cases) / len(cases),
            "mean_native_quant_plus_mm_ms": sum(c["timings_ms"]["native_cross_quant_plus_mm_ms"] for c in cases)
            / len(cases),
            "mean_native_mm_only_speedup": sum(c["timings_ms"]["native_mm_only_speedup_vs_reference"] for c in cases)
            / len(cases),
            "mean_native_quant_plus_mm_speedup": sum(
                c["timings_ms"]["native_quant_plus_mm_speedup_vs_reference"] for c in cases
            )
            / len(cases),
            "min_cosine": min(c["diff_vs_reference"]["cosine"] for c in cases),
        },
        "interpretation": [
            "This computes top_k cross expert blocks and keeps only the diagonal, so a real grouped kernel has up to top_k headroom over this probe.",
            "If this is already competitive, native FP4 tensor-core math is worth a dedicated grouped/block-diagonal kernel.",
            "If this is slower, torch._scaled_mm composition is not enough and P45 must be a custom grouped FP4 kernel.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
