#!/usr/bin/env python3
"""P200: Qwen3.5-9B dense FP4xFP8 FFN stage profiler.

P195 proves the FP4xFP8 FFN island is numerically usable but the resident path
only gives a small end-to-end lift.  This profiler splits the current decode
FFN implementation into the costs that matter for the next native boundary:

  gate quant -> gate MMA
  up quant   -> up MMA
  SiLU * up
  intermediate quant -> down MMA

The goal is to decide whether the next kernel should first remove duplicated
activation quantization, fuse gate/up projection launches, or fuse the native
intermediate/down boundary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p191_qwen35_9b_dense_fp4x_fp8_cute_probe import (  # noqa: E402
    _quantize_to_fp8_e4m3,
)


def _bench(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]], iters: int) -> tuple[Any, float]:
    out: Any = None
    for _ in range(max(1, min(10, iters))):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end) / iters)


def _project_prequant(
    ext: Any,
    side: dict[str, torch.Tensor],
    proj: str,
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
) -> torch.Tensor:
    w_packed = side[f"{proj}.weight_packed"]
    w_scale = side[f"{proj}.weight_scale"].float()
    w_global = side[f"{proj}.weight_global_scale"].float().view(-1)
    n, k_half = w_packed.shape
    k = k_half * 2
    return ext.dense_fp4xfp8_mma_scaled_probe(
        act_fp8.contiguous(),
        act_scale.contiguous(),
        w_packed.contiguous(),
        w_scale.contiguous(),
        w_global.contiguous(),
        1,
        n,
        k,
    )


def _project_gate_up_dual_prequant(
    ext: Any,
    side: dict[str, torch.Tensor],
    act_fp8: torch.Tensor,
    act_scale: torch.Tensor,
) -> torch.Tensor:
    gate_packed = side["gate_proj.weight_packed"]
    gate_scale = side["gate_proj.weight_scale"].float()
    gate_global = side["gate_proj.weight_global_scale"].float().view(-1)
    up_packed = side["up_proj.weight_packed"]
    up_scale = side["up_proj.weight_scale"].float()
    up_global = side["up_proj.weight_global_scale"].float().view(-1)
    n, k_half = gate_packed.shape
    k = k_half * 2
    return ext.dense_fp4xfp8_mma_scaled_dual_probe(
        act_fp8.contiguous(),
        act_scale.contiguous(),
        gate_packed.contiguous(),
        gate_scale.contiguous(),
        gate_global.contiguous(),
        up_packed.contiguous(),
        up_scale.contiguous(),
        up_global.contiguous(),
        1,
        n,
        k,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().view(-1)
    bf = b.float().view(-1)
    return float(torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    ))


def _profile_fixture(
    ext: Any,
    fixture: dict[str, torch.Tensor],
    side: dict[str, torch.Tensor],
    *,
    iters: int,
) -> dict[str, Any]:
    ffn_in = fixture["ffn_in"].to(torch.bfloat16).view(-1)
    bf16_ref = fixture["ffn_output"].to(torch.bfloat16).view(-1)

    (gate_act_fp8, gate_act_scale), gate_quant_ms = _bench(
        lambda: _quantize_to_fp8_e4m3(ffn_in),
        iters,
    )
    (up_act_fp8, up_act_scale), up_quant_ms = _bench(
        lambda: _quantize_to_fp8_e4m3(ffn_in),
        iters,
    )
    gate, gate_mma_ms = _bench(
        lambda: _project_prequant(ext, side, "gate_proj", gate_act_fp8, gate_act_scale),
        iters,
    )
    up, up_mma_ms = _bench(
        lambda: _project_prequant(ext, side, "up_proj", up_act_fp8, up_act_scale),
        iters,
    )
    dual_gate_up = None
    dual_gate_up_mma_ms = None
    dual_gate_vs_single_max_abs = None
    dual_up_vs_single_max_abs = None
    if hasattr(ext, "dense_fp4xfp8_mma_scaled_dual_probe"):
        # The dual path is allowed to share one quantized activation because
        # gate/up consume the same ffn_in tensor.
        dual_gate_up, dual_gate_up_mma_ms = _bench(
            lambda: _project_gate_up_dual_prequant(ext, side, gate_act_fp8, gate_act_scale),
            iters,
        )
        dual_gate_vs_single_max_abs = float((dual_gate_up[0].float() - gate.float()).abs().max())
        dual_up_vs_single_max_abs = float((dual_gate_up[1].float() - up.float()).abs().max())
    inter, silu_mul_ms = _bench(
        lambda: F.silu(gate.float()) * up.float(),
        iters,
    )
    (down_act_fp8, down_act_scale), inter_quant_ms = _bench(
        lambda: _quantize_to_fp8_e4m3(inter),
        iters,
    )
    down, down_mma_ms = _bench(
        lambda: _project_prequant(ext, side, "down_proj", down_act_fp8, down_act_scale),
        iters,
    )

    def current_ffn() -> torch.Tensor:
        gate_a, gate_s = _quantize_to_fp8_e4m3(ffn_in)
        gate_o = _project_prequant(ext, side, "gate_proj", gate_a, gate_s)
        up_a, up_s = _quantize_to_fp8_e4m3(ffn_in)
        up_o = _project_prequant(ext, side, "up_proj", up_a, up_s)
        inter_o = F.silu(gate_o.float()) * up_o.float()
        down_a, down_s = _quantize_to_fp8_e4m3(inter_o)
        return _project_prequant(ext, side, "down_proj", down_a, down_s)

    def shared_input_quant_ffn() -> torch.Tensor:
        act_a, act_s = _quantize_to_fp8_e4m3(ffn_in)
        gate_o = _project_prequant(ext, side, "gate_proj", act_a, act_s)
        up_o = _project_prequant(ext, side, "up_proj", act_a, act_s)
        inter_o = F.silu(gate_o.float()) * up_o.float()
        down_a, down_s = _quantize_to_fp8_e4m3(inter_o)
        return _project_prequant(ext, side, "down_proj", down_a, down_s)

    current_out, current_total_ms = _bench(current_ffn, iters)
    shared_out, shared_input_quant_total_ms = _bench(shared_input_quant_ffn, iters)

    diff = down.float().view(-1) - bf16_ref.float().view(-1)
    return {
        "gate_input_quant_ms": gate_quant_ms,
        "up_input_quant_ms": up_quant_ms,
        "gate_mma_ms": gate_mma_ms,
        "up_mma_ms": up_mma_ms,
        "dual_gate_up_mma_ms": dual_gate_up_mma_ms,
        "dual_gate_vs_single_max_abs": dual_gate_vs_single_max_abs,
        "dual_up_vs_single_max_abs": dual_up_vs_single_max_abs,
        "silu_mul_ms": silu_mul_ms,
        "intermediate_quant_ms": inter_quant_ms,
        "down_mma_ms": down_mma_ms,
        "current_total_ms": current_total_ms,
        "shared_input_quant_total_ms": shared_input_quant_total_ms,
        "estimated_stage_sum_ms": (
            gate_quant_ms + gate_mma_ms + up_quant_ms + up_mma_ms
            + silu_mul_ms + inter_quant_ms + down_mma_ms
        ),
        "estimated_shared_input_quant_sum_ms": (
            gate_quant_ms + gate_mma_ms + up_mma_ms + silu_mul_ms
            + inter_quant_ms + down_mma_ms
        ),
        "duplicated_input_quant_ms": up_quant_ms,
        "projection_mma_ms_sum": gate_mma_ms + up_mma_ms + down_mma_ms,
        "dual_projection_mma_ms_sum": (
            (dual_gate_up_mma_ms + down_mma_ms) if dual_gate_up_mma_ms is not None else None
        ),
        "torch_boundary_ms_sum": gate_quant_ms + up_quant_ms + silu_mul_ms + inter_quant_ms,
        "dual_gate_up_speedup_vs_two_launches": (
            (gate_mma_ms + up_mma_ms) / dual_gate_up_mma_ms
            if dual_gate_up_mma_ms and dual_gate_up_mma_ms > 0 else None
        ),
        "estimated_dual_gate_up_sum_ms": (
            gate_quant_ms + (dual_gate_up_mma_ms or 0.0) + silu_mul_ms + inter_quant_ms + down_mma_ms
            if dual_gate_up_mma_ms is not None else None
        ),
        "down_vs_bf16_max_abs": float(diff.abs().max()),
        "down_vs_bf16_rel_l2": float(
            torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(bf16_ref.float()).clamp_min(1e-12)
        ),
        "down_vs_bf16_cosine": _cosine(down, bf16_ref),
        "shared_matches_current": bool(torch.equal(shared_out.view(-1), current_out.view(-1))),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    sidecar_dir = Path(args.sidecar_dir)
    manifest_path = fixtures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if args.layers.strip().lower() in {"all", "*"}:
        layer_filter = None
    else:
        layer_filter = {int(x) for x in args.layers.split(",") if x.strip()}
    items = [
        item for item in manifest.get("fixtures", [])
        if layer_filter is None or int(item["layer_id"]) in layer_filter
    ]
    if args.max_fixtures > 0:
        items = items[:args.max_fixtures]
    if not items:
        raise RuntimeError(f"no fixtures selected from {fixtures_dir}")

    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "dense_fp4xfp8_mma_scaled_probe"):
        raise RuntimeError("dense_fp4xfp8_mma_scaled_probe missing from native extension")

    results: list[dict[str, Any]] = []
    for item in items:
        layer = int(item["layer_id"])
        fixture_path = fixtures_dir / item["file"]
        sidecar_path = sidecar_dir / f"layer_{layer:02d}.safetensors"
        fixture = load_file(str(fixture_path), device="cuda")
        side = load_file(str(sidecar_path), device="cuda")
        profile = _profile_fixture(ext, fixture, side, iters=args.iters)
        row = {
            "layer": layer,
            "fixture_file": fixture_path.name,
            **profile,
        }
        results.append(row)
        print(
            f"L{layer:02d} {fixture_path.name}: total={profile['current_total_ms']:.4f}ms "
            f"mma_sum={profile['projection_mma_ms_sum']:.4f}ms "
            f"torch_boundary={profile['torch_boundary_ms_sum']:.4f}ms "
            f"dup_quant={profile['duplicated_input_quant_ms']:.4f}ms"
        )

    summary_keys = [
        "gate_input_quant_ms",
        "up_input_quant_ms",
        "gate_mma_ms",
        "up_mma_ms",
        "silu_mul_ms",
        "intermediate_quant_ms",
        "down_mma_ms",
        "current_total_ms",
        "shared_input_quant_total_ms",
        "estimated_stage_sum_ms",
        "estimated_shared_input_quant_sum_ms",
        "duplicated_input_quant_ms",
        "projection_mma_ms_sum",
        "torch_boundary_ms_sum",
        "down_vs_bf16_rel_l2",
        "down_vs_bf16_cosine",
        "dual_gate_up_mma_ms",
        "dual_projection_mma_ms_sum",
        "dual_gate_up_speedup_vs_two_launches",
        "estimated_dual_gate_up_sum_ms",
    ]
    summary = {f"{key}_mean": _mean(results, key) for key in summary_keys}
    current = summary["current_total_ms_mean"] or 0.0
    shared = summary["shared_input_quant_total_ms_mean"] or 0.0
    if current > 0.0:
        summary["shared_input_quant_speedup_estimate"] = current / shared if shared > 0 else None
        summary["projection_mma_fraction"] = (summary["projection_mma_ms_sum_mean"] or 0.0) / current
        summary["torch_boundary_fraction"] = (summary["torch_boundary_ms_sum_mean"] or 0.0) / current
        dual_sum = summary.get("estimated_dual_gate_up_sum_ms_mean")
        if dual_sum:
            summary["estimated_dual_gate_up_stage_speedup"] = (
                (summary["estimated_stage_sum_ms_mean"] or current) / dual_sum
            )
    return {
        "schema": "lynn-qwen35-9b-p200-fp4xfp8-stage-profile-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures_dir": str(fixtures_dir),
        "sidecar_dir": str(sidecar_dir),
        "layers_arg": args.layers,
        "iters": args.iters,
        "fixtures_tested": len(results),
        "summary": summary,
        "decision_hints": {
            "duplicated_input_quant_is_worth_fusing": (
                (summary["duplicated_input_quant_ms_mean"] or 0.0) >= 0.03
            ),
            "torch_boundary_dominates": (
                (summary.get("torch_boundary_fraction") or 0.0) >= 0.35
            ),
            "projection_mma_dominates": (
                (summary.get("projection_mma_fraction") or 0.0) >= 0.50
            ),
        },
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--layers", default="0,8,16,24,31")
    ap.add_argument("--max-fixtures", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
