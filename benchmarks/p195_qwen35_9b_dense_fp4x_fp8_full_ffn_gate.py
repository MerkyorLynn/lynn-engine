#!/usr/bin/env python3
"""P195: Qwen3.5-9B dense FFN full FP4xFP8 composition gate.

This is the last fixture gate before resident wiring.  It composes the P194
scaled-MMA projections into the real dense FFN:

  down_proj(silu(gate_proj(ffn_in)) * up_proj(ffn_in))

The scalar path uses the same FP8 activations and NVFP4 weights/scales as the
MMA path, so scaled-vs-scalar measures kernel correctness.  The comparison to
fixture `ffn_output` measures the expected W4A8-style quality drift.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p191_qwen35_9b_dense_fp4x_fp8_cute_probe import (  # noqa: E402
    _quantize_to_fp8_e4m3,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().view(-1)
    bf = b.float().view(-1)
    return float(torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    ))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.float().view(-1) - b.float().view(-1)
    return float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b.float().view(-1)).clamp_min(1e-12))


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = a.float().view(-1) - b.float().view(-1)
    return {
        "max_abs": float(diff.abs().max()),
        "rel_l2": _rel_l2(a, b),
        "cosine": _cosine(a, b),
    }


def _bench(fn: Callable[[], torch.Tensor], iters: int) -> tuple[torch.Tensor, float]:
    out = fn()
    for _ in range(5):
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


def _project(ext, side: dict[str, torch.Tensor], proj: str, x: torch.Tensor, *, mma: bool) -> torch.Tensor:
    w_packed = side[f"{proj}.weight_packed"]
    w_scale = side[f"{proj}.weight_scale"].float()
    w_global = side[f"{proj}.weight_global_scale"].float().view(-1)
    n, k_half = w_packed.shape
    k = k_half * 2
    act_fp8, act_scale = _quantize_to_fp8_e4m3(x.view(-1)[:k])
    if mma:
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
    return ext.dense_fp4xfp8_scalar_reference(
        act_fp8.contiguous(),
        act_scale.contiguous(),
        w_packed.contiguous(),
        w_scale.contiguous(),
        w_global.contiguous(),
    )


def _ffn(ext, side: dict[str, torch.Tensor], ffn_in: torch.Tensor, *, mma: bool) -> torch.Tensor:
    gate = _project(ext, side, "gate_proj", ffn_in, mma=mma)
    up = _project(ext, side, "up_proj", ffn_in, mma=mma)
    inter = F.silu(gate.float()) * up.float()
    return _project(ext, side, "down_proj", inter, mma=mma)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    sidecar_dir = Path(args.sidecar_dir)
    candidate_dir = Path(args.candidate_output_dir) if args.candidate_output_dir else None
    if candidate_dir:
        candidate_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixtures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if args.layers.strip().lower() in {"all", "*"}:
        layer_filter = None
    else:
        layer_filter = {int(x) for x in args.layers.split(",") if x.strip()}
    fixture_items: list[dict[str, Any]]
    if manifest is not None:
        fixture_items = [
            item for item in manifest.get("fixtures", [])
            if layer_filter is None or int(item["layer_id"]) in layer_filter
        ]
    else:
        fixture_items = []
        for layer in sorted(layer_filter or []):
            fixture_items.append({"layer_id": layer, "file": f"layer_{layer:02d}_prompt_00.safetensors"})

    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "dense_fp4xfp8_scalar_reference"):
        raise RuntimeError("dense_fp4xfp8_scalar_reference missing from native extension")
    if not hasattr(ext, "dense_fp4xfp8_mma_scaled_probe"):
        raise RuntimeError("dense_fp4xfp8_mma_scaled_probe missing from native extension")

    results: list[dict[str, Any]] = []
    layers_seen: list[int] = []
    for item in fixture_items:
        layer = int(item["layer_id"])
        if layer not in layers_seen:
            layers_seen.append(layer)
        fixture_path = fixtures_dir / item["file"]
        sidecar_path = sidecar_dir / f"layer_{layer:02d}.safetensors"
        fixture = load_file(str(fixture_path), device="cuda")
        side = load_file(str(sidecar_path), device="cuda")
        ffn_in = fixture["ffn_in"].to(torch.bfloat16)
        bf16_ref = fixture["ffn_output"].to(torch.bfloat16).view(-1)

        scalar_out, scalar_ms = _bench(lambda: _ffn(ext, side, ffn_in, mma=False), args.iters)
        mma_out, mma_ms = _bench(lambda: _ffn(ext, side, ffn_in, mma=True), args.iters)
        scalar_flat = scalar_out.view(-1)
        mma_flat = mma_out.view(-1)
        scaled_vs_scalar = _metrics(mma_flat, scalar_flat)
        scaled_vs_bf16 = _metrics(mma_flat, bf16_ref)
        scalar_vs_bf16 = _metrics(scalar_flat, bf16_ref)

        row = {
            "layer": layer,
            "fixture_file": fixture_path.name,
            "scalar_ms": scalar_ms,
            "scaled_mma_ms": mma_ms,
            "speedup_vs_scalar": scalar_ms / mma_ms if mma_ms > 0 else None,
            "scaled_vs_scalar": scaled_vs_scalar,
            "scaled_vs_bf16": scaled_vs_bf16,
            "scalar_vs_bf16": scalar_vs_bf16,
        }
        row["ok"] = (
            scaled_vs_scalar["cosine"] >= args.scaled_cosine_min
            and scaled_vs_scalar["rel_l2"] <= args.scaled_rel_l2_max
        )
        row["numeric_ok"] = (
            scaled_vs_scalar["cosine"] >= args.scaled_cosine_min
            and scaled_vs_scalar["rel_l2"] <= args.amber_scaled_rel_l2_max
            and scaled_vs_scalar["max_abs"] <= args.amber_scaled_max_abs
            and abs(scaled_vs_bf16["rel_l2"] - scalar_vs_bf16["rel_l2"]) <= args.amber_bf16_rel_l2_delta_max
            and abs(scaled_vs_bf16["cosine"] - scalar_vs_bf16["cosine"]) <= args.amber_bf16_cosine_delta_max
        )
        results.append(row)
        status = "GREEN" if row["ok"] else ("AMBER" if row["numeric_ok"] else "RED")
        print(
            f"L{layer:02d}: {status} full_ffn scaled_rel={scaled_vs_scalar['rel_l2']:.3e} "
            f"scaled_ms={mma_ms:.4f} speedup={row['speedup_vs_scalar']:.2f}x "
            f"vs_bf16_cos={scaled_vs_bf16['cosine']:.6f}"
        )

        if candidate_dir:
            save_file(
                {
                    "ffn_output": mma_flat.to(torch.bfloat16).view(1, -1).cpu(),
                    "scalar_output": scalar_flat.to(torch.bfloat16).view(1, -1).cpu(),
                },
                str(candidate_dir / fixture_path.name),
            )

    strict_ok = all(bool(r["ok"]) for r in results)
    numeric_ok = all(bool(r["numeric_ok"]) for r in results)
    strict_passed = sum(1 for r in results if r["ok"])
    numeric_passed = sum(1 for r in results if r["numeric_ok"])
    if strict_ok:
        overall = "GREEN_STRICT"
    elif numeric_ok:
        overall = "AMBER_NUMERIC"
    else:
        overall = "RED"
    scaled_ms_values = [float(r["scaled_mma_ms"]) for r in results]
    speedup_values = [float(r["speedup_vs_scalar"]) for r in results]
    rel_values = [float(r["scaled_vs_scalar"]["rel_l2"]) for r in results]
    max_abs_values = [float(r["scaled_vs_scalar"]["max_abs"]) for r in results]
    cosine_values = [float(r["scaled_vs_scalar"]["cosine"]) for r in results]
    bf16_rel_delta_values = [
        abs(float(r["scaled_vs_bf16"]["rel_l2"]) - float(r["scalar_vs_bf16"]["rel_l2"]))
        for r in results
    ]
    bf16_cosine_delta_values = [
        abs(float(r["scaled_vs_bf16"]["cosine"]) - float(r["scalar_vs_bf16"]["cosine"]))
        for r in results
    ]
    return {
        "schema": "lynn-qwen35-9b-p195-fp4x-fp8-full-ffn-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures_dir": str(fixtures_dir),
        "sidecar_dir": str(sidecar_dir),
        "candidate_output_dir": str(candidate_dir) if candidate_dir else None,
        "layers": layers_seen,
        "fixtures_tested": len(fixture_items),
        "thresholds": {
            "scaled_cosine_min": args.scaled_cosine_min,
            "scaled_rel_l2_max": args.scaled_rel_l2_max,
            "amber_scaled_rel_l2_max": args.amber_scaled_rel_l2_max,
            "amber_scaled_max_abs": args.amber_scaled_max_abs,
            "amber_bf16_rel_l2_delta_max": args.amber_bf16_rel_l2_delta_max,
            "amber_bf16_cosine_delta_max": args.amber_bf16_cosine_delta_max,
        },
        "summary": {
            "strict_passed": strict_passed,
            "numeric_passed": numeric_passed,
            "total": len(results),
            "scaled_mma_ms_mean": sum(scaled_ms_values) / len(scaled_ms_values) if scaled_ms_values else None,
            "speedup_vs_scalar_mean": sum(speedup_values) / len(speedup_values) if speedup_values else None,
            "scaled_vs_scalar_rel_l2_max": max(rel_values) if rel_values else None,
            "scaled_vs_scalar_max_abs_max": max(max_abs_values) if max_abs_values else None,
            "scaled_vs_scalar_cosine_min": min(cosine_values) if cosine_values else None,
            "bf16_rel_l2_delta_max": max(bf16_rel_delta_values) if bf16_rel_delta_values else None,
            "bf16_cosine_delta_max": max(bf16_cosine_delta_values) if bf16_cosine_delta_values else None,
        },
        "results": results,
        "overall": overall,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--layers", default="0,16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--scaled-cosine-min", type=float, default=0.999999)
    ap.add_argument("--scaled-rel-l2-max", type=float, default=1e-5)
    ap.add_argument("--amber-scaled-rel-l2-max", type=float, default=3e-4)
    ap.add_argument("--amber-scaled-max-abs", type=float, default=5e-3)
    ap.add_argument("--amber-bf16-rel-l2-delta-max", type=float, default=2e-5)
    ap.add_argument("--amber-bf16-cosine-delta-max", type=float, default=5e-7)
    ap.add_argument("--candidate-output-dir", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = run_gate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Overall: {report['overall']}")
    print(f"Report: {out}")
    return 0 if report["overall"] in {"GREEN_STRICT", "AMBER_NUMERIC"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
