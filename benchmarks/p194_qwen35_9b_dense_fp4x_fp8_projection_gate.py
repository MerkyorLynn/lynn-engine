#!/usr/bin/env python3
"""P194: Qwen3.5-9B dense FFN FP4xFP8 projection gate.

P191 proves one gate projection can use R6000 E4M3xE2M1 MMA.  P194 extends the
same scaled-MMA contract to the full dense FFN projection set:

  gate_proj(ffn_in) -> gate_output
  up_proj(ffn_in) -> up_output
  down_proj(intermediate) -> ffn_output

All packed weights/scales are loaded from the P192 sidecar.  The gate blocks
resident promotion until scaled MMA matches the scalar scaled reference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p191_qwen35_9b_dense_fp4x_fp8_cute_probe import (  # noqa: E402
    _quantize_to_fp8_e4m3,
)

PROJ_SPECS: dict[str, tuple[str, str]] = {
    "gate_proj": ("ffn_in", "gate_output"),
    "up_proj": ("ffn_in", "up_output"),
    "down_proj": ("intermediate", "ffn_output"),
}


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


def _bench(fn, iters: int) -> float:
    for _ in range(10):
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


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    sidecar_dir = Path(args.sidecar_dir)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    projections = [x.strip() for x in args.projections.split(",") if x.strip()]
    for proj in projections:
        if proj not in PROJ_SPECS:
            raise ValueError(f"unknown projection {proj}; expected one of {sorted(PROJ_SPECS)}")

    from engine.native_cuda import load_lynn_native_extension

    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "dense_fp4xfp8_scalar_reference"):
        raise RuntimeError("dense_fp4xfp8_scalar_reference missing from native extension")
    if not hasattr(ext, "dense_fp4xfp8_mma_scaled_probe"):
        raise RuntimeError("dense_fp4xfp8_mma_scaled_probe missing from native extension")

    results: list[dict[str, Any]] = []
    for layer in layers:
        fixture_path = fixtures_dir / f"layer_{layer:02d}_prompt_00.safetensors"
        sidecar_path = sidecar_dir / f"layer_{layer:02d}.safetensors"
        if not fixture_path.exists():
            results.append({"layer": layer, "ok": False, "error": f"missing fixture {fixture_path}"})
            continue
        if not sidecar_path.exists():
            results.append({"layer": layer, "ok": False, "error": f"missing sidecar {sidecar_path}"})
            continue
        fixture = load_file(str(fixture_path), device="cuda")
        side = load_file(str(sidecar_path), device="cuda")

        for proj in projections:
            input_key, ref_key = PROJ_SPECS[proj]
            x = fixture[input_key].to(torch.bfloat16).view(-1)
            ref = fixture[ref_key].to(torch.bfloat16).view(-1)
            w_packed = side[f"{proj}.weight_packed"]
            w_scale = side[f"{proj}.weight_scale"].float()
            w_global = side[f"{proj}.weight_global_scale"].float().view(-1)
            n, k_half = w_packed.shape
            k = k_half * 2
            act_fp8, act_scale = _quantize_to_fp8_e4m3(x[:k])

            scalar = ext.dense_fp4xfp8_scalar_reference(
                act_fp8.contiguous(),
                act_scale.contiguous(),
                w_packed.contiguous(),
                w_scale.contiguous(),
                w_global.contiguous(),
            )
            scaled = ext.dense_fp4xfp8_mma_scaled_probe(
                act_fp8.contiguous(),
                act_scale.contiguous(),
                w_packed.contiguous(),
                w_scale.contiguous(),
                w_global.contiguous(),
                1,
                n,
                k,
            )

            scalar_ms = _bench(
                lambda: ext.dense_fp4xfp8_scalar_reference(
                    act_fp8, act_scale, w_packed, w_scale, w_global
                ),
                args.iters,
            )
            scaled_ms = _bench(
                lambda: ext.dense_fp4xfp8_mma_scaled_probe(
                    act_fp8, act_scale, w_packed, w_scale, w_global, 1, n, k
                ),
                args.iters,
            )

            scalar_ref = scalar[:n]
            ref_flat = ref[:n]
            scaled_flat = scaled[:n]
            scaled_diff = scaled_flat.float() - scalar_ref.float()
            row = {
                "layer": layer,
                "projection": proj,
                "shape": [int(n), int(k)],
                "scalar_vs_bf16_cosine": _cosine(scalar_ref, ref_flat),
                "scalar_vs_bf16_rel_l2": _rel_l2(scalar_ref, ref_flat),
                "scalar_vs_bf16_max_abs": float((scalar_ref.float() - ref_flat.float()).abs().max()),
                "scaled_vs_scalar_cosine": _cosine(scaled_flat, scalar_ref),
                "scaled_vs_scalar_rel_l2": _rel_l2(scaled_flat, scalar_ref),
                "scaled_vs_scalar_max_abs": float(scaled_diff.abs().max()),
                "scalar_ms": scalar_ms,
                "scaled_ms": scaled_ms,
                "speedup_vs_scalar": scalar_ms / scaled_ms if scaled_ms > 0 else None,
            }
            row["ok"] = (
                row["scaled_vs_scalar_cosine"] >= args.scaled_cosine_min
                and row["scaled_vs_scalar_rel_l2"] <= args.scaled_rel_l2_max
            )
            results.append(row)
            status = "GREEN" if row["ok"] else "RED"
            print(
                f"L{layer:02d} {proj}: {status} "
                f"scaled_rel={row['scaled_vs_scalar_rel_l2']:.3e} "
                f"scaled_ms={scaled_ms:.4f} speedup={row['speedup_vs_scalar']:.2f}x"
            )

    ok = all(bool(r.get("ok")) for r in results)
    return {
        "schema": "lynn-qwen35-9b-p194-fp4x-fp8-projection-gate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures_dir": str(fixtures_dir),
        "sidecar_dir": str(sidecar_dir),
        "layers": layers,
        "projections": projections,
        "thresholds": {
            "scaled_cosine_min": args.scaled_cosine_min,
            "scaled_rel_l2_max": args.scaled_rel_l2_max,
        },
        "results": results,
        "overall": "GREEN" if ok else "RED",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--layers", default="0,16")
    ap.add_argument("--projections", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--scaled-cosine-min", type=float, default=0.999999)
    ap.add_argument("--scaled-rel-l2-max", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = run_gate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Overall: {report['overall']}")
    print(f"Report: {out}")
    return 0 if report["overall"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
