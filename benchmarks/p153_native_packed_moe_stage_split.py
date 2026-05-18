#!/usr/bin/env python3
"""P153 · Split native packed-MoE drift into gate/up vs down stages.

P152 showed that the packed NVFP4 native slot MoE path is close to the
production Triton stage but not exact.  This harness runs the native gate/up
and down kernels separately against the P147 Triton-stage reference files so
the next kernel edit can target the real source of drift instead of guessing.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)
    max_abs = float(diff.abs().max())
    cosine = float(torch.dot(rf, cf) / (ref_norm * cand_norm))
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": cosine,
        "exact": 1 if max_abs == 0.0 else 0,
    }


def _bench_ms(fn, *, warmup: int, iters: int) -> float:
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


def _ref_file_for(reference_dir: Path, layer_id: int, prompt_id: int) -> Path:
    path = reference_dir / f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_triton_stage.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"P147 reference not found: {path}")
    return path


def _max_metric(rows: list[dict[str, Any]], key: str, metric: str) -> float | None:
    values = [r[key][metric] for r in rows if r.get(key) is not None]
    return max(values) if values else None


def _sum_exact(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(r[key]["exact"]) for r in rows if r.get(key) is not None)


def main() -> int:
    ap = argparse.ArgumentParser(description="Split native packed MoE drift by stage.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())

    ext = load_lynn_native_extension(verbose=False)
    missing = [
        name
        for name in (
            "moe_slot_packed_nvfp4_inter_probe",
            "moe_slot_packed_nvfp4_down_probe",
            "moe_slot_packed_nvfp4_probe",
        )
        if not hasattr(ext, name)
    ]
    if missing:
        raise RuntimeError(f"native extension lacks required P153 symbols: {missing}")

    print("[p153] Native packed MoE stage split")
    print(f"[p153] packed_fixtures={packed_dir}")
    print(f"[p153] p147_reference_dir={ref_dir}")

    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)
        ref = _load_fixture(_ref_file_for(ref_dir, layer_id, prompt_id), args.device)

        x = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        routing = data["routing_weights"].to(torch.float32).contiguous()
        gu_packed = data["slot_gate_up_packed"].contiguous()
        gu_scale = data["slot_gate_up_scale"].to(torch.float16).contiguous()
        gu_global = data["slot_gate_up_global_scale"].to(torch.float16).contiguous()
        down_packed = data["slot_down_packed"].contiguous()
        down_scale = data["slot_down_scale"].to(torch.float16).contiguous()
        down_global = data["slot_down_global_scale"].to(torch.float16).contiguous()

        triton_inter = ref["triton_inter"].to(torch.bfloat16).contiguous()
        triton_out = ref["routed_output"].to(torch.bfloat16).view(1, -1).contiguous()

        def native_inter_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_inter_probe(
                x,
                gu_packed,
                gu_scale,
                gu_global,
            )

        def native_down_from_triton_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_down_probe(
                triton_inter,
                routing,
                down_packed,
                down_scale,
                down_global,
            )

        def native_full_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_probe(
                x,
                routing,
                gu_packed,
                gu_scale,
                gu_global,
                down_packed,
                down_scale,
                down_global,
            )

        native_inter = native_inter_fn().contiguous()
        down_from_triton = native_down_from_triton_fn().view(1, -1).contiguous()
        native_full = native_full_fn().view(1, -1).contiguous()
        down_from_native_inter = ext.moe_slot_packed_nvfp4_down_probe(
            native_inter,
            routing,
            down_packed,
            down_scale,
            down_global,
        ).view(1, -1).contiguous()

        inter_ms = _bench_ms(native_inter_fn, warmup=args.warmup, iters=args.iters)
        down_ms = _bench_ms(native_down_from_triton_fn, warmup=args.warmup, iters=args.iters)
        full_ms = _bench_ms(native_full_fn, warmup=args.warmup, iters=args.iters)

        row = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "native_inter_ms": inter_ms,
            "native_down_ms_from_triton_inter": down_ms,
            "native_full_ms": full_ms,
            "inter_vs_triton": _metric(triton_inter, native_inter),
            "down_from_triton_inter_vs_triton_out": _metric(triton_out, down_from_triton),
            "down_from_native_inter_vs_triton_out": _metric(triton_out, down_from_native_inter),
            "native_full_vs_triton_out": _metric(triton_out, native_full),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"inter_abs={row['inter_vs_triton']['max_abs']:.2e} "
            f"down_abs={row['down_from_triton_inter_vs_triton_out']['max_abs']:.2e} "
            f"full_abs={row['native_full_vs_triton_out']['max_abs']:.2e} "
            f"ms={inter_ms:.4f}+{down_ms:.4f}/{full_ms:.4f}",
            flush=True,
        )

    total = len(rows)
    inter_exact = _sum_exact(rows, "inter_vs_triton")
    down_exact = _sum_exact(rows, "down_from_triton_inter_vs_triton_out")
    full_exact = _sum_exact(rows, "native_full_vs_triton_out")
    if inter_exact < total:
        diagnosis = "GATEUP_DRIFT"
    elif down_exact < total:
        diagnosis = "DOWN_DRIFT"
    elif full_exact < total:
        diagnosis = "FULL_COMPOSITION_DRIFT"
    else:
        diagnosis = "STAGE_EXACT"

    report = {
        "schema": "lynn-p153-native-packed-moe-stage-split-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir),
        "total": total,
        "diagnosis": diagnosis,
        "inter_exact": inter_exact,
        "down_from_triton_inter_exact": down_exact,
        "native_full_exact": full_exact,
        "native_inter_ms_mean": sum(r["native_inter_ms"] for r in rows) / total if total else None,
        "native_down_ms_mean_from_triton_inter": (
            sum(r["native_down_ms_from_triton_inter"] for r in rows) / total if total else None
        ),
        "native_full_ms_mean": sum(r["native_full_ms"] for r in rows) / total if total else None,
        "inter_max_abs_max": _max_metric(rows, "inter_vs_triton", "max_abs"),
        "down_from_triton_inter_max_abs_max": _max_metric(
            rows, "down_from_triton_inter_vs_triton_out", "max_abs"
        ),
        "down_from_native_inter_max_abs_max": _max_metric(
            rows, "down_from_native_inter_vs_triton_out", "max_abs"
        ),
        "native_full_max_abs_max": _max_metric(rows, "native_full_vs_triton_out", "max_abs"),
        "results": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[p153] diagnosis={diagnosis}")
    print(f"[p153] inter_exact={inter_exact}/{total} down_exact={down_exact}/{total} full_exact={full_exact}/{total}")
    print(f"[p153] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
