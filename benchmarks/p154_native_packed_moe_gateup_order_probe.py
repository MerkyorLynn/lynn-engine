#!/usr/bin/env python3
"""P154 · Probe Triton-like hidden-block reduction for native packed gate/up."""
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


def _sum_exact(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(r[key]["exact"]) for r in rows)


def _max_abs(rows: list[dict[str, Any]], key: str) -> float:
    return max(float(r[key]["max_abs"]) for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe native packed MoE gate/up reduction order.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())
    ext = load_lynn_native_extension(verbose=False)
    for name in (
        "moe_slot_packed_nvfp4_inter_probe",
        "moe_slot_packed_nvfp4_inter_triton_order_probe",
        "moe_slot_packed_nvfp4_down_probe",
    ):
        if not hasattr(ext, name):
            raise RuntimeError(f"native extension lacks {name}")

    print("[p154] Native packed MoE gate/up reduction-order probe")
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

        def base_inter_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_inter_probe(x, gu_packed, gu_scale, gu_global)

        def order_inter_fn() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_inter_triton_order_probe(x, gu_packed, gu_scale, gu_global)

        base_inter = base_inter_fn().contiguous()
        order_inter = order_inter_fn().contiguous()
        order_out = ext.moe_slot_packed_nvfp4_down_probe(
            order_inter, routing, down_packed, down_scale, down_global
        ).view(1, -1).contiguous()

        row = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "base_inter_ms": _bench_ms(base_inter_fn, warmup=args.warmup, iters=args.iters),
            "triton_order_inter_ms": _bench_ms(order_inter_fn, warmup=args.warmup, iters=args.iters),
            "base_inter_vs_triton": _metric(triton_inter, base_inter),
            "triton_order_inter_vs_triton": _metric(triton_inter, order_inter),
            "triton_order_full_vs_triton_out": _metric(triton_out, order_out),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"base={row['base_inter_vs_triton']['max_abs']:.2e} "
            f"order={row['triton_order_inter_vs_triton']['max_abs']:.2e} "
            f"full={row['triton_order_full_vs_triton_out']['max_abs']:.2e}",
            flush=True,
        )

    total = len(rows)
    report = {
        "schema": "lynn-p154-native-packed-moe-gateup-order-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir),
        "total": total,
        "base_inter_exact": _sum_exact(rows, "base_inter_vs_triton"),
        "triton_order_inter_exact": _sum_exact(rows, "triton_order_inter_vs_triton"),
        "triton_order_full_exact": _sum_exact(rows, "triton_order_full_vs_triton_out"),
        "base_inter_max_abs_max": _max_abs(rows, "base_inter_vs_triton"),
        "triton_order_inter_max_abs_max": _max_abs(rows, "triton_order_inter_vs_triton"),
        "triton_order_full_max_abs_max": _max_abs(rows, "triton_order_full_vs_triton_out"),
        "base_inter_ms_mean": sum(r["base_inter_ms"] for r in rows) / total if total else None,
        "triton_order_inter_ms_mean": sum(r["triton_order_inter_ms"] for r in rows) / total if total else None,
        "diagnosis": "TRITON_ORDER_FIXED_GATEUP"
        if _sum_exact(rows, "triton_order_inter_vs_triton") == total
        else "TRITON_ORDER_STILL_DRIFTS",
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        "[p154] base_exact="
        f"{report['base_inter_exact']}/{total} order_exact={report['triton_order_inter_exact']}/{total} "
        f"order_full_exact={report['triton_order_full_exact']}/{total}"
    )
    print(f"[p154] diagnosis={report['diagnosis']}")
    print(f"[p154] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
