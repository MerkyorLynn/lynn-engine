#!/usr/bin/env python3
"""P157 · Correct Triton active-MoE stage timing.

P147's reference output is correct, but its ``gateup_ms`` timing calls a helper
that computes both gate/up and down before returning the intermediate.  This
probe measures the exact production Triton stage with separate gate/up, down,
and combined timings so Native MoE work is compared against the right baseline.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
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
    max_abs = float(diff.abs().max())
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / ref_norm),
        "cosine": float(torch.dot(rf, cf) / (ref_norm * cand_norm)),
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Correctly time Triton active-MoE stage pieces.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--gate-num-warps", type=int, default=4)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--down-num-warps", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from triton_kernels.nvfp4_moe import (
        nvfp4_grouped_down_weighted_sum,
        nvfp4_grouped_gate_up_silu_fast_decode,
    )

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())

    rows: list[dict[str, Any]] = []
    print("[p157] Correct Triton active-MoE stage timing")
    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)
        ref = _load_fixture(_ref_file_for(ref_dir, layer_id, prompt_id), args.device)

        hidden = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        top_k = int(data["slot_gate_up_packed"].shape[0])
        slot_ids = torch.arange(top_k, device=hidden.device, dtype=torch.int32)
        routing_weights = data["routing_weights"].to(torch.float32).contiguous()

        def gate_fn() -> torch.Tensor:
            return nvfp4_grouped_gate_up_silu_fast_decode(
                hidden,
                slot_ids,
                data["slot_gate_up_packed"].contiguous(),
                data["slot_gate_up_scale"].contiguous(),
                data["slot_gate_up_global_scale"].to(hidden.device).contiguous(),
                block_inter=args.gate_block_inter,
                block_hidden=args.gate_block_hidden,
                num_warps=args.gate_num_warps,
            )

        inter = gate_fn().contiguous()

        def down_fn() -> torch.Tensor:
            return nvfp4_grouped_down_weighted_sum(
                inter,
                slot_ids,
                routing_weights,
                data["slot_down_packed"].contiguous(),
                data["slot_down_scale"].contiguous(),
                data["slot_down_global_scale"].to(hidden.device).contiguous(),
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
            )

        def combined_fn() -> torch.Tensor:
            inter_local = gate_fn()
            return nvfp4_grouped_down_weighted_sum(
                inter_local,
                slot_ids,
                routing_weights,
                data["slot_down_packed"].contiguous(),
                data["slot_down_scale"].contiguous(),
                data["slot_down_global_scale"].to(hidden.device).contiguous(),
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
            )

        out = down_fn().view(1, -1).contiguous()
        row = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "gateup_ms": _bench_ms(gate_fn, warmup=args.warmup, iters=args.iters),
            "down_ms": _bench_ms(down_fn, warmup=args.warmup, iters=args.iters),
            "combined_ms": _bench_ms(combined_fn, warmup=args.warmup, iters=args.iters),
            "inter_vs_p147": _metric(ref["triton_inter"].to(torch.bfloat16), inter),
            "out_vs_p147": _metric(ref["routed_output"].to(torch.bfloat16).view(1, -1), out),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"gate={row['gateup_ms']:.4f} down={row['down_ms']:.4f} "
            f"combined={row['combined_ms']:.4f} "
            f"exact=({row['inter_vs_p147']['exact']},{row['out_vs_p147']['exact']})",
            flush=True,
        )

    total = len(rows)
    report = {
        "schema": "lynn-p157-triton-moe-stage-timing-correction-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir),
        "total": total,
        "inter_exact": sum(int(r["inter_vs_p147"]["exact"]) for r in rows),
        "out_exact": sum(int(r["out_vs_p147"]["exact"]) for r in rows),
        "gateup_ms_mean": statistics.mean(float(r["gateup_ms"]) for r in rows) if rows else None,
        "down_ms_mean": statistics.mean(float(r["down_ms"]) for r in rows) if rows else None,
        "combined_ms_mean": statistics.mean(float(r["combined_ms"]) for r in rows) if rows else None,
        "gate_plus_down_ms_mean": (
            statistics.mean(float(r["gateup_ms"]) + float(r["down_ms"]) for r in rows) if rows else None
        ),
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[p157] exact inter={report['inter_exact']}/{total} out={report['out_exact']}/{total} "
        f"gate={report['gateup_ms_mean']:.5f} down={report['down_ms_mean']:.5f} "
        f"combined={report['combined_ms_mean']:.5f}"
    )
    print(f"[p157] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
