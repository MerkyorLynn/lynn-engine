#!/usr/bin/env python3
"""P167 · Probe exact prepared BF16 shared-expert boundary.

This is a fixture-level admission probe. It does not wire resident serving.
It compares the default shared-expert BF16 path with caller-owned scratch
variants using the same real Qwen3.6-35B layer weights and p138 hidden inputs
used by the MoE component profiles.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _load_fixture(path: Path, device: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load as load_buffer
    from safetensors.torch import load_file

    if len(path.suffixes) >= 2 and path.suffixes[-2:] == [".safetensors", ".gz"]:
        with gzip.open(str(path), "rb") as f:
            raw = f.read()
        return {k: v.to(device) for k, v in load_buffer(raw).items()}
    return load_file(str(path), device=device)


def _bench_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
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


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    max_abs = float(diff.abs().max().item())
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean().item()),
        "rel_l2": float(torch.linalg.vector_norm(diff).item() / float(ref_norm.item())),
        "cosine": float(torch.dot(rf, cf).item() / float((ref_norm * cand_norm).item())),
        "exact": 1 if max_abs == 0.0 else 0,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _shared_default(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    if "mlp.shared_expert._gate_up_proj.weight" in w:
        gate_up = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.shared_expert.down_proj.weight"])


def _finalize_default(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    w: dict[str, Any],
) -> torch.Tensor:
    if "mlp.shared_expert_gate.weight" in w:
        shared = shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    return moe_out + shared


def _shared_mm_out(
    h_flat: torch.Tensor,
    w: dict[str, Any],
    gate_up_scratch: torch.Tensor,
    shared_scratch: torch.Tensor,
) -> torch.Tensor:
    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
    gate, up = gate_up_scratch.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    torch.mm(hidden, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
    return shared_scratch


def _shared_mm_out_inplace_silu(
    h_flat: torch.Tensor,
    w: dict[str, Any],
    gate_up_scratch: torch.Tensor,
    inter_scratch: torch.Tensor,
    shared_scratch: torch.Tensor,
) -> torch.Tensor:
    torch.mm(h_flat, w["mlp.shared_expert._gate_up_proj.weight"].t(), out=gate_up_scratch)
    gate, up = gate_up_scratch.chunk(2, dim=-1)
    torch.sigmoid(gate, out=inter_scratch)
    inter_scratch.mul_(gate)
    inter_scratch.mul_(up)
    torch.mm(inter_scratch, w["mlp.shared_expert.down_proj.weight"].t(), out=shared_scratch)
    return shared_scratch


def _finalize_inplace(
    h_flat: torch.Tensor,
    moe_out: torch.Tensor,
    shared: torch.Tensor,
    w: dict[str, Any],
    gate_scratch: torch.Tensor,
    out_scratch: torch.Tensor,
) -> torch.Tensor:
    out_scratch.copy_(moe_out)
    if "mlp.shared_expert_gate.weight" in w:
        torch.mm(h_flat, w["mlp.shared_expert_gate.weight"].t(), out=gate_scratch)
        gate_scratch.sigmoid_()
        shared = shared * gate_scratch
    out_scratch.add_(shared)
    return out_scratch


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe exact prepared BF16 shared-expert boundary.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-fixtures", type=int, default=18)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    t0 = time.time()
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    load_seconds = time.time() - t0

    packed_dir = Path(args.packed_fixtures)
    manifest = json.loads((packed_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"][: args.max_fixtures]:
        layer_id = int(entry["layer_id"])
        data = _load_fixture(packed_dir / entry["fixture_file"], args.device)
        h_flat = data["hidden_in"].to(dtype).view(1, -1).contiguous()
        w = runner.layer_weights[layer_id]
        if "mlp.shared_expert.gate_proj.weight" not in w:
            continue
        if "mlp.shared_expert._gate_up_proj.weight" not in w:
            w["mlp.shared_expert._gate_up_proj.weight"] = torch.cat(
                [w["mlp.shared_expert.gate_proj.weight"], w["mlp.shared_expert.up_proj.weight"]],
                dim=0,
            ).contiguous()
        moe_out = torch.zeros_like(h_flat)
        gate_up_scratch = torch.empty((1, w["mlp.shared_expert._gate_up_proj.weight"].shape[0]), device=args.device, dtype=dtype)
        inter_scratch = torch.empty((1, w["mlp.shared_expert.down_proj.weight"].shape[1]), device=args.device, dtype=dtype)
        shared_scratch = torch.empty_like(h_flat)
        gate_scratch = torch.empty((1, 1), device=args.device, dtype=dtype)
        final_scratch = torch.empty_like(h_flat)

        shared_ref = _shared_default(h_flat, w)
        final_ref = _finalize_default(h_flat, moe_out, shared_ref, w)

        def shared_mm_out_fn() -> torch.Tensor:
            return _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)

        def shared_inplace_silu_fn() -> torch.Tensor:
            return _shared_mm_out_inplace_silu(h_flat, w, gate_up_scratch, inter_scratch, shared_scratch)

        shared_mm = shared_mm_out_fn().clone()
        shared_inplace = shared_inplace_silu_fn().clone()

        def final_mm_out_fn() -> torch.Tensor:
            shared = _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)
            return _finalize_default(h_flat, moe_out, shared, w)

        def final_prepared_fn() -> torch.Tensor:
            shared = _shared_mm_out(h_flat, w, gate_up_scratch, shared_scratch)
            return _finalize_inplace(h_flat, moe_out, shared, w, gate_scratch, final_scratch)

        final_mm = final_mm_out_fn().clone()
        final_prepared = final_prepared_fn().clone()

        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": int(entry["prompt_id"]),
            "shared_default_ms": _bench_ms(lambda: _shared_default(h_flat, w), warmup=args.warmup, iters=args.iters),
            "shared_mm_out_ms": _bench_ms(shared_mm_out_fn, warmup=args.warmup, iters=args.iters),
            "shared_inplace_silu_ms": _bench_ms(shared_inplace_silu_fn, warmup=args.warmup, iters=args.iters),
            "final_default_ms": _bench_ms(lambda: _finalize_default(h_flat, moe_out, shared_ref, w), warmup=args.warmup, iters=args.iters),
            "final_mm_out_ms": _bench_ms(final_mm_out_fn, warmup=args.warmup, iters=args.iters),
            "final_prepared_ms": _bench_ms(final_prepared_fn, warmup=args.warmup, iters=args.iters),
            "shared_mm_out_vs_default": _metric(shared_ref, shared_mm),
            "shared_inplace_silu_vs_default": _metric(shared_ref, shared_inplace),
            "final_mm_out_vs_default": _metric(final_ref, final_mm),
            "final_prepared_vs_default": _metric(final_ref, final_prepared),
        }
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{row['prompt_id']:02d} "
            f"shared default={row['shared_default_ms']:.5f} mm_out={row['shared_mm_out_ms']:.5f} "
            f"final default={row['final_default_ms']:.5f} prepared={row['final_prepared_ms']:.5f} "
            f"exact=({row['shared_mm_out_vs_default']['exact']},{row['final_prepared_vs_default']['exact']})",
            flush=True,
        )

    total = len(rows)
    report = {
        "schema": "lynn-p167-qwen36-shared-expert-prepared-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "packed_fixtures": str(packed_dir),
        "load_seconds": load_seconds,
        "total": total,
        "means": {
            "shared_default_ms": _mean(rows, "shared_default_ms"),
            "shared_mm_out_ms": _mean(rows, "shared_mm_out_ms"),
            "shared_inplace_silu_ms": _mean(rows, "shared_inplace_silu_ms"),
            "final_default_ms": _mean(rows, "final_default_ms"),
            "final_mm_out_ms": _mean(rows, "final_mm_out_ms"),
            "final_prepared_ms": _mean(rows, "final_prepared_ms"),
        },
        "exact_counts": {
            "shared_mm_out": sum(int(r["shared_mm_out_vs_default"]["exact"]) for r in rows),
            "shared_inplace_silu": sum(int(r["shared_inplace_silu_vs_default"]["exact"]) for r in rows),
            "final_mm_out": sum(int(r["final_mm_out_vs_default"]["exact"]) for r in rows),
            "final_prepared": sum(int(r["final_prepared_vs_default"]["exact"]) for r in rows),
        },
        "max_abs_max": {
            "shared_mm_out": max((float(r["shared_mm_out_vs_default"]["max_abs"]) for r in rows), default=None),
            "shared_inplace_silu": max((float(r["shared_inplace_silu_vs_default"]["max_abs"]) for r in rows), default=None),
            "final_mm_out": max((float(r["final_mm_out_vs_default"]["max_abs"]) for r in rows), default=None),
            "final_prepared": max((float(r["final_prepared_vs_default"]["max_abs"]) for r in rows), default=None),
        },
        "results": rows,
    }
    means = report["means"]
    exact = report["exact_counts"]
    shared_delta = (means["shared_default_ms"] or 0.0) - (means["shared_mm_out_ms"] or 0.0)
    final_delta = (means["final_default_ms"] or 0.0) - (means["final_prepared_ms"] or 0.0)
    if exact["shared_mm_out"] == total and shared_delta > 0.002:
        verdict = "SHARED_MM_OUT_CANDIDATE"
    elif exact["final_prepared"] == total and final_delta > 0.002:
        verdict = "FINAL_PREPARED_CANDIDATE"
    else:
        verdict = "CLOSED_OR_FLAT"
    report["deltas_ms"] = {
        "shared_mm_out_vs_default": shared_delta,
        "final_prepared_vs_default": final_delta,
    }
    report["verdict"] = verdict

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "verdict": verdict, "means": means, "exact_counts": exact}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
