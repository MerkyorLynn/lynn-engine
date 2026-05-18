#!/usr/bin/env python3
"""P162 · Qwen3.6 shared-expert finalize boundary probe.

P158 showed the MoE finalize/shared-add tail costs ~0.03 ms/layer.  This probe
keeps router, active expert math, and shared-expert matmul out of scope, then
compares only the final scalar-gate/multiply/add boundary:

    default: active + shared * sigmoid(linear(h, shared_gate))
    candidate: torch scalar gate + Triton fused shared*gate + active add

It uses real resident layer weights plus P138 slot-packed fixtures for the
active output.  No serving path is touched.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import dual_scalar_bridge  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.nvfp4_moe import (  # noqa: E402
    nvfp4_grouped_down_weighted_sum,
    nvfp4_grouped_gate_up_silu_fast_decode,
)
from triton_kernels.shared_expert_gate import add_shared_expert_gate_from_scalar_triton  # noqa: E402


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
    max_abs = float(diff.abs().max())
    if max_abs == 0.0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0, "exact": 1}
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / ref_norm),
        "cosine": float(torch.dot(rf, cf) / (ref_norm * cand_norm)),
        "exact": 0,
    }


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


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def _shared_expert_forward(h_flat: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    if (
        "mlp.shared_expert.gate_proj.weight.packed" in w
        and "mlp.shared_expert.up_proj.weight.packed" in w
        and "mlp.shared_expert.down_proj.weight.packed" in w
    ):
        gate_s, up_s = dual_scalar_bridge(
            h_flat[0],
            w["mlp.shared_expert.gate_proj.weight.packed"],
            w["mlp.shared_expert.up_proj.weight.packed"],
        )
        return w["mlp.shared_expert.down_proj.weight.packed"](
            (F.silu(gate_s) * up_s).to(h_flat.dtype)
        ).reshape_as(h_flat)
    if "mlp.shared_expert._gate_up_proj.weight" in w:
        gate_up_s = F.linear(h_flat, w["mlp.shared_expert._gate_up_proj.weight"])
        gate_s, up_s = gate_up_s.chunk(2, dim=-1)
    else:
        gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    return F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])


def _active_out_from_slot_fixture(fixture: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden = fixture["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
    top_k = int(fixture["slot_gate_up_packed"].shape[0])
    slot_ids = torch.arange(top_k, device=hidden.device, dtype=torch.int32)
    inter = nvfp4_grouped_gate_up_silu_fast_decode(
        hidden,
        slot_ids,
        fixture["slot_gate_up_packed"].contiguous(),
        fixture["slot_gate_up_scale"].contiguous(),
        fixture["slot_gate_up_global_scale"].to(hidden.device).contiguous(),
        block_inter=8,
        block_hidden=256,
        num_warps=4,
    )
    return nvfp4_grouped_down_weighted_sum(
        inter.contiguous(),
        slot_ids,
        fixture["routing_weights"].to(torch.float32).contiguous(),
        fixture["slot_down_packed"].contiguous(),
        fixture["slot_down_scale"].contiguous(),
        fixture["slot_down_global_scale"].to(hidden.device).contiguous(),
        block_hidden=8,
        block_inter=512,
        num_warps=8,
    ).reshape(1, -1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Qwen3.6 shared finalize boundary.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--max-fixtures", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P162")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=False)
    packed_dir = Path(args.packed_fixtures)
    manifest = json.loads((packed_dir / "manifest.json").read_text(encoding="utf-8"))
    fixture_entries = manifest["fixtures"][: args.max_fixtures or None]

    rows: list[dict[str, Any]] = []
    for entry in fixture_entries:
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        w = runner.layer_weights[layer_id]
        if "mlp.shared_expert_gate.weight" not in w:
            continue
        fixture = _load_fixture(packed_dir / entry["fixture_file"], args.device)
        h_flat = fixture["hidden_in"].to(dtype).reshape(1, -1).contiguous()
        active = _active_out_from_slot_fixture(fixture).to(dtype).contiguous()
        shared = _shared_expert_forward(h_flat, w).to(dtype).contiguous()

        def gate_fn() -> torch.Tensor:
            return torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))

        gate = gate_fn()

        def default_fn() -> torch.Tensor:
            return active + shared * torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))

        def split_fn() -> torch.Tensor:
            g = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
            return add_shared_expert_gate_from_scalar_triton(active, shared, g)

        def triton_add_only_fn() -> torch.Tensor:
            return add_shared_expert_gate_from_scalar_triton(active, shared, gate)

        default_out = default_fn()
        split_out = split_fn()
        add_only_out = triton_add_only_fn()
        m_split = _metric(default_out, split_out)
        m_add = _metric(default_out, add_only_out)

        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "gate_ms": _bench_ms(gate_fn, warmup=args.warmup, iters=args.iters),
            "default_finalize_ms": _bench_ms(default_fn, warmup=args.warmup, iters=args.iters),
            "torch_gate_triton_add_ms": _bench_ms(split_fn, warmup=args.warmup, iters=args.iters),
            "triton_add_only_ms": _bench_ms(triton_add_only_fn, warmup=args.warmup, iters=args.iters),
            "split_max_abs": m_split["max_abs"],
            "split_cosine": m_split["cosine"],
            "split_exact": m_split["exact"],
            "add_only_max_abs": m_add["max_abs"],
            "add_only_cosine": m_add["cosine"],
            "add_only_exact": m_add["exact"],
        }
        row["split_delta_ms"] = row["torch_gate_triton_add_ms"] - row["default_finalize_ms"]
        row["add_only_delta_ms"] = row["triton_add_only_ms"] - row["default_finalize_ms"]
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} default={row['default_finalize_ms']:.5f} "
            f"split={row['torch_gate_triton_add_ms']:.5f} add_only={row['triton_add_only_ms']:.5f} "
            f"exact=({row['split_exact']},{row['add_only_exact']})",
            flush=True,
        )

    report = {
        "schema": "lynn-p162-qwen36-shared-finalize-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "packed_fixtures": str(packed_dir),
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "fixtures": len(rows),
        "summary": {
            "default_finalize_ms_mean": _mean(rows, "default_finalize_ms"),
            "torch_gate_triton_add_ms_mean": _mean(rows, "torch_gate_triton_add_ms"),
            "triton_add_only_ms_mean": _mean(rows, "triton_add_only_ms"),
            "split_delta_ms_mean": _mean(rows, "split_delta_ms"),
            "add_only_delta_ms_mean": _mean(rows, "add_only_delta_ms"),
            "split_exact": sum(int(r["split_exact"]) for r in rows),
            "add_only_exact": sum(int(r["add_only_exact"]) for r in rows),
            "max_split_abs": max((float(r["split_max_abs"]) for r in rows), default=None),
            "max_add_only_abs": max((float(r["add_only_max_abs"]) for r in rows), default=None),
        },
        "decision": "SHARED_FINALIZE_CLOSED_OR_FLAT",
        "rows": rows,
    }
    if rows and report["summary"]["split_exact"] == len(rows) and (report["summary"]["split_delta_ms_mean"] or 0.0) < -0.001:
        report["decision"] = "SHARED_FINALIZE_CANDIDATE"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "decision": report["decision"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
