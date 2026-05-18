#!/usr/bin/env python3
"""P159 · Triton active-MoE boundary timing with caller-owned scratch.

P157 measured the exact existing Triton active stage correctly, but each timed
combined call allocated the gate/up intermediate and down output.  This probe
keeps the same Triton kernels and slot-packed fixture contract, preallocates
the intermediate/output tensors at the benchmark boundary, and compares timing,
CUDA profiler event counts, and exactness against P147/P157 references.
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


def _ref_file_for(reference_dir: Path, layer_id: int, prompt_id: int) -> Path:
    path = reference_dir / f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_triton_stage.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"P147/P157 reference not found: {path}")
    return path


def _is_cuda_event(evt: Any) -> bool:
    device_type = getattr(evt, "device_type", None)
    if device_type is not None and str(device_type).lower().endswith("cuda"):
        return True
    return float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0) > 0.0


def _event_self_cuda_us(evt: Any) -> float:
    return float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0)


def _profile_events(fn: Callable[[], Any]) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        fn()
        torch.cuda.synchronize()

    events = list(prof.events())
    cuda_events = [evt for evt in events if _is_cuda_event(evt)]
    cpu_by_name: dict[str, int] = {}
    cuda_by_name: dict[str, dict[str, Any]] = {}
    for evt in events:
        name = getattr(evt, "name", None) or getattr(evt, "key", None) or "<unknown>"
        if _is_cuda_event(evt):
            rec = cuda_by_name.setdefault(name, {"name": name, "count": 0, "self_cuda_time_us": 0.0})
            rec["count"] += 1
            rec["self_cuda_time_us"] += _event_self_cuda_us(evt)
        else:
            cpu_by_name[name] = cpu_by_name.get(name, 0) + 1

    return {
        "cpu_event_count_total": len(events) - len(cuda_events),
        "cuda_event_count_total": len(cuda_events),
        "aten_empty_count": cpu_by_name.get("aten::empty", 0),
        "aten_empty_strided_count": cpu_by_name.get("aten::empty_strided", 0),
        "aten_contiguous_count": cpu_by_name.get("aten::contiguous", 0),
        "aten_to_count": cpu_by_name.get("aten::to", 0),
        "top_cuda_events": sorted(cuda_by_name.values(), key=lambda r: r["self_cuda_time_us"], reverse=True)[:12],
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(values) if values else None


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def _diff(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    return num - den


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Triton active-MoE with caller-owned inter/out scratch.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--p147-reference-dir", required=True)
    ap.add_argument("--p157-report", default=None, help="Optional P157 JSON report to compare observed means against.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-fixtures", type=int, default=0, help="0 means all fixtures in manifest.")
    ap.add_argument("--gate-block-inter", type=int, default=8)
    ap.add_argument("--gate-block-hidden", type=int, default=256)
    ap.add_argument("--gate-num-warps", type=int, default=4)
    ap.add_argument("--down-block-hidden", type=int, default=8)
    ap.add_argument("--down-block-inter", type=int, default=512)
    ap.add_argument("--down-num-warps", type=int, default=8)
    ap.add_argument("--profile-events", action="store_true", help="Run torch.profiler once per path/fixture.")
    ap.add_argument("--p157-expected-gateup-ms", type=float, default=0.045)
    ap.add_argument("--p157-expected-down-ms", type=float, default=0.007)
    ap.add_argument("--p157-expected-combined-ms", type=float, default=0.052)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P159 Triton active-boundary probe")

    from triton_kernels.nvfp4_moe import (
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        nvfp4_grouped_down_weighted_sum,
        nvfp4_grouped_gate_up_silu_fast_decode,
    )

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.p147_reference_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())
    fixtures = manifest["fixtures"][: args.max_fixtures or None]

    p157_report: dict[str, Any] | None = None
    if args.p157_report:
        p157_report = json.loads(Path(args.p157_report).read_text())

    rows: list[dict[str, Any]] = []
    print("[p159] Triton active-MoE caller-owned boundary probe")
    for entry in fixtures:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)
        ref = _load_fixture(_ref_file_for(ref_dir, layer_id, prompt_id), args.device)

        hidden = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        top_k = int(data["slot_gate_up_packed"].shape[0])
        slot_ids = torch.arange(top_k, device=hidden.device, dtype=torch.int32)
        routing_weights = data["routing_weights"].to(torch.float32).contiguous()

        gate_up_packed = data["slot_gate_up_packed"].contiguous()
        gate_up_scale = data["slot_gate_up_scale"].contiguous()
        gate_up_global_scale = data["slot_gate_up_global_scale"].to(hidden.device).contiguous()
        down_packed = data["slot_down_packed"].contiguous()
        down_scale = data["slot_down_scale"].contiguous()
        down_global_scale = data["slot_down_global_scale"].to(hidden.device).contiguous()

        inter_scratch = torch.empty((top_k, INTERMEDIATE_SIZE), device=hidden.device, dtype=torch.bfloat16)
        out_scratch = torch.empty((HIDDEN_SIZE,), device=hidden.device, dtype=torch.bfloat16)

        def gate_alloc_fn() -> torch.Tensor:
            return nvfp4_grouped_gate_up_silu_fast_decode(
                hidden,
                slot_ids,
                gate_up_packed,
                gate_up_scale,
                gate_up_global_scale,
                block_inter=args.gate_block_inter,
                block_hidden=args.gate_block_hidden,
                num_warps=args.gate_num_warps,
            )

        def gate_scratch_fn() -> torch.Tensor:
            return nvfp4_grouped_gate_up_silu_fast_decode(
                hidden,
                slot_ids,
                gate_up_packed,
                gate_up_scale,
                gate_up_global_scale,
                block_inter=args.gate_block_inter,
                block_hidden=args.gate_block_hidden,
                num_warps=args.gate_num_warps,
                out=inter_scratch,
            )

        inter_alloc = gate_alloc_fn().contiguous()
        inter_scratch_out = gate_scratch_fn()

        def down_alloc_fn() -> torch.Tensor:
            return nvfp4_grouped_down_weighted_sum(
                inter_alloc,
                slot_ids,
                routing_weights,
                down_packed,
                down_scale,
                down_global_scale,
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
            )

        def down_scratch_fn() -> torch.Tensor:
            return nvfp4_grouped_down_weighted_sum(
                inter_scratch_out,
                slot_ids,
                routing_weights,
                down_packed,
                down_scale,
                down_global_scale,
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
                out=out_scratch,
            )

        def combined_alloc_fn() -> torch.Tensor:
            inter_local = gate_alloc_fn()
            return nvfp4_grouped_down_weighted_sum(
                inter_local,
                slot_ids,
                routing_weights,
                down_packed,
                down_scale,
                down_global_scale,
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
            )

        def combined_scratch_fn() -> torch.Tensor:
            inter_local = gate_scratch_fn()
            return nvfp4_grouped_down_weighted_sum(
                inter_local,
                slot_ids,
                routing_weights,
                down_packed,
                down_scale,
                down_global_scale,
                block_hidden=args.down_block_hidden,
                block_inter=args.down_block_inter,
                num_warps=args.down_num_warps,
                out=out_scratch,
            )

        out_alloc = combined_alloc_fn().view(1, -1).contiguous()
        out_scratch_candidate = combined_scratch_fn().view(1, -1).contiguous()
        ref_inter = ref["triton_inter"].to(torch.bfloat16)
        ref_out = ref["routed_output"].to(torch.bfloat16).view(1, -1)

        row: dict[str, Any] = {
            "fixture_file": fixture_file,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "top_k": top_k,
            "alloc_gateup_ms": _bench_ms(gate_alloc_fn, warmup=args.warmup, iters=args.iters),
            "scratch_gateup_ms": _bench_ms(gate_scratch_fn, warmup=args.warmup, iters=args.iters),
            "alloc_down_ms": _bench_ms(down_alloc_fn, warmup=args.warmup, iters=args.iters),
            "scratch_down_ms": _bench_ms(down_scratch_fn, warmup=args.warmup, iters=args.iters),
            "alloc_combined_ms": _bench_ms(combined_alloc_fn, warmup=args.warmup, iters=args.iters),
            "scratch_combined_ms": _bench_ms(combined_scratch_fn, warmup=args.warmup, iters=args.iters),
            "alloc_inter_vs_p147": _metric(ref_inter, inter_alloc),
            "scratch_inter_vs_p147": _metric(ref_inter, inter_scratch_out),
            "alloc_out_vs_p147": _metric(ref_out, out_alloc),
            "scratch_out_vs_p147": _metric(ref_out, out_scratch_candidate),
            "scratch_vs_alloc_inter": _metric(inter_alloc, inter_scratch_out),
            "scratch_vs_alloc_out": _metric(out_alloc, out_scratch_candidate),
        }
        row["scratch_minus_alloc_combined_ms"] = row["scratch_combined_ms"] - row["alloc_combined_ms"]
        row["scratch_over_alloc_combined"] = _ratio(row["scratch_combined_ms"], row["alloc_combined_ms"])
        row["scratch_combined_minus_p157_expected_ms"] = row["scratch_combined_ms"] - args.p157_expected_combined_ms
        row["scratch_combined_over_p157_expected"] = _ratio(
            row["scratch_combined_ms"], args.p157_expected_combined_ms
        )

        if args.profile_events:
            row["alloc_profile_events"] = _profile_events(combined_alloc_fn)
            row["scratch_profile_events"] = _profile_events(combined_scratch_fn)

        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"alloc={row['alloc_combined_ms']:.5f} scratch={row['scratch_combined_ms']:.5f} "
            f"delta={row['scratch_minus_alloc_combined_ms']:+.5f} "
            f"exact=({row['scratch_inter_vs_p147']['exact']},{row['scratch_out_vs_p147']['exact']})",
            flush=True,
        )

    alloc_gate = _mean(rows, "alloc_gateup_ms")
    scratch_gate = _mean(rows, "scratch_gateup_ms")
    alloc_down = _mean(rows, "alloc_down_ms")
    scratch_down = _mean(rows, "scratch_down_ms")
    alloc_combined = _mean(rows, "alloc_combined_ms")
    scratch_combined = _mean(rows, "scratch_combined_ms")
    p157_gate = (
        float(p157_report["gateup_ms_mean"])
        if p157_report and p157_report.get("gateup_ms_mean") is not None
        else args.p157_expected_gateup_ms
    )
    p157_down = (
        float(p157_report["down_ms_mean"])
        if p157_report and p157_report.get("down_ms_mean") is not None
        else args.p157_expected_down_ms
    )
    p157_combined = (
        float(p157_report["combined_ms_mean"])
        if p157_report and p157_report.get("combined_ms_mean") is not None
        else args.p157_expected_combined_ms
    )

    total = len(rows)
    report = {
        "schema": "lynn-p159-qwen36-triton-active-boundary-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "p147_reference_dir": str(ref_dir),
        "p157_report": str(args.p157_report) if args.p157_report else None,
        "total": total,
        "kernel_config": {
            "gate_block_inter": args.gate_block_inter,
            "gate_block_hidden": args.gate_block_hidden,
            "gate_num_warps": args.gate_num_warps,
            "down_block_hidden": args.down_block_hidden,
            "down_block_inter": args.down_block_inter,
            "down_num_warps": args.down_num_warps,
        },
        "exact_counts": {
            "scratch_inter_vs_p147": sum(int(r["scratch_inter_vs_p147"]["exact"]) for r in rows),
            "scratch_out_vs_p147": sum(int(r["scratch_out_vs_p147"]["exact"]) for r in rows),
            "scratch_vs_alloc_inter": sum(int(r["scratch_vs_alloc_inter"]["exact"]) for r in rows),
            "scratch_vs_alloc_out": sum(int(r["scratch_vs_alloc_out"]["exact"]) for r in rows),
        },
        "means_ms": {
            "alloc_gateup": alloc_gate,
            "scratch_gateup": scratch_gate,
            "alloc_down": alloc_down,
            "scratch_down": scratch_down,
            "alloc_combined": alloc_combined,
            "scratch_combined": scratch_combined,
            "scratch_minus_alloc_combined": _diff(scratch_combined, alloc_combined),
            "scratch_over_alloc_combined": _ratio(scratch_combined, alloc_combined),
        },
        "p157_expectation": {
            "source": "report" if p157_report else "cli_defaults",
            "gateup_ms_mean": p157_gate,
            "down_ms_mean": p157_down,
            "combined_ms_mean": p157_combined,
            "scratch_gateup_minus_p157_ms": _diff(scratch_gate, p157_gate),
            "scratch_down_minus_p157_ms": _diff(scratch_down, p157_down),
            "scratch_combined_minus_p157_ms": _diff(scratch_combined, p157_combined),
            "scratch_combined_over_p157": _ratio(scratch_combined, p157_combined),
        },
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[p159] exact scratch inter={report['exact_counts']['scratch_inter_vs_p147']}/{total} "
        f"out={report['exact_counts']['scratch_out_vs_p147']}/{total} "
        f"alloc={alloc_combined:.5f} scratch={scratch_combined:.5f} "
        f"p157={p157_combined:.5f}"
    )
    print(f"[p159] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
