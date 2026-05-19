#!/usr/bin/env python3
"""P177 · Qwen3.6 router linear caller-owned output probe.

P163 proved that `torch.topk(..., out=...)` is exact for the decode router and
saves a small boundary cost. This probe tests the preceding router projection:

    reference: F.linear(hidden, gate_weight)
    candidate: torch.mm(hidden, gate_weight.t(), out=preallocated_logits)

It is fixture-only and does not touch the serving path.
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


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def _max(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Qwen3.6 router linear out-buffer exactness and timing.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=160)
    ap.add_argument("--max-fixtures", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P177")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=False)

    packed_dir = Path(args.packed_fixtures)
    manifest = json.loads((packed_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["fixtures"][: args.max_fixtures or None]
    rows: list[dict[str, Any]] = []

    for entry in entries:
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        fixture = _load_fixture(packed_dir / entry["fixture_file"], args.device)
        h_flat = fixture["hidden_in"].to(dtype).reshape(1, -1).contiguous()
        gate_w = runner.layer_weights[layer_id]["mlp.gate.weight"]
        top_k = int(runner.layer_cfgs[layer_id].get("num_experts_per_tok", 8))
        logits_out = torch.empty((1, gate_w.shape[0]), device=h_flat.device, dtype=h_flat.dtype)
        topk_vals = torch.empty((1, top_k), device=h_flat.device, dtype=h_flat.dtype)
        topk_idx = torch.empty((1, top_k), device=h_flat.device, dtype=torch.long)
        softmax_out = torch.empty((1, top_k), device=h_flat.device, dtype=torch.float32)
        gate_w_t = gate_w.t().contiguous()

        def linear_ref() -> torch.Tensor:
            return F.linear(h_flat, gate_w)

        def linear_out() -> torch.Tensor:
            torch.mm(h_flat, gate_w_t, out=logits_out)
            return logits_out

        def router_ref() -> tuple[torch.Tensor, torch.Tensor]:
            logits = F.linear(h_flat, gate_w)
            vals, idx = torch.topk(logits, top_k, dim=-1, sorted=False)
            route = F.softmax(vals, dim=-1, dtype=torch.float32)
            return idx[0].to(torch.int32).contiguous(), route[0].contiguous()

        def router_out() -> tuple[torch.Tensor, torch.Tensor]:
            torch.mm(h_flat, gate_w_t, out=logits_out)
            torch.topk(logits_out, top_k, dim=-1, sorted=False, out=(topk_vals, topk_idx))
            torch.softmax(topk_vals, dim=-1, dtype=torch.float32, out=softmax_out)
            return topk_idx[0].to(torch.int32).contiguous(), softmax_out[0].contiguous()

        ref_logits = linear_ref()
        out_logits = linear_out()
        ref_ids, ref_route = router_ref()
        out_ids, out_route = router_out()
        logits_abs = _max_abs(ref_logits, out_logits)
        route_abs = _max_abs(ref_route, out_route)
        ids_exact = bool(torch.equal(ref_ids, out_ids))

        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "linear_ref_ms": _bench_ms(linear_ref, warmup=args.warmup, iters=args.iters),
            "linear_out_ms": _bench_ms(linear_out, warmup=args.warmup, iters=args.iters),
            "router_ref_ms": _bench_ms(router_ref, warmup=args.warmup, iters=args.iters),
            "router_out_ms": _bench_ms(router_out, warmup=args.warmup, iters=args.iters),
            "logits_max_abs": logits_abs,
            "route_max_abs": route_abs,
            "ids_exact": ids_exact,
        }
        row["linear_delta_ms"] = row["linear_out_ms"] - row["linear_ref_ms"]
        row["router_delta_ms"] = row["router_out_ms"] - row["router_ref_ms"]
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} linear={row['linear_ref_ms']:.5f}->{row['linear_out_ms']:.5f} "
            f"router={row['router_ref_ms']:.5f}->{row['router_out_ms']:.5f} "
            f"ids={ids_exact} logits_abs={logits_abs:.3g} route_abs={route_abs:.3g}",
            flush=True,
        )

    exact = sum(
        1
        for r in rows
        if r["ids_exact"] and float(r["logits_max_abs"]) == 0.0 and float(r["route_max_abs"]) == 0.0
    )
    summary = {
        "exact": exact,
        "linear_ref_ms_mean": _mean(rows, "linear_ref_ms"),
        "linear_out_ms_mean": _mean(rows, "linear_out_ms"),
        "linear_delta_ms_mean": _mean(rows, "linear_delta_ms"),
        "router_ref_ms_mean": _mean(rows, "router_ref_ms"),
        "router_out_ms_mean": _mean(rows, "router_out_ms"),
        "router_delta_ms_mean": _mean(rows, "router_delta_ms"),
        "logits_max_abs_max": _max(rows, "logits_max_abs"),
        "route_max_abs_max": _max(rows, "route_max_abs"),
    }
    decision = "ROUTER_LINEAR_OUT_CLOSED"
    if rows and exact == len(rows) and (summary["router_delta_ms_mean"] or 0.0) < -0.001:
        decision = "ROUTER_LINEAR_OUT_CANDIDATE"

    report = {
        "schema": "lynn-p177-qwen36-router-linear-out-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "packed_fixtures": str(packed_dir),
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "fixtures": len(rows),
        "summary": summary,
        "decision": decision,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "decision": decision, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
