#!/usr/bin/env python3
"""P164 · Qwen3.6 router softmax output-buffer boundary probe.

P163 promoted caller-owned `torch.topk(..., out=...)`.  P164 tests the next
tiny exact boundary: `torch.softmax(..., dtype=float32, out=scratch)` for the
top-k router logits.
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


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Qwen3.6 router softmax output-buffer boundary.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--max-fixtures", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P164")
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
        w = runner.layer_weights[layer_id]
        top_k = int(runner.layer_cfgs[layer_id].get("num_experts_per_tok", 8))
        gate_w = w["mlp.gate.weight"]
        vals_buf = torch.empty((1, top_k), device=h_flat.device, dtype=h_flat.dtype)
        idx_buf = torch.empty((1, top_k), device=h_flat.device, dtype=torch.long)
        softmax_buf = torch.empty((1, top_k), device=h_flat.device, dtype=torch.float32)

        def default_router() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = F.linear(h_flat, gate_w)
            vals, idx = torch.topk(logits, top_k, dim=-1, sorted=False)
            route = F.softmax(vals, dim=-1, dtype=torch.float32)[0].contiguous()
            ids = idx[0].to(torch.int32).contiguous()
            return logits, ids, route

        def topk_out_router() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = F.linear(h_flat, gate_w)
            torch.topk(logits, top_k, dim=-1, sorted=False, out=(vals_buf, idx_buf))
            route = F.softmax(vals_buf, dim=-1, dtype=torch.float32)[0].contiguous()
            ids = idx_buf[0].to(torch.int32).contiguous()
            return logits, ids, route

        def topk_softmax_out_router() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = F.linear(h_flat, gate_w)
            torch.topk(logits, top_k, dim=-1, sorted=False, out=(vals_buf, idx_buf))
            torch.softmax(vals_buf, dim=-1, dtype=torch.float32, out=softmax_buf)
            ids = idx_buf[0].to(torch.int32).contiguous()
            return logits, ids, softmax_buf[0]

        logits = F.linear(h_flat, gate_w)
        topk_vals, _ = torch.topk(logits, top_k, dim=-1, sorted=False)

        def softmax_alloc_only() -> torch.Tensor:
            return F.softmax(topk_vals, dim=-1, dtype=torch.float32)

        def softmax_out_only() -> torch.Tensor:
            torch.softmax(topk_vals, dim=-1, dtype=torch.float32, out=softmax_buf)
            return softmax_buf

        ref_logits, ref_ids, ref_route = default_router()
        topk_logits, topk_ids, topk_route = topk_out_router()
        cand_logits, cand_ids, cand_route = topk_softmax_out_router()
        row = {
            "fixture_file": entry["fixture_file"],
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "top_k": top_k,
            "default_router_ms": _bench_ms(default_router, warmup=args.warmup, iters=args.iters),
            "topk_out_router_ms": _bench_ms(topk_out_router, warmup=args.warmup, iters=args.iters),
            "topk_softmax_out_router_ms": _bench_ms(topk_softmax_out_router, warmup=args.warmup, iters=args.iters),
            "softmax_alloc_only_ms": _bench_ms(softmax_alloc_only, warmup=args.warmup, iters=args.iters),
            "softmax_out_only_ms": _bench_ms(softmax_out_only, warmup=args.warmup, iters=args.iters),
            "topk_ids_exact": bool(torch.equal(ref_ids, topk_ids)),
            "candidate_ids_exact": bool(torch.equal(ref_ids, cand_ids)),
            "topk_route_max_abs": _max_abs(ref_route, topk_route),
            "candidate_route_max_abs": _max_abs(ref_route, cand_route),
            "candidate_logits_max_abs": _max_abs(ref_logits, cand_logits),
        }
        row["softmax_out_minus_topk_out_router_ms"] = row["topk_softmax_out_router_ms"] - row["topk_out_router_ms"]
        row["softmax_out_minus_alloc_only_ms"] = row["softmax_out_only_ms"] - row["softmax_alloc_only_ms"]
        rows.append(row)
        print(
            f"  L{layer_id:02d}/P{prompt_id:02d} "
            f"topk={row['topk_out_router_ms']:.5f} softmax_out={row['topk_softmax_out_router_ms']:.5f} "
            f"delta={row['softmax_out_minus_topk_out_router_ms']:+.5f} "
            f"exact={row['candidate_ids_exact']} abs={row['candidate_route_max_abs']:.3g}",
            flush=True,
        )

    exact = sum(
        1
        for r in rows
        if r["candidate_ids_exact"]
        and float(r["candidate_route_max_abs"]) == 0.0
        and float(r["candidate_logits_max_abs"]) == 0.0
    )
    report = {
        "schema": "lynn-p164-qwen36-router-softmax-boundary-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model,
        "packed_fixtures": str(packed_dir),
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
        "fixtures": len(rows),
        "summary": {
            "exact": exact,
            "topk_out_router_ms_mean": _mean(rows, "topk_out_router_ms"),
            "topk_softmax_out_router_ms_mean": _mean(rows, "topk_softmax_out_router_ms"),
            "softmax_out_minus_topk_out_router_ms_mean": _mean(rows, "softmax_out_minus_topk_out_router_ms"),
            "softmax_alloc_only_ms_mean": _mean(rows, "softmax_alloc_only_ms"),
            "softmax_out_only_ms_mean": _mean(rows, "softmax_out_only_ms"),
            "softmax_out_minus_alloc_only_ms_mean": _mean(rows, "softmax_out_minus_alloc_only_ms"),
            "candidate_route_max_abs_max": max((float(r["candidate_route_max_abs"]) for r in rows), default=None),
            "candidate_logits_max_abs_max": max((float(r["candidate_logits_max_abs"]) for r in rows), default=None),
        },
        "decision": "ROUTER_SOFTMAX_OUT_BUFFER_CLOSED_OR_FLAT",
        "rows": rows,
    }
    delta = report["summary"]["softmax_out_minus_topk_out_router_ms_mean"]
    if rows and exact == len(rows) and delta is not None and delta < -0.0005:
        report["decision"] = "ROUTER_SOFTMAX_OUT_BUFFER_CANDIDATE"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "decision": report["decision"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
