#!/usr/bin/env python3
"""P152 · Export native packed-MoE stage outputs for P147 comparison.

This is a bridge harness: it runs the existing native
``moe_slot_packed_nvfp4_probe`` on p138 slot-packed fixtures, writes one
candidate safetensors file per fixture, and records fixture-stage latency.

The output directory is intentionally compatible with P147's
``--candidate-output-dir``.  P152 itself does not decide promotion; P147 is the
strict Triton-stage admission gate.
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Export native packed MoE outputs for P147.")
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--candidate-output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from safetensors.torch import save_file
    from engine.native_cuda import load_lynn_native_extension

    packed_dir = Path(args.packed_fixtures)
    out_dir = Path(args.candidate_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((packed_dir / "manifest.json").read_text())

    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_slot_packed_nvfp4_probe"):
        raise RuntimeError("native extension lacks moe_slot_packed_nvfp4_probe")

    print("[p152] Native packed MoE candidate output export")
    print(f"[p152] packed_fixtures={packed_dir}")
    print(f"[p152] candidate_output_dir={out_dir}")

    rows: list[dict[str, Any]] = []
    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = int(entry["layer_id"])
        prompt_id = int(entry["prompt_id"])
        data = _load_fixture(packed_dir / fixture_file, args.device)
        x = data["hidden_in"].to(torch.bfloat16).view(-1).contiguous()
        routing = data["routing_weights"].to(torch.float32).contiguous()

        def run() -> torch.Tensor:
            return ext.moe_slot_packed_nvfp4_probe(
                x,
                routing,
                data["slot_gate_up_packed"].contiguous(),
                data["slot_gate_up_scale"].to(torch.float16).contiguous(),
                data["slot_gate_up_global_scale"].to(torch.float16).contiguous(),
                data["slot_down_packed"].contiguous(),
                data["slot_down_scale"].to(torch.float16).contiguous(),
                data["slot_down_global_scale"].to(torch.float16).contiguous(),
            )

        out = run().view(1, -1).contiguous()
        ms = _bench_ms(run, warmup=args.warmup, iters=args.iters)
        out_name = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slot_packed.safetensors"
        save_file(
            {
                "candidate_output": out.detach().cpu(),
                "routing_weights": data["routing_weights"].detach().cpu().contiguous(),
                "expert_ids": data["expert_ids"].detach().cpu().contiguous(),
            },
            str(out_dir / out_name),
        )
        row = {
            "fixture_file": fixture_file,
            "candidate_file": out_name,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "candidate_ms": ms,
        }
        rows.append(row)
        print(f"  L{layer_id:02d}/P{prompt_id:02d} candidate_ms={ms:.5f}", flush=True)

    mean_ms = sum(r["candidate_ms"] for r in rows) / len(rows) if rows else None
    report = {
        "schema": "lynn-p152-native-packed-moe-stage-output-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "packed_fixtures": str(packed_dir),
        "candidate_output_dir": str(out_dir),
        "candidate": "moe_slot_packed_nvfp4_probe",
        "total": len(rows),
        "candidate_ms_mean": mean_ms,
        "candidate_ms_max": max((r["candidate_ms"] for r in rows), default=None),
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[p152] candidate_ms_mean={mean_ms:.5f}" if mean_ms is not None else "[p152] no rows")
    print(f"[p152] report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
