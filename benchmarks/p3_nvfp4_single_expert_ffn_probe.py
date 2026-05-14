#!/usr/bin/env python3
"""P3-F probe: one expert FFN from packed NVFP4 gate/up/down."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.dequant import dequantize_nvfp4_v8_rtn_weight
from engine.nvfp4_runtime import PackedNVFP4Linear
from triton_kernels.nvfp4_linear import nvfp4_dual_matvec_packed


def _load(v8_dir: Path, layer: int, expert: int, proj: str, device: str) -> PackedNVFP4Linear:
    base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(st, base, name=base, device=device)


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(cosine),
    }


def _bench(fn, *, warmup: int, iters: int) -> float:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    v8_dir = Path(args.v8)
    gate = _load(v8_dir, args.layer, args.expert, "gate_proj", args.device)
    up = _load(v8_dir, args.layer, args.expert, "up_proj", args.device)
    down = _load(v8_dir, args.layer, args.expert, "down_proj", args.device)

    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)
    x = torch.randn(gate.in_features, device=args.device, dtype=torch.bfloat16, generator=gen)

    gate_w = dequantize_nvfp4_v8_rtn_weight(
        gate.weight_packed, gate.weight_scale, gate.weight_global_scale, output_dtype=torch.float32
    )
    up_w = dequantize_nvfp4_v8_rtn_weight(
        up.weight_packed, up.weight_scale, up.weight_global_scale, output_dtype=torch.float32
    )
    down_w = dequantize_nvfp4_v8_rtn_weight(
        down.weight_packed, down.weight_scale, down.weight_global_scale, output_dtype=torch.float32
    )

    def resident_ffn():
        gate_out = F.linear(x.float().unsqueeze(0), gate_w)
        up_out = F.linear(x.float().unsqueeze(0), up_w)
        inter = F.silu(gate_out) * up_out
        return F.linear(inter, down_w).squeeze(0).to(torch.bfloat16)

    def packed_ffn():
        gate_out, up_out = nvfp4_dual_matvec_packed(
            x,
            gate.weight_packed,
            gate.weight_scale,
            gate.weight_global_scale,
            up.weight_packed,
            up.weight_scale,
            up.weight_global_scale,
        )
        inter = F.silu(gate_out).to(torch.bfloat16) * up_out.to(torch.bfloat16)
        return down(inter).to(torch.bfloat16)

    ref = resident_ffn()
    out = packed_ffn()
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-single-expert-ffn-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "expert": args.expert,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "comparison": _compare(out, ref),
        "timing_ms": {
            "resident_expert_ffn": _bench(resident_ffn, warmup=args.warmup, iters=args.iters),
            "packed_gate_up_down_expert_ffn": _bench(packed_ffn, warmup=args.warmup, iters=args.iters),
        },
        "notes": [
            "gate/up are fused in one packed kernel; down uses PackedNVFP4Linear",
            "router/top-k is not included; this isolates one active expert FFN",
        ],
    }
    result["timing_ms"]["speedup"] = (
        result["timing_ms"]["resident_expert_ffn"]
        / result["timing_ms"]["packed_gate_up_down_expert_ffn"]
    )
    c = result["comparison"]
    result["verdict"] = "PASS" if c["cosine"] > 0.999 and c["rel_l2"] < 0.01 else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

