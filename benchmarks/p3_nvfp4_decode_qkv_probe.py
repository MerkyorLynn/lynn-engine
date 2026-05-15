#!/usr/bin/env python3
"""P3-C/D probe: route linear-attention decode projections through PackedNVFP4Linear."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.incremental_decode import decode_linear_attn
from engine.loader import load_qwen36_layer
from engine.nvfp4_runtime import PackedNVFP4Linear
from engine.qwen36_linear_attn_block import CONV_KERNEL, HEAD_K_DIM, HEAD_V_DIM, NUM_V_HEADS


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
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
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


def _load_packed_linear(v8_dir: Path, layer: int, short_name: str, device: str) -> PackedNVFP4Linear:
    base = f"model.language_model.layers.{layer}.{short_name.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(st, base, name=short_name, device=device)


def _make_inputs(device: str, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    h_new = torch.randn(1, 1, 2048, device=device, dtype=dtype, generator=gen)
    recurrent_state = (
        torch.randn(
            1,
            NUM_V_HEADS,
            HEAD_K_DIM,
            HEAD_V_DIM,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.01
    )
    conv_state = torch.randn(1, 8192, CONV_KERNEL - 1, device=device, dtype=dtype, generator=gen) * 0.01
    return h_new, recurrent_state, conv_state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--replace",
        choices=["qkv", "all-linear-attn"],
        default="qkv",
        help="Which linear-attention decode projections to replace with PackedNVFP4Linear.",
    )
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)

    resident_w, cfg = load_qwen36_layer(
        str(v8_dir),
        args.layer,
        num_experts=256,
        device=args.device,
        dequant_dtype=dtype,
    )
    packed_w = copy.copy(resident_w)
    replace_names = ["linear_attn.in_proj_qkv.weight"]
    if args.replace == "all-linear-attn":
        replace_names = [
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.out_proj.weight",
        ]
    for name in replace_names:
        packed_w[name] = _load_packed_linear(v8_dir, args.layer, name, args.device)

    h_new, recurrent_state, conv_state = _make_inputs(args.device, dtype, args.seed)
    ref_out, ref_state, ref_conv = decode_linear_attn(
        h_new,
        resident_w,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    packed_out, packed_state, packed_conv = decode_linear_attn(
        h_new,
        packed_w,
        recurrent_state.clone(),
        conv_state.clone(),
    )
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-nvfp4-decode-qkv-probe-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "replace_mode": args.replace,
        "replaced": replace_names,
        "comparisons": {
            "decode_output": _compare(packed_out, ref_out),
            "recurrent_state": _compare(packed_state, ref_state),
            "conv_state": _compare(packed_conv, ref_conv),
        },
        "timing_ms": {
            "resident_decode_linear_attn": _bench(
                lambda: decode_linear_attn(h_new, resident_w, recurrent_state.clone(), conv_state.clone())[0],
                warmup=args.warmup,
                iters=args.iters,
            ),
            "packed_qkv_decode_linear_attn": _bench(
                lambda: decode_linear_attn(h_new, packed_w, recurrent_state.clone(), conv_state.clone())[0],
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "notes": [
            "Selected linear-attention decode projections are routed through PackedNVFP4Linear",
            "all non-selected projections use resident dequantized tensors",
        ],
    }
    c = result["comparisons"]["decode_output"]
    result["verdict"] = "PASS" if c["cosine"] > 0.999 and c["rel_l2"] < 0.01 else "FAIL"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
