#!/usr/bin/env python3
"""P5-D probe: linear-attention decode with opt-in native FP4 fastpath projections.

P5-D asks whether the speed-gated `native_fast_2d` path remains correct when
threaded through the real `decode_linear_attn` function.

This probe does not change engine defaults. It sets selected packed projection
objects to `default_backend=native_fast_2d`, then passes those objects through
the existing decode_linear_attn dispatch.
"""
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

from engine.incremental_decode import decode_linear_attn  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from engine.nvfp4_runtime import PackedNVFP4Linear  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    CONV_KERNEL,
    HEAD_K_DIM,
    HEAD_V_DIM,
    NUM_V_HEADS,
)


LINEAR_ATTN_WEIGHT_NAMES = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
]

REPLACE_SETS = {
    "qkv": ["linear_attn.in_proj_qkv.weight"],
    "z": ["linear_attn.in_proj_z.weight"],
    "b": ["linear_attn.in_proj_b.weight"],
    "a": ["linear_attn.in_proj_a.weight"],
    "out": ["linear_attn.out_proj.weight"],
    "all-linear-attn": LINEAR_ATTN_WEIGHT_NAMES,
}


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


def _load_packed_linear(
    v8_dir: Path,
    layer: int,
    short_name: str,
    device: str,
    *,
    native: bool,
) -> PackedNVFP4Linear:
    base_key = f"model.language_model.layers.{layer}.{short_name.removesuffix('.weight')}"
    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        return PackedNVFP4Linear.from_safetensors(
            st,
            base_key,
            name=short_name,
            device=device,
            default_backend="native_fast_2d" if native else "scalar_bridge",
        )


def _with_packed_weights(
    resident_w: dict[str, Any],
    v8_dir: Path,
    layer: int,
    names: list[str],
    device: str,
    *,
    native: bool,
) -> dict[str, Any]:
    weights = copy.copy(resident_w)
    for name in names:
        weights[name] = _load_packed_linear(v8_dir, layer, name, device, native=native)
    return weights


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
        choices=sorted(REPLACE_SETS),
        default="all-linear-attn",
        help="Which linear-attention decode projections to route through packed weights.",
    )
    ap.add_argument("--native-cosine-threshold", type=float, default=0.98)
    ap.add_argument("--native-rel-l2-threshold", type=float, default=0.25)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)
    replace_names = REPLACE_SETS[args.replace]

    resident_w, _ = load_qwen36_layer(
        str(v8_dir),
        args.layer,
        num_experts=256,
        device=args.device,
        dequant_dtype=dtype,
    )
    scalar_w = _with_packed_weights(
        resident_w,
        v8_dir,
        args.layer,
        replace_names,
        args.device,
        native=False,
    )
    native_w = _with_packed_weights(
        resident_w,
        v8_dir,
        args.layer,
        replace_names,
        args.device,
        native=True,
    )

    h_new, recurrent_state, conv_state = _make_inputs(args.device, dtype, args.seed)
    resident_out, resident_state, resident_conv = decode_linear_attn(
        h_new, resident_w, recurrent_state.clone(), conv_state.clone()
    )
    scalar_out, scalar_state, scalar_conv = decode_linear_attn(
        h_new, scalar_w, recurrent_state.clone(), conv_state.clone()
    )
    native_out, native_state, native_conv = decode_linear_attn(
        h_new, native_w, recurrent_state.clone(), conv_state.clone()
    )
    torch.cuda.synchronize()

    native_vs_scalar = {
        "decode_output": _compare(native_out, scalar_out),
        "recurrent_state": _compare(native_state, scalar_state),
        "conv_state": _compare(native_conv, scalar_conv),
    }
    native_vs_resident = {
        "decode_output": _compare(native_out, resident_out),
        "recurrent_state": _compare(native_state, resident_state),
        "conv_state": _compare(native_conv, resident_conv),
    }
    scalar_vs_resident = {
        "decode_output": _compare(scalar_out, resident_out),
        "recurrent_state": _compare(scalar_state, resident_state),
        "conv_state": _compare(scalar_conv, resident_conv),
    }

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p5d-native-fp4-fastpath-decode-projection-gate-v1",
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "replace_mode": args.replace,
        "replaced": replace_names,
        "device": {
            "name": torch.cuda.get_device_name(args.device),
            "capability": list(torch.cuda.get_device_capability(args.device)),
        },
        "comparisons": {
            "native_vs_scalar_bridge": native_vs_scalar,
            "native_vs_resident_dequant": native_vs_resident,
            "scalar_bridge_vs_resident_dequant": scalar_vs_resident,
        },
        "timing_ms": {
            "resident_decode_linear_attn": _bench(
                lambda: decode_linear_attn(h_new, resident_w, recurrent_state.clone(), conv_state.clone())[0],
                warmup=args.warmup,
                iters=args.iters,
            ),
            "scalar_bridge_decode_linear_attn": _bench(
                lambda: decode_linear_attn(h_new, scalar_w, recurrent_state.clone(), conv_state.clone())[0],
                warmup=args.warmup,
                iters=args.iters,
            ),
            "native_fast_2d_decode_linear_attn": _bench(
                lambda: decode_linear_attn(h_new, native_w, recurrent_state.clone(), conv_state.clone())[0],
                warmup=args.warmup,
                iters=args.iters,
            ),
        },
        "thresholds": {
            "native_cosine_threshold": args.native_cosine_threshold,
            "native_rel_l2_threshold": args.native_rel_l2_threshold,
        },
        "notes": [
            "P5-D is an opt-in fastpath drift/speed gate; engine defaults still use scalar_bridge.",
            "native_fast_2d quantizes activations to FP4, so native_vs_scalar is allowed more drift than P3 scalar bridge gates.",
        ],
    }
    result["timing_ms"]["native_vs_scalar_speed_ratio"] = (
        result["timing_ms"]["scalar_bridge_decode_linear_attn"]
        / result["timing_ms"]["native_fast_2d_decode_linear_attn"]
    )

    native_cmp = native_vs_scalar["decode_output"]
    result["verdict"] = (
        "PASS"
        if (
            native_cmp["cosine"] >= args.native_cosine_threshold
            and native_cmp["rel_l2"] <= args.native_rel_l2_threshold
        )
        else "FAIL"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
