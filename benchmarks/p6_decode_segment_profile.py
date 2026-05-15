#!/usr/bin/env python3
"""P6-A: segment-level profile for one-token linear-attention decode.

P5-D proved isolated native FP4 projection replacement is not enough. This
script profiles the real decode function by segments so P6 can optimize the
dominant decode-loop cost rather than chase individual projection wins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _linear, _recurrent_gated_delta_rule  # noqa: E402
try:
    from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare  # noqa: E402
except Exception:  # pragma: no cover
    recurrent_gated_delta_fused_prepare = None
from engine.loader import load_qwen36_layer  # noqa: E402
from engine.nvfp4_runtime import PackedNVFP4Linear  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    CONV_KERNEL,
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    RMS_EPS,
    VALUE_DIM,
    V_PER_K,
    rms_norm_gated,
)


LINEAR_ATTN_WEIGHT_NAMES = [
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
]


def _bench(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
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
        return PackedNVFP4Linear.from_safetensors(
            st,
            base,
            name=short_name,
            device=device,
            default_backend="scalar_bridge",
        )


def _with_packed_linear_attn(resident_w: dict[str, Any], v8_dir: Path, layer: int, device: str) -> dict[str, Any]:
    weights = dict(resident_w)
    for name in LINEAR_ATTN_WEIGHT_NAMES:
        weights[name] = _load_packed_linear(v8_dir, layer, name, device)
    return weights


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


def _run_recurrent(q, k, v, g, beta, recurrent_state, backend: str):
    if backend == "torch":
        return _recurrent_gated_delta_rule(q, k, v, g, beta, recurrent_state)
    if backend == "triton_fused_prepare":
        if recurrent_gated_delta_fused_prepare is None:
            raise RuntimeError("triton_fused_prepare requested but kernel is unavailable")
        return recurrent_gated_delta_fused_prepare(q, k, v, g, beta, recurrent_state)
    raise ValueError(f"unknown recurrent_backend={backend!r}")


def _prepare_intermediates(h_new: torch.Tensor, w: dict[str, Any], recurrent_state: torch.Tensor, conv_state: torch.Tensor, recurrent_backend: str) -> dict[str, Any]:
    bsz = h_new.shape[0]
    mixed_new = _linear(h_new, w["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)
    conv_input = torch.cat([conv_state, mixed_new], dim=-1)
    out_conv = F.conv1d(
        conv_input,
        w["linear_attn.conv1d.weight"],
        bias=None,
        padding=0,
        groups=mixed_new.shape[1],
    )
    out_conv = F.silu(out_conv).transpose(1, 2)
    q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(bsz, 1, NUM_V_HEADS, HEAD_V_DIM)
    z = _linear(h_new, w["linear_attn.in_proj_z.weight"]).reshape(bsz, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = _linear(h_new, w["linear_attn.in_proj_b.weight"]).sigmoid()
    a = _linear(h_new, w["linear_attn.in_proj_a.weight"])
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"].float())
    if V_PER_K > 1:
        q = q.repeat_interleave(V_PER_K, dim=2)
        k = k.repeat_interleave(V_PER_K, dim=2)
    core_attn_out, new_state = _run_recurrent(q, k, v, g, beta, recurrent_state, recurrent_backend)
    flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
    flat_z = z.reshape(-1, HEAD_V_DIM)
    normed = rms_norm_gated(flat_x, w["linear_attn.norm.weight"], flat_z, eps=RMS_EPS)
    core_normed = normed.reshape(bsz, 1, NUM_V_HEADS * HEAD_V_DIM)
    return {
        "mixed_new": mixed_new,
        "conv_input": conv_input,
        "out_conv": out_conv,
        "q": q,
        "k": k,
        "v": v,
        "z": z,
        "beta": beta,
        "g": g,
        "core_attn_out": core_attn_out,
        "new_state": new_state,
        "core_normed": core_normed,
    }


def _profile_case(
    *,
    name: str,
    w: dict[str, Any],
    h_new: torch.Tensor,
    recurrent_state: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_backend: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    bsz = h_new.shape[0]
    mid = _prepare_intermediates(h_new, w, recurrent_state, conv_state, recurrent_backend)

    def qkv():
        return _linear(h_new, w["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)

    def conv():
        mixed_new = mid["mixed_new"]
        conv_input = torch.cat([conv_state, mixed_new], dim=-1)
        out_conv = F.conv1d(
            conv_input,
            w["linear_attn.conv1d.weight"],
            bias=None,
            padding=0,
            groups=mixed_new.shape[1],
        )
        return F.silu(out_conv).transpose(1, 2), conv_input[:, :, 1:].contiguous()

    def split_reshape():
        q, k, v = torch.split(mid["out_conv"], [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(bsz, 1, NUM_V_HEADS, HEAD_V_DIM)
        if V_PER_K > 1:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)
        return q, k, v

    def z_proj():
        return _linear(h_new, w["linear_attn.in_proj_z.weight"]).reshape(bsz, 1, NUM_V_HEADS, HEAD_V_DIM)

    def b_proj():
        return _linear(h_new, w["linear_attn.in_proj_b.weight"]).sigmoid()

    def a_g_proj():
        a = _linear(h_new, w["linear_attn.in_proj_a.weight"])
        return -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"].float())

    def recurrent():
        return _run_recurrent(mid["q"], mid["k"], mid["v"], mid["g"], mid["beta"], recurrent_state, recurrent_backend)

    def norm():
        flat_x = mid["core_attn_out"].reshape(-1, HEAD_V_DIM)
        flat_z = mid["z"].reshape(-1, HEAD_V_DIM)
        y = rms_norm_gated(flat_x, w["linear_attn.norm.weight"], flat_z, eps=RMS_EPS)
        return y.reshape(bsz, 1, NUM_V_HEADS * HEAD_V_DIM)

    def out_proj():
        return _linear(mid["core_normed"], w["linear_attn.out_proj.weight"])

    def full_decode():
        mixed_new = qkv()
        conv_input = torch.cat([conv_state, mixed_new], dim=-1)
        out_conv = F.conv1d(
            conv_input,
            w["linear_attn.conv1d.weight"],
            bias=None,
            padding=0,
            groups=mixed_new.shape[1],
        )
        out_conv = F.silu(out_conv).transpose(1, 2)
        q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = q.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
        k = k.reshape(bsz, 1, NUM_K_HEADS, HEAD_K_DIM)
        v = v.reshape(bsz, 1, NUM_V_HEADS, HEAD_V_DIM)
        z = z_proj()
        beta = b_proj()
        g = a_g_proj()
        if V_PER_K > 1:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)
        core_attn_out, _ = _run_recurrent(q, k, v, g, beta, recurrent_state, recurrent_backend)
        flat_x = core_attn_out.reshape(-1, HEAD_V_DIM)
        flat_z = z.reshape(-1, HEAD_V_DIM)
        y = rms_norm_gated(flat_x, w["linear_attn.norm.weight"], flat_z, eps=RMS_EPS)
        core_normed = y.reshape(bsz, 1, NUM_V_HEADS * HEAD_V_DIM)
        return _linear(core_normed, w["linear_attn.out_proj.weight"])

    segment_fns = {
        "qkv_projection": qkv,
        "conv_update": conv,
        "split_repeat": split_reshape,
        "z_projection": z_proj,
        "b_projection_beta": b_proj,
        "a_projection_g": a_g_proj,
        "recurrent_rule": recurrent,
        "rmsnorm_gated": norm,
        "out_projection": out_proj,
        "full_decode_recomposed": full_decode,
    }
    lat = {key: _bench(fn, warmup=warmup, iters=iters) for key, fn in segment_fns.items()}
    total_known = sum(v for k, v in lat.items() if k != "full_decode_recomposed")
    return {
        "name": name,
        "latency_ms": lat,
        "sum_segments_ms": total_known,
        "segment_over_full_ratio": total_known / lat["full_decode_recomposed"],
        "top_segments": sorted(
            [{"segment": k, "latency_ms": v} for k, v in lat.items() if k != "full_decode_recomposed"],
            key=lambda x: x["latency_ms"],
            reverse=True,
        )[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--seed", type=int, default=20260515)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--recurrent-backend", choices=["torch", "triton_fused_prepare"], default="torch")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    v8_dir = Path(args.v8)
    resident_w, _ = load_qwen36_layer(
        str(v8_dir),
        args.layer,
        num_experts=256,
        device=args.device,
        dequant_dtype=dtype,
    )
    scalar_w = None
    if (v8_dir / "model.safetensors").exists():
        scalar_w = _with_packed_linear_attn(resident_w, v8_dir, args.layer, args.device)
    h_new, recurrent_state, conv_state = _make_inputs(args.device, dtype, args.seed)

    result = {
        "schema_version": "lynn-engine-p6-decode-segment-profile-v1",
        "layer": args.layer,
        "model": str(v8_dir),
        "device": torch.cuda.get_device_name(args.device),
        "cases": [],
    }
    result["cases"].append(
        _profile_case(
            name="resident_dequant",
            w=resident_w,
            h_new=h_new,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            recurrent_backend=args.recurrent_backend,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    if scalar_w is not None:
        result["cases"].append(
            _profile_case(
                name="scalar_packed_linear_attn",
                w=scalar_w,
                h_new=h_new,
                recurrent_state=recurrent_state,
                conv_state=conv_state,
                recurrent_backend=args.recurrent_backend,
                warmup=args.warmup,
                iters=args.iters,
            )
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
