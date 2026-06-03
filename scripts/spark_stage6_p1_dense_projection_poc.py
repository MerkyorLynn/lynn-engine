#!/usr/bin/env python3
"""Stage 6 Phase 1: single dense projection packed-NVFP4 PoC.

This is the first hard gate for the zero-shadow kernel line. It loads one real
Lynn-native 35B dense projection from the quant manifest, runs the existing
packed-NVFP4 Triton matvec directly from uint8 E2M1 + checkpoint-native scale
storage, and compares it against explicit dequant references.

It deliberately does not construct LynnIncrementalRunner or materialize the full
35B BF16 shadow. The timed packed kernel runs after the BF16 reference tensors
are deleted, so the report can distinguish valid packed execution from hidden
BF16-shadow reads.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.dequant import dequantize_nvfp4_v8_rtn_weight  # noqa: E402
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed  # noqa: E402


DEFAULT_MODEL = "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
DEFAULT_BASE_KEY = "model.language_model.layers.0.linear_attn.in_proj_qkv"
GIB = 1024**3


def _read_weight_map(model_dir: Path) -> dict[str, str]:
    return json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str, *, device: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(model_dir / weight_map[key], framework="pt", device=device) as st:
        return st.get_tensor(key)


def _load_projection(model_dir: Path, base_key: str, *, device: str) -> dict[str, Any]:
    manifest = json.loads((model_dir / "lynn_quant_manifest.json").read_text())
    rec = manifest["quantized_tensors"].get(base_key + ".weight")
    if rec is None:
        raise KeyError(f"{base_key}.weight is not quantized in lynn_quant_manifest.json")
    weight_map = _read_weight_map(model_dir)
    packed = _load_tensor(model_dir, weight_map, rec["packed_key"], device=device).contiguous()
    scale = _load_tensor(model_dir, weight_map, rec["scale_key"], device=device).contiguous()
    global_scale = _load_tensor(model_dir, weight_map, rec["global_scale_key"], device=device).contiguous()
    return {
        "record": rec,
        "packed": packed,
        "scale": scale,
        "global_scale": global_scale,
    }


def _nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().flatten().double()
    bf = b.detach().float().flatten().double()
    diff = af - bf
    an = af.norm()
    bn = bf.norm()
    dn = diff.norm()
    cos = (af * bf).sum() / (an * bn + 1e-30)
    argmax_a = int(torch.argmax(af).item())
    argmax_b = int(torch.argmax(bf).item())
    return {
        "cosine": float(cos.item()),
        "rel_l2": float((dn / (bn + 1e-30)).item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "argmax_a": argmax_a,
        "argmax_b": argmax_b,
        "argmax_match": argmax_a == argmax_b,
    }


def _bench_cuda(fn, *, warmup: int, iters: int, repeats: int) -> dict[str, Any]:
    times_us: list[float] = []
    for _ in range(repeats):
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
        times_us.append(float(start.elapsed_time(end) * 1000.0 / iters))
    return {
        "warmup": warmup,
        "iters": iters,
        "repeats": repeats,
        "median_us": float(statistics.median(times_us)),
        "min_us": float(min(times_us)),
        "max_us": float(max(times_us)),
        "all_us": times_us,
    }


def _fmt_bytes(n: int) -> str:
    if n >= 1024**2:
        return f"{n / 1024**2:.2f} MiB"
    if n >= 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n} B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-key", default=DEFAULT_BASE_KEY)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda"
    model_dir = Path(args.model)
    torch.manual_seed(args.seed)

    loaded = _load_projection(model_dir, args.base_key, device=device)
    rec = loaded["record"]
    packed: torch.Tensor = loaded["packed"]
    scale: torch.Tensor = loaded["scale"]
    global_scale: torch.Tensor = loaded["global_scale"]
    out_features, in_features = map(int, rec["original_shape"])
    if list(packed.shape) != [out_features, in_features // 2]:
        raise ValueError(f"unexpected packed shape {tuple(packed.shape)} for original {rec['original_shape']}")
    if list(scale.shape) != [out_features, in_features // 16]:
        raise ValueError(f"unexpected scale shape {tuple(scale.shape)} for original {rec['original_shape']}")

    x = (torch.randn((in_features,), device=device, dtype=torch.float32) * 0.35).to(torch.bfloat16)
    torch.cuda.synchronize()

    bf16_shadow_bytes = out_features * in_features * 2
    packed_weight_bytes = _nbytes(packed) + _nbytes(scale) + _nbytes(global_scale)
    timed_arg_bytes = packed_weight_bytes + _nbytes(x) + out_features * 4

    print("=============== STAGE 6 PHASE 1 DENSE PROJECTION POC ===============", flush=True)
    print(f"model          : {model_dir}", flush=True)
    print(f"base_key       : {args.base_key}", flush=True)
    print(f"shape          : out={out_features} in={in_features}", flush=True)
    print(f"packed dtype   : {packed.dtype} shape={tuple(packed.shape)} bytes={_fmt_bytes(_nbytes(packed))}", flush=True)
    print(f"scale dtype    : {scale.dtype} shape={tuple(scale.shape)} bytes={_fmt_bytes(_nbytes(scale))}", flush=True)
    print(f"global dtype   : {global_scale.dtype} shape={tuple(global_scale.shape)} bytes={_fmt_bytes(_nbytes(global_scale))}", flush=True)
    print(f"BF16 shadow    : {_fmt_bytes(bf16_shadow_bytes)}", flush=True)
    print(f"packed+scale   : {_fmt_bytes(packed_weight_bytes)}", flush=True)

    # Build explicit references. This is outside the packed-kernel memory proof.
    w_fp32 = dequantize_nvfp4_v8_rtn_weight(
        packed,
        scale,
        global_scale,
        output_dtype=torch.float32,
    )
    ref_fp32 = torch.mv(w_fp32, x.float())
    w_bf16 = w_fp32.to(torch.bfloat16)
    ref_bf16_shadow = F.linear(x.reshape(1, -1), w_bf16).reshape(-1).float()
    y = nvfp4_matvec_packed(
        x,
        packed,
        scale,
        global_scale,
        block_m=args.block_m,
        block_n=args.block_n,
    )
    torch.cuda.synchronize()

    diff_fp32 = _diff_stats(y, ref_fp32)
    diff_bf16 = _diff_stats(y, ref_bf16_shadow)
    print(
        "[numeric] vs fp32-dequant "
        f"cos={diff_fp32['cosine']:.9f} rel_l2={diff_fp32['rel_l2']:.3e} "
        f"max_abs={diff_fp32['max_abs']:.3e} argmax={diff_fp32['argmax_match']}",
        flush=True,
    )
    print(
        "[numeric] vs bf16-shadow  "
        f"cos={diff_bf16['cosine']:.9f} rel_l2={diff_bf16['rel_l2']:.3e} "
        f"max_abs={diff_bf16['max_abs']:.3e} argmax={diff_bf16['argmax_match']}",
        flush=True,
    )

    bench_bf16 = _bench_cuda(
        lambda: F.linear(x.reshape(1, -1), w_bf16).reshape(-1),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )

    # Delete every wide reference before timing packed execution. The packed
    # benchmark below cannot read a resident BF16 shadow because none remains.
    del w_fp32, w_bf16, ref_fp32, ref_bf16_shadow, y
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_packed_alloc = int(torch.cuda.memory_allocated())
    bench_packed = _bench_cuda(
        lambda: nvfp4_matvec_packed(
            x,
            packed,
            scale,
            global_scale,
            block_m=args.block_m,
            block_n=args.block_n,
        ),
        warmup=args.warmup,
        iters=args.iters,
        repeats=args.repeats,
    )
    after_packed_alloc = int(torch.cuda.memory_allocated())
    packed_peak_alloc = int(torch.cuda.max_memory_allocated())

    speedup = bench_bf16["median_us"] / bench_packed["median_us"] if bench_packed["median_us"] else math.nan
    byte_reduction = bf16_shadow_bytes / packed_weight_bytes if packed_weight_bytes else math.nan
    print(
        f"[bench] packed median={bench_packed['median_us']:.2f} us  "
        f"bf16 median={bench_bf16['median_us']:.2f} us  speedup={speedup:.3f}x",
        flush=True,
    )
    print(
        f"[bytes] BF16/packed_weight={byte_reduction:.3f}x  timed_arg_bytes={_fmt_bytes(timed_arg_bytes)}",
        flush=True,
    )
    print(
        "[memory] packed bench after deleting BF16 refs: "
        f"before={before_packed_alloc / GIB:.4f} GiB "
        f"after={after_packed_alloc / GIB:.4f} GiB "
        f"peak={packed_peak_alloc / GIB:.4f} GiB",
        flush=True,
    )

    numeric_pass = (
        diff_fp32["cosine"] >= 0.99999
        and diff_fp32["rel_l2"] <= 2.0e-3
        and diff_fp32["argmax_match"]
    )
    no_shadow_pass = packed_peak_alloc < bf16_shadow_bytes
    perf_pass = speedup >= 1.0
    all_pass = bool(numeric_pass and no_shadow_pass and perf_pass)
    result = {
        "schema": "lynn-stage6-phase1-dense-projection-poc-v1",
        "model": str(model_dir),
        "base_key": args.base_key,
        "seed": args.seed,
        "shape": {"out_features": out_features, "in_features": in_features},
        "dtypes": {
            "packed": str(packed.dtype),
            "scale": str(scale.dtype),
            "global_scale": str(global_scale.dtype),
            "activation": str(x.dtype),
            "kernel_output": "torch.float32",
        },
        "bytes": {
            "bf16_shadow": bf16_shadow_bytes,
            "packed": _nbytes(packed),
            "scale": _nbytes(scale),
            "global_scale": _nbytes(global_scale),
            "packed_weight_total": packed_weight_bytes,
            "activation": _nbytes(x),
            "output_fp32": out_features * 4,
            "timed_arg_total": timed_arg_bytes,
            "bf16_to_packed_weight_ratio": byte_reduction,
        },
        "numeric": {
            "vs_fp32_dequant": diff_fp32,
            "vs_bf16_shadow": diff_bf16,
        },
        "bench": {
            "packed_triton": bench_packed,
            "bf16_shadow_linear": bench_bf16,
            "speedup_vs_bf16": speedup,
        },
        "memory_after_deleting_bf16_refs": {
            "before_packed_bench_gib": before_packed_alloc / GIB,
            "after_packed_bench_gib": after_packed_alloc / GIB,
            "peak_packed_bench_gib": packed_peak_alloc / GIB,
        },
        "passes": {
            "numeric": bool(numeric_pass),
            "no_bf16_shadow_allocated_for_packed_bench": bool(no_shadow_pass),
            "perf_speedup_vs_bf16": bool(perf_pass),
            "all": all_pass,
        },
        "notes": [
            "Packed timing runs after deleting FP32/BF16 reference weights.",
            "This is M=1 matvec only; batched prefill kernels remain Phase 2 work.",
        ],
    }

    print("=============== RESULT JSON ===============", flush=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not numeric_pass or not no_shadow_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
