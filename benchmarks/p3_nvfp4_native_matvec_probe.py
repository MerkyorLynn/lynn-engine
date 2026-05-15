#!/usr/bin/env python3
"""P3-A probe: packed NVFP4 matvec without materializing full weights."""
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

from engine.dequant import dequantize_nvfp4_v8_rtn_weight
from triton_kernels.nvfp4_linear import nvfp4_matvec_packed


DEFAULT_TENSORS = [
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
    "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
]


def _load_v8_group(model_dir: Path, name: str) -> dict[str, torch.Tensor]:
    base = name.removesuffix(".weight")
    keys = {
        "weight_packed": base + ".weight_packed",
        "weight_scale": base + ".weight_scale",
        "weight_global_scale": base + ".weight_global_scale",
    }
    out: dict[str, torch.Tensor] = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as st:
        for label, key in keys.items():
            out[label] = st.get_tensor(key)
    return out


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


def _bench(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
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


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}


def _run_one(
    *,
    v8_dir: Path,
    tensor: str,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
    block_m: int,
    block_n: int,
) -> dict[str, Any]:
    group_cpu = _load_v8_group(v8_dir, tensor)
    packed = group_cpu["weight_packed"].to(device=device, non_blocking=True).contiguous()
    # Convert scales once at load time. This avoids full weight materialization
    # while keeping the kernel focused on packed FP4 weight decode.
    scale = group_cpu["weight_scale"].to(device=device, non_blocking=True).float().contiguous()
    global_scale = group_cpu["weight_global_scale"].to(device=device, non_blocking=True).float().contiguous()

    in_features = packed.shape[1] * 2
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(in_features, device=device, generator=generator, dtype=torch.bfloat16)

    # Reference path keeps a resident dequantized tensor only for comparison.
    reference_weight = dequantize_nvfp4_v8_rtn_weight(
        packed,
        scale,
        global_scale,
        output_dtype=torch.float32,
    ).contiguous()
    reference = F.linear(x.float().unsqueeze(0), reference_weight).squeeze(0)

    packed_out = nvfp4_matvec_packed(
        x,
        packed,
        scale,
        global_scale,
        block_m=block_m,
        block_n=block_n,
    )
    torch.cuda.synchronize()
    comparison = _compare(packed_out, reference)

    packed_ms = _bench(
        lambda: nvfp4_matvec_packed(
            x,
            packed,
            scale,
            global_scale,
            block_m=block_m,
            block_n=block_n,
        ),
        warmup=warmup,
        iters=iters,
    )
    resident_dequant_ms = _bench(
        lambda: F.linear(x.float().unsqueeze(0), reference_weight).squeeze(0),
        warmup=warmup,
        iters=iters,
    )
    materialize_each_call_ms = _bench(
        lambda: F.linear(
            x.float().unsqueeze(0),
            dequantize_nvfp4_v8_rtn_weight(
                packed,
                scale,
                global_scale,
                output_dtype=torch.float32,
            ),
        ).squeeze(0),
        warmup=1,
        iters=max(3, min(iters, 10)),
    )
    verdict = "PASS" if comparison["cosine"] > 0.999 and comparison["rel_l2"] < 0.01 else "FAIL"
    return {
        "tensor": tensor,
        "inputs": {
            "x": _tensor_meta(x),
            "weight_packed": _tensor_meta(packed),
            "weight_scale": _tensor_meta(scale),
            "weight_global_scale": _tensor_meta(global_scale),
            "block_m": block_m,
            "block_n": block_n,
        },
        "comparison": comparison,
        "timing_ms": {
            "packed_nvfp4_kernel": packed_ms,
            "resident_dequantized_linear": resident_dequant_ms,
            "materialize_dequant_each_call": materialize_each_call_ms,
        },
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8", required=True, help="NVFP4 v8-RTN checkpoint dir")
    ap.add_argument(
        "--tensor",
        action="append",
        default=None,
        help="Tensor to probe. May be passed multiple times. Defaults to representative layer0 QKV/expert tensors.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=128)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tensor_names = args.tensor or DEFAULT_TENSORS
    tensor_results = [
        _run_one(
            v8_dir=Path(args.v8),
            tensor=tensor,
            device=device,
            seed=args.seed + idx,
            warmup=args.warmup,
            iters=args.iters,
            block_m=args.block_m,
            block_n=args.block_n,
        )
        for idx, tensor in enumerate(tensor_names)
    ]

    result: dict[str, Any] = {
        "schema_version": "lynn-engine-p3-native-nvfp4-matvec-probe-v1",
        "tensors": tensor_names,
        "v8_model": str(Path(args.v8)),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "results": tensor_results,
        "notes": [
            "packed_nvfp4_kernel consumes uint8 weight_packed directly and does not materialize full BF16 weights",
            "resident_dequantized_linear is the P2 slow-path baseline with full weight materialized once",
            "this is a single-token matvec bridge, not the final tensor-core FP4 GEMM",
        ],
    }
    result["verdict"] = "PASS" if all(x["verdict"] == "PASS" for x in tensor_results) else "FAIL"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
