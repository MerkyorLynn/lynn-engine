#!/usr/bin/env python3
"""P2 gate: dequantize one NVFP4 v8-RTN tensor and compare references."""
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


DEFAULT_TENSOR = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"


def _load_bf16_weight(model_dir: Path, name: str) -> torch.Tensor:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    file_name = index["weight_map"][name]
    with safe_open(model_dir / file_name, framework="pt", device="cpu") as st:
        return st.get_tensor(name)


def _load_v8_group(model_dir: Path, name: str) -> dict[str, torch.Tensor]:
    base = name.removesuffix(".weight")
    keys = {
        "weight_packed": base + ".weight_packed",
        "weight_scale": base + ".weight_scale",
        "weight_global_scale": base + ".weight_global_scale",
        "input_global_scale": base + ".input_global_scale",
    }
    out: dict[str, torch.Tensor] = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as st:
        for label, key in keys.items():
            out[label] = st.get_tensor(key)
    return out


def _stats(x: torch.Tensor) -> dict[str, Any]:
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "absmax": float(xf.abs().max()),
        "l2": float(torch.linalg.vector_norm(xf)),
    }


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    denom = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    centered_a = af - af.mean()
    centered_b = bf - bf.mean()
    corr = torch.dot(centered_a, centered_b) / (
        torch.linalg.vector_norm(centered_a).clamp_min(1e-12)
        * torch.linalg.vector_norm(centered_b).clamp_min(1e-12)
    )
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
        "pearson": float(corr),
        "sign_match": float((torch.signbit(af) == torch.signbit(bf)).float().mean()),
    }


def _linear_forward_probe(
    bf16_weight: torch.Tensor,
    dequant_weight: torch.Tensor,
    *,
    hidden_size: int,
    tokens: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(tokens, hidden_size, generator=generator, dtype=torch.float32)

    # CPU BF16 matmul support varies by platform; use FP32 for the correctness
    # probe so the measured delta is from weights, not CPU kernel dtype behavior.
    bf16_out = F.linear(x, bf16_weight.float())
    dequant_out = F.linear(x, dequant_weight.float())
    return {
        "input": {
            "shape": list(x.shape),
            "seed": seed,
            "dtype": str(x.dtype),
        },
        "bf16_output": _stats(bf16_out),
        "dequant_output": _stats(dequant_out),
        "comparison": _compare(dequant_out, bf16_out),
    }


def _reference_dequant_if_available(group: dict[str, torch.Tensor]) -> torch.Tensor | None:
    try:
        from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    except Exception:
        return None

    state = {
        "weight_packed": group["weight_packed"],
        "weight_scale": group["weight_scale"],
        "weight_global_scale": group["weight_global_scale"],
    }
    return NVFP4PackedCompressor.decompress(state, scheme=None)["weight"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="BF16 merged checkpoint dir")
    ap.add_argument("--v8", required=True, help="NVFP4 v8-RTN checkpoint dir")
    ap.add_argument("--tensor", default=DEFAULT_TENSOR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward-tokens", type=int, default=8)
    ap.add_argument("--forward-seed", type=int, default=20260514)
    args = ap.parse_args()

    bf16_dir = Path(args.bf16)
    v8_dir = Path(args.v8)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bf16 = _load_bf16_weight(bf16_dir, args.tensor)
    group = _load_v8_group(v8_dir, args.tensor)
    lynn_dequant = dequantize_nvfp4_v8_rtn_weight(
        group["weight_packed"],
        group["weight_scale"],
        group["weight_global_scale"],
        output_dtype=torch.bfloat16,
    )

    reference = _reference_dequant_if_available(group)
    reference_compare = None
    reference_stats = None
    if reference is not None:
        reference_stats = _stats(reference)
        reference_compare = _compare(lynn_dequant, reference)

    bf16_compare = _compare(lynn_dequant, bf16)
    forward_probe = _linear_forward_probe(
        bf16,
        lynn_dequant,
        hidden_size=bf16.shape[1],
        tokens=args.forward_tokens,
        seed=args.forward_seed,
    )

    result = {
        "schema_version": "lynn-engine-p2-nvfp4-single-tensor-v1",
        "tensor": args.tensor,
        "bf16_model": str(bf16_dir),
        "v8_model": str(v8_dir),
        "group": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in group.items()
        },
        "stats": {
            "bf16_oracle": _stats(bf16),
            "lynn_dequant": _stats(lynn_dequant),
            "compressed_tensors_reference": reference_stats,
        },
        "comparisons": {
            "lynn_vs_compressed_tensors_reference": reference_compare,
            "lynn_dequant_vs_bf16_oracle": bf16_compare,
        },
        "linear_forward_probe": forward_probe,
        "verdicts": {
            "reference_match": (
                reference_compare is not None
                and reference_compare["max_abs"] == 0.0
                and reference_compare["mean_abs"] == 0.0
            ),
            "shape_match_bf16": list(lynn_dequant.shape) == list(bf16.shape),
            "bf16_oracle_correlation_ok": bf16_compare["cosine"] > 0.97,
            "linear_forward_correlation_ok": forward_probe["comparison"]["cosine"]
            > 0.97,
        },
    }
    result["verdict"] = (
        "PASS"
        if result["verdicts"]["shape_match_bf16"]
        and result["verdicts"]["reference_match"]
        and result["verdicts"]["bf16_oracle_correlation_ok"]
        and result["verdicts"]["linear_forward_correlation_ok"]
        else "FAIL"
    )

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
