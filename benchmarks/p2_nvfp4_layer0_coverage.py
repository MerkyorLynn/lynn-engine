#!/usr/bin/env python3
"""P2-A3 gate: cover representative layer-0 NVFP4 linears.

This expands the single-tensor P2 gate into:

- all linear-attention packed linears in layer 0,
- shared expert linears,
- representative MoE experts mapped back to Lynn's fused BF16 oracle layout.

It is still a slow CPU correctness gate, not a runtime benchmark.
"""
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
from engine.loader import load_qwen36_layer


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
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    pearson = torch.dot(centered_a, centered_b) / (
        torch.linalg.vector_norm(centered_a).clamp_min(1e-12)
        * torch.linalg.vector_norm(centered_b).clamp_min(1e-12)
    )
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(cosine),
        "pearson": float(pearson),
        "sign_match": float((torch.signbit(af) == torch.signbit(bf)).float().mean()),
    }


def _load_v8_group(st, full_weight_name: str) -> dict[str, torch.Tensor]:
    base = full_weight_name.removesuffix(".weight")
    return {
        "weight_packed": st.get_tensor(base + ".weight_packed"),
        "weight_scale": st.get_tensor(base + ".weight_scale"),
        "weight_global_scale": st.get_tensor(base + ".weight_global_scale"),
        "input_global_scale": st.get_tensor(base + ".input_global_scale"),
    }


def _dequant_group(st, full_weight_name: str) -> torch.Tensor:
    group = _load_v8_group(st, full_weight_name)
    return dequantize_nvfp4_v8_rtn_weight(
        group["weight_packed"],
        group["weight_scale"],
        group["weight_global_scale"],
        output_dtype=torch.bfloat16,
    )


def _forward_compare(
    oracle_weight: torch.Tensor,
    dequant_weight: torch.Tensor,
    *,
    tokens: int,
    seed: int,
) -> dict[str, Any]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(tokens, oracle_weight.shape[1], generator=gen, dtype=torch.float32)
    oracle_out = F.linear(x, oracle_weight.float())
    dequant_out = F.linear(x, dequant_weight.float())
    return {
        "input_shape": list(x.shape),
        "output_shape": list(oracle_out.shape),
        "comparison": _compare(dequant_out, oracle_out),
    }


def _item(
    label: str,
    full_name: str,
    oracle: torch.Tensor,
    st,
    *,
    tokens: int,
    seed: int,
) -> dict[str, Any]:
    dequant = _dequant_group(st, full_name)
    weight_cmp = _compare(dequant, oracle)
    forward = _forward_compare(oracle, dequant, tokens=tokens, seed=seed)
    return {
        "label": label,
        "full_name": full_name,
        "oracle": _stats(oracle),
        "dequant": _stats(dequant),
        "weight_compare": weight_cmp,
        "linear_forward": forward,
        "verdict": (
            "PASS"
            if weight_cmp["cosine"] > 0.97
            and forward["comparison"]["cosine"] > 0.97
            else "FAIL"
        ),
    }


def _infer_gate_up_order(bf16_layer: dict[str, torch.Tensor], st, expert: int) -> dict[str, Any]:
    fused = bf16_layer["mlp.experts.gate_up_proj"][expert]
    half = fused.shape[0] // 2
    first, second = fused[:half], fused[half:]
    prefix = f"model.language_model.layers.0.mlp.experts.{expert}"
    gate = _dequant_group(st, prefix + ".gate_proj.weight")
    up = _dequant_group(st, prefix + ".up_proj.weight")
    gate_first = (
        _compare(gate, first)["cosine"] + _compare(up, second)["cosine"]
    ) / 2.0
    up_first = (
        _compare(gate, second)["cosine"] + _compare(up, first)["cosine"]
    ) / 2.0
    order = "gate_up" if gate_first >= up_first else "up_gate"
    return {
        "expert": expert,
        "gate_first_score": gate_first,
        "up_first_score": up_first,
        "order": order,
    }


def _expert_oracle(
    bf16_layer: dict[str, torch.Tensor],
    expert: int,
    proj: str,
    gate_up_order: str,
) -> torch.Tensor:
    if proj == "down_proj":
        return bf16_layer["mlp.experts.down_proj"][expert]
    fused = bf16_layer["mlp.experts.gate_up_proj"][expert]
    half = fused.shape[0] // 2
    if gate_up_order == "gate_up":
        return fused[:half] if proj == "gate_proj" else fused[half:]
    return fused[half:] if proj == "gate_proj" else fused[:half]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", default="0,1,42,127,255")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260514)
    args = ap.parse_args()

    bf16_dir = Path(args.bf16)
    v8_dir = Path(args.v8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    experts = [int(x) for x in args.experts.split(",") if x.strip()]

    bf16_layer, cfg = load_qwen36_layer(str(bf16_dir), 0, device="cpu")
    results: list[dict[str, Any]] = []
    gate_up_order_probe = None

    with safe_open(v8_dir / "model.safetensors", framework="pt", device="cpu") as st:
        direct = [
            ("linear_attn.in_proj_a", "linear_attn.in_proj_a.weight"),
            ("linear_attn.in_proj_b", "linear_attn.in_proj_b.weight"),
            ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkv.weight"),
            ("linear_attn.in_proj_z", "linear_attn.in_proj_z.weight"),
            ("linear_attn.out_proj", "linear_attn.out_proj.weight"),
            ("mlp.shared_expert.gate_proj", "mlp.shared_expert.gate_proj.weight"),
            ("mlp.shared_expert.up_proj", "mlp.shared_expert.up_proj.weight"),
            ("mlp.shared_expert.down_proj", "mlp.shared_expert.down_proj.weight"),
        ]
        for label, short_key in direct:
            full = "model.language_model.layers.0." + short_key
            results.append(
                _item(
                    label,
                    full,
                    bf16_layer[short_key],
                    st,
                    tokens=args.tokens,
                    seed=args.seed,
                )
            )

        gate_up_order_probe = _infer_gate_up_order(bf16_layer, st, experts[0])
        gate_up_order = gate_up_order_probe["order"]
        for expert in experts:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                full = f"model.language_model.layers.0.mlp.experts.{expert}.{proj}.weight"
                oracle = _expert_oracle(bf16_layer, expert, proj, gate_up_order)
                results.append(
                    _item(
                        f"mlp.experts.{expert}.{proj}",
                        full,
                        oracle,
                        st,
                        tokens=args.tokens,
                        seed=args.seed + expert,
                    )
                )

    failures = [x for x in results if x["verdict"] != "PASS"]
    summary = {
        "schema_version": "lynn-engine-p2-layer0-coverage-v1",
        "bf16_model": str(bf16_dir),
        "v8_model": str(v8_dir),
        "layer": 0,
        "experts": experts,
        "bf16_loader_config": cfg,
        "gate_up_order_probe": gate_up_order_probe,
        "num_items": len(results),
        "num_pass": len(results) - len(failures),
        "num_fail": len(failures),
        "min_weight_cosine": min(x["weight_compare"]["cosine"] for x in results),
        "min_forward_cosine": min(
            x["linear_forward"]["comparison"]["cosine"] for x in results
        ),
        "results": results,
        "verdict": "PASS" if not failures else "FAIL",
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

