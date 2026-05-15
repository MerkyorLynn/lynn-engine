#!/usr/bin/env python3
"""P2-B gate: run multiple sequential layers with BF16 and NVFP4 weights.

This extends the layer-0 gate into a small transformer-block chain. It keeps the
same deterministic hidden state for the BF16 oracle and NVFP4 slow-dequant path,
then loads each layer through Lynn engine's public loader and compares the
running hidden state after every layer.

It is intentionally slow and CPU-oriented. The goal is correctness surface area,
not tokens/sec.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward
from engine.loader import load_qwen36_layer


def _cfg_from(model_dir: Path) -> tuple[dict[str, Any], list[str]]:
    config = json.loads((model_dir / "config.json").read_text())
    text_config = config["text_config"]
    rope = text_config.get("rope_parameters", {})
    return {
        "hidden_size": text_config["hidden_size"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "head_dim": text_config["head_dim"],
        "num_experts": text_config["num_experts"],
        "num_experts_per_tok": text_config["num_experts_per_tok"],
        "rope_theta": rope.get("rope_theta", text_config.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope.get("partial_rotary_factor", 1.0),
    }, text_config["layer_types"]


def _compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(bf).clamp_min(1e-12)
        ),
        "cosine": float(cosine),
        "sign_match": float((torch.signbit(af) == torch.signbit(bf)).float().mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="BF16 merged checkpoint dir")
    ap.add_argument("--v8", required=True, help="NVFP4 v8-RTN checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--cosine-threshold", type=float, default=0.97)
    args = ap.parse_args()

    bf16_dir = Path(args.bf16)
    v8_dir = Path(args.v8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg, layer_types = _cfg_from(bf16_dir)
    end_layer = args.start_layer + args.num_layers
    if args.start_layer < 0 or end_layer > len(layer_types):
        raise ValueError(
            f"Requested layers [{args.start_layer}, {end_layer}) outside "
            f"model layer count {len(layer_types)}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    h_bf16 = torch.randn(
        1,
        args.tokens,
        cfg["hidden_size"],
        generator=generator,
        dtype=torch.float32,
    )
    h_nvfp4 = h_bf16.clone()
    position_ids = torch.arange(args.tokens, dtype=torch.long).unsqueeze(0)

    layer_results: list[dict[str, Any]] = []
    for layer_idx in range(args.start_layer, end_layer):
        layer_type = layer_types[layer_idx]
        bf16_weights, _ = load_qwen36_layer(
            str(bf16_dir), layer_idx, device="cpu", dequant_dtype=torch.float32
        )
        nvfp4_weights, _ = load_qwen36_layer(
            str(v8_dir), layer_idx, device="cpu", dequant_dtype=torch.float32
        )
        with torch.no_grad():
            h_bf16 = _layer_forward(h_bf16, position_ids, layer_type, bf16_weights, cfg)
            h_nvfp4 = _layer_forward(h_nvfp4, position_ids, layer_type, nvfp4_weights, cfg)
        comparison = _compare(h_nvfp4, h_bf16)
        layer_results.append(
            {
                "layer": layer_idx,
                "layer_type": layer_type,
                "output_shape": list(h_bf16.shape),
                "comparison": comparison,
                "verdict": (
                    "PASS"
                    if comparison["cosine"] >= args.cosine_threshold
                    else "FAIL"
                ),
            }
        )

    min_cosine = min(item["comparison"]["cosine"] for item in layer_results)
    final_comparison = _compare(h_nvfp4, h_bf16)
    result = {
        "schema_version": "lynn-engine-p2-multilayer-forward-v1",
        "bf16_model": str(bf16_dir),
        "v8_model": str(v8_dir),
        "start_layer": args.start_layer,
        "num_layers": args.num_layers,
        "tokens": args.tokens,
        "seed": args.seed,
        "layer_results": layer_results,
        "summary": {
            "min_layer_cosine": min_cosine,
            "final_comparison": final_comparison,
        },
        "thresholds": {
            "cosine": args.cosine_threshold,
        },
        "verdict": (
            "PASS"
            if all(item["verdict"] == "PASS" for item in layer_results)
            and final_comparison["cosine"] >= args.cosine_threshold
            else "FAIL"
        ),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
