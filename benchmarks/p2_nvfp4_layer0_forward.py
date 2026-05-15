#!/usr/bin/env python3
"""P2-A5 gate: run one real layer with BF16 and NVFP4 slow-dequant weights.

This is the first layer-level forward gate for Lynn engine's NVFP4 path. It
loads layer 0 through the public loader for both checkpoints, runs the existing
Lynn engine layer forward on a deterministic hidden state, and compares the
NVFP4 slow-dequant output against the BF16 oracle.

It is a CPU correctness gate, not a performance benchmark.
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
    layer_types = text_config["layer_types"]
    return {
        "hidden_size": text_config["hidden_size"],
        "num_attention_heads": text_config["num_attention_heads"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "head_dim": text_config["head_dim"],
        "num_experts": text_config["num_experts"],
        "num_experts_per_tok": text_config["num_experts_per_tok"],
        "rope_theta": rope.get("rope_theta", text_config.get("rope_theta", 1e6)),
        "partial_rotary_factor": rope.get("partial_rotary_factor", 1.0),
    }, layer_types


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
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--cosine-threshold", type=float, default=0.97)
    args = ap.parse_args()

    bf16_dir = Path(args.bf16)
    v8_dir = Path(args.v8)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg, layer_types = _cfg_from(bf16_dir)
    layer_type = layer_types[args.layer]
    bf16_weights, _ = load_qwen36_layer(
        str(bf16_dir), args.layer, device="cpu", dequant_dtype=torch.float32
    )
    nvfp4_weights, _ = load_qwen36_layer(
        str(v8_dir), args.layer, device="cpu", dequant_dtype=torch.float32
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    hidden = torch.randn(
        1,
        args.tokens,
        cfg["hidden_size"],
        generator=generator,
        dtype=torch.float32,
    )
    position_ids = torch.arange(args.tokens, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        bf16_output = _layer_forward(hidden, position_ids, layer_type, bf16_weights, cfg)
        nvfp4_output = _layer_forward(hidden, position_ids, layer_type, nvfp4_weights, cfg)

    comparison = _compare(nvfp4_output, bf16_output)
    result = {
        "schema_version": "lynn-engine-p2-layer-forward-v1",
        "bf16_model": str(bf16_dir),
        "v8_model": str(v8_dir),
        "layer": args.layer,
        "layer_type": layer_type,
        "input_shape": list(hidden.shape),
        "output_shape": list(bf16_output.shape),
        "seed": args.seed,
        "tokens": args.tokens,
        "comparison": comparison,
        "thresholds": {
            "cosine": args.cosine_threshold,
        },
        "verdict": (
            "PASS"
            if comparison["cosine"] >= args.cosine_threshold
            else "FAIL"
        ),
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
