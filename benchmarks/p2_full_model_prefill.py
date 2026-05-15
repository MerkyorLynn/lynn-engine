#!/usr/bin/env python3
"""P2-C gate: full-model prefill/logits for BF16 and NVFP4 checkpoints.

This is the smallest end-to-end Lynn engine startup proof for public HF-style
checkpoints. It runs the complete 40-layer prefill path through Lynn engine and
compares the final logits from an NVFP4 v8-RTN checkpoint against the BF16
oracle for the same prompt.

The path is intentionally layer-streamed and slow; it is a correctness gate, not
a throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _rms_norm, load_outside_weights
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


def _logit_compare(a: torch.Tensor, b: torch.Tensor, top_k: int) -> dict[str, Any]:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    cosine = torch.dot(af, bf) / (
        torch.linalg.vector_norm(af).clamp_min(1e-12)
        * torch.linalg.vector_norm(bf).clamp_min(1e-12)
    )
    a_top = torch.topk(af, top_k)
    b_top = torch.topk(bf, top_k)
    a_set = set(a_top.indices.tolist())
    b_set = set(b_top.indices.tolist())
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(bf).clamp_min(1e-12)
        ),
        "cosine": float(cosine),
        "top1_match": int(a_top.indices[0]) == int(b_top.indices[0]),
        "topk_overlap": len(a_set & b_set) / top_k,
        "nvfp4_top_ids": [int(x) for x in a_top.indices.tolist()],
        "bf16_top_ids": [int(x) for x in b_top.indices.tolist()],
        "nvfp4_top_values": [float(x) for x in a_top.values.tolist()],
        "bf16_top_values": [float(x) for x in b_top.values.tolist()],
    }


def _run_prefill(
    model_dir: Path,
    prompt: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    cfg, layer_types = _cfg_from(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device, dtype=torch.long).unsqueeze(0)

    started = time.time()
    outside = load_outside_weights(str(model_dir), device, dtype)
    h = F.embedding(input_ids, outside["model.language_model.embed_tokens.weight"])
    layer_timings = []
    for layer_idx, layer_type in enumerate(layer_types):
        load_start = time.time()
        weights, _ = load_qwen36_layer(
            str(model_dir),
            layer_idx,
            num_experts=cfg["num_experts"],
            device=device,
            dequant_dtype=dtype,
        )
        load_seconds = time.time() - load_start
        forward_start = time.time()
        h = _layer_forward(h, position_ids, layer_type, weights, cfg)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_seconds = time.time() - forward_start
        layer_timings.append(
            {
                "layer": layer_idx,
                "layer_type": layer_type,
                "load_seconds": load_seconds,
                "forward_seconds": forward_seconds,
                "hidden_abs_mean": float(h.float().abs().mean()),
            }
        )
        del weights
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    h = _rms_norm(h, outside["model.language_model.norm.weight"])
    logits = F.linear(h[:, -1, :], outside["lm_head.weight"]).detach().cpu()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    top = torch.topk(logits[0].float(), 10)
    return {
        "prompt": prompt,
        "input_ids": input_ids.detach().cpu().tolist()[0],
        "logits": logits,
        "top_ids": [int(x) for x in top.indices.tolist()],
        "top_values": [float(x) for x in top.values.tolist()],
        "top_text": [tokenizer.decode([int(x)]) for x in top.indices.tolist()],
        "layer_timings": layer_timings,
        "total_seconds": time.time() - started,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--v8", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="你好")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cosine-threshold", type=float, default=0.95)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    bf16 = _run_prefill(Path(args.bf16), args.prompt, device=args.device, dtype=dtype)
    nvfp4 = _run_prefill(Path(args.v8), args.prompt, device=args.device, dtype=dtype)
    comparison = _logit_compare(nvfp4["logits"], bf16["logits"], args.top_k)

    # Logits are large; keep metrics and top-k only in JSON.
    del bf16["logits"]
    del nvfp4["logits"]

    result = {
        "schema_version": "lynn-engine-p2-full-model-prefill-v1",
        "bf16_model": args.bf16,
        "v8_model": args.v8,
        "prompt": args.prompt,
        "device": args.device,
        "dtype": args.dtype,
        "bf16": bf16,
        "nvfp4": nvfp4,
        "comparison": comparison,
        "thresholds": {
            "logit_cosine": args.cosine_threshold,
        },
        "verdict": (
            "PASS"
            if comparison["cosine"] >= args.cosine_threshold
            else "FAIL"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
