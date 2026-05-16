#!/usr/bin/env python3
"""A100 preflight for the W4A8 + MTP campaign.

This script is intentionally lightweight: it does not instantiate the full
27B model. It checks the BF16 variable-expert artifact directly from its
safetensors index, then loads one MoE layer at a time to measure how much
E4M3-per16 activation rounding perturbs active expert math.

It also records the MTP/NEXTN implementation state so the A100 training run
does not silently assume a draft head that is not present in the artifact.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


MTP_PATTERNS = (
    "mtp",
    "nextn",
    "next_n",
    "draft",
    "spec",
    "medusa",
    "eagle",
    "multi_token",
    "lookahead",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index = _read_json(model_dir / "model.safetensors.index.json")
    return index.get("weight_map", {})


def _tensor_from_index(model_dir: Path, weight_map: dict[str, str], key: str, device: str) -> torch.Tensor:
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing tensor key in index: {key}")
    path = model_dir / rel
    tensors = load_file(path, device=device)
    if key not in tensors:
        raise KeyError(f"tensor key {key!r} not found inside {path}")
    return tensors[key]


def _fp8_dtype(name: str) -> torch.dtype:
    if name == "e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required")
        return torch.float8_e4m3fn
    if name == "e5m2":
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("torch.float8_e5m2 is required")
        return torch.float8_e5m2
    raise ValueError(name)


def _fake_quant_fp8(x: torch.Tensor, *, fmt: str = "e4m3", group_size: int = 16) -> torch.Tensor:
    """Dynamic per-16 FP8 round-trip, returning the original dtype."""
    dtype = _fp8_dtype(fmt)
    max_fp8 = float(torch.finfo(dtype).max)
    x32 = x.float()
    if x32.shape[-1] % group_size != 0:
        raise ValueError(f"last dim must be divisible by {group_size}, got {tuple(x.shape)}")
    shape = x32.shape
    grouped = x32.reshape(-1, shape[-1] // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / max_fp8).clamp_min(1e-8)
    return ((grouped / scale).to(dtype).float() * scale).reshape(shape).to(x.dtype)


def _diff(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float]:
    rf = ref.float().reshape(-1)
    gf = got.float().reshape(-1)
    delta = gf - rf
    denom = torch.linalg.vector_norm(rf).clamp_min(1e-20)
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(delta) / denom).item()),
        "cosine": float(F.cosine_similarity(rf, gf, dim=0).item()),
    }


def _make_hidden(kind: str, *, seed: int, hidden_size: int, device: str) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if kind == "gaussian":
        x = torch.randn(hidden_size, generator=gen)
    elif kind == "wide":
        x = torch.randn(hidden_size, generator=gen) * 2.5
    elif kind == "outlier":
        x = torch.randn(hidden_size, generator=gen)
        mask = torch.rand(hidden_size, generator=gen) < 0.05
        x[mask] *= 16.0
    else:
        raise ValueError(kind)
    return x.to(device=device, dtype=torch.bfloat16).contiguous()


def _active_moe(
    hidden: torch.Tensor,
    *,
    router: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    top_k: int,
    mode: str,
    fmt: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one token through active routed experts with optional W4A8 rounding."""
    h_router = hidden.view(1, -1)
    logits = F.linear(h_router, router)
    routing_weights, expert_ids = torch.topk(logits, top_k, dim=-1, sorted=False)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids_1d = expert_ids[0].to(torch.long)

    h_gate = _fake_quant_fp8(hidden, fmt=fmt) if mode in {"gateup", "full"} else hidden
    outs: list[torch.Tensor] = []
    inter_diffs: list[dict[str, float]] = []
    for slot, expert_id_t in enumerate(expert_ids_1d):
        expert_id = int(expert_id_t.item())
        gu = F.linear(h_gate, gate_up[expert_id])
        gate, up = gu.chunk(2, dim=-1)
        inter = F.silu(gate.float()) * up.float()
        if mode == "full":
            inter_q = _fake_quant_fp8(inter.to(torch.bfloat16), fmt=fmt).float()
            inter_diffs.append(_diff(inter, inter_q))
            inter = inter_q
        out = F.linear(inter.to(torch.bfloat16), down[expert_id]).float()
        outs.append(out * routing_weights[slot])
    return torch.stack(outs, dim=0).sum(dim=0), {
        "expert_ids": [int(x) for x in expert_ids_1d.tolist()],
        "routing_weights": [float(x) for x in routing_weights.detach().cpu().tolist()],
        "inter_quant_diffs": inter_diffs,
    }


def _layer_probe(
    model_dir: Path,
    weight_map: dict[str, str],
    *,
    layer: int,
    seeds: list[int],
    distributions: list[str],
    top_k: int,
    device: str,
    fmt: str,
) -> dict[str, Any]:
    prefix = f"model.language_model.layers.{layer}"
    t0 = time.time()
    router = _tensor_from_index(model_dir, weight_map, f"{prefix}.mlp.gate.weight", device).to(torch.bfloat16)
    gate_up = _tensor_from_index(model_dir, weight_map, f"{prefix}.mlp.experts.gate_up_proj", device).to(torch.bfloat16)
    down = _tensor_from_index(model_dir, weight_map, f"{prefix}.mlp.experts.down_proj", device).to(torch.bfloat16)
    torch.cuda.synchronize()
    load_s = time.time() - t0
    cases = []
    hidden_size = int(router.shape[1])
    for dist in distributions:
        for seed in seeds:
            hidden = _make_hidden(dist, seed=seed, hidden_size=hidden_size, device=device)
            ref, meta = _active_moe(
                hidden,
                router=router,
                gate_up=gate_up,
                down=down,
                top_k=top_k,
                mode="off",
                fmt=fmt,
            )
            gateup, meta_gate = _active_moe(
                hidden,
                router=router,
                gate_up=gate_up,
                down=down,
                top_k=top_k,
                mode="gateup",
                fmt=fmt,
            )
            full, meta_full = _active_moe(
                hidden,
                router=router,
                gate_up=gate_up,
                down=down,
                top_k=top_k,
                mode="full",
                fmt=fmt,
            )
            hidden_q = _fake_quant_fp8(hidden, fmt=fmt)
            cases.append(
                {
                    "distribution": dist,
                    "seed": seed,
                    "expert_ids": meta["expert_ids"],
                    "routing_weights": meta["routing_weights"],
                    "gateup_same_experts": meta["expert_ids"] == meta_gate["expert_ids"],
                    "full_same_experts": meta["expert_ids"] == meta_full["expert_ids"],
                    "hidden_quant_diff": _diff(hidden, hidden_q),
                    "gateup_diff": _diff(ref, gateup),
                    "full_diff": _diff(ref, full),
                    "max_inter_rel_l2": max(
                        [d["rel_l2"] for d in meta_full["inter_quant_diffs"]],
                        default=0.0,
                    ),
                }
            )
            del hidden, ref, gateup, full
    summary = {
        "case_count": len(cases),
        "all_gateup_relaxed": all(c["gateup_diff"]["cosine"] >= 0.999 and c["gateup_diff"]["rel_l2"] <= 0.03 for c in cases),
        "all_full_relaxed": all(c["full_diff"]["cosine"] >= 0.999 and c["full_diff"]["rel_l2"] <= 0.03 for c in cases),
        "min_gateup_cosine": min(c["gateup_diff"]["cosine"] for c in cases),
        "max_gateup_rel_l2": max(c["gateup_diff"]["rel_l2"] for c in cases),
        "min_full_cosine": min(c["full_diff"]["cosine"] for c in cases),
        "max_full_rel_l2": max(c["full_diff"]["rel_l2"] for c in cases),
        "max_hidden_rel_l2": max(c["hidden_quant_diff"]["rel_l2"] for c in cases),
        "max_inter_rel_l2": max(c["max_inter_rel_l2"] for c in cases),
    }
    shapes = {
        "router": list(router.shape),
        "gate_up": list(gate_up.shape),
        "down": list(down.shape),
    }
    del router, gate_up, down
    torch.cuda.empty_cache()
    return {
        "layer": layer,
        "load_seconds": load_s,
        "shapes": shapes,
        "cases": cases,
        "summary": summary,
    }


def _mtp_preflight(model_dir: Path, config: dict[str, Any], weight_map: dict[str, str]) -> dict[str, Any]:
    keys = sorted(weight_map)
    mtp_keys = [k for k in keys if any(pattern in k.lower() for pattern in MTP_PATTERNS)]
    text_cfg = config.get("text_config", {})
    result: dict[str, Any] = {
        "config_mtp_num_hidden_layers": text_cfg.get("mtp_num_hidden_layers", config.get("mtp_num_hidden_layers")),
        "config_mtp_use_dedicated_embeddings": text_cfg.get("mtp_use_dedicated_embeddings", config.get("mtp_use_dedicated_embeddings")),
        "artifact_mtp_like_key_count": len(mtp_keys),
        "artifact_mtp_like_key_sample": mtp_keys[:50],
        "needs_lynn_owned_mtp_head": True,
    }
    try:
        mod = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
        cfg_mod = importlib.import_module("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")
        source = inspect.getsource(mod)
        result.update(
            {
                "transformers_modeling_file": getattr(mod, "__file__", None),
                "transformers_config_file": getattr(cfg_mod, "__file__", None),
                "transformers_modeling_mtp_token_count": sum(source.lower().count(p) for p in MTP_PATTERNS),
                "has_named_mtp_class": any("MTP" in name or "Next" in name or "Draft" in name for name in dir(mod)),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        result["transformers_probe_error"] = repr(exc)

    hidden_size = int(text_cfg.get("hidden_size", 2048))
    vocab_size = int(text_cfg.get("vocab_size", config.get("vocab_size", 0)) or 0)
    lm_head_bytes_bf16 = hidden_size * vocab_size * 2 if vocab_size else None
    result["rough_mtp_head_param_budget"] = {
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "lm_head_like_bf16_gib": None if lm_head_bytes_bf16 is None else lm_head_bytes_bf16 / (1024**3),
        "recommendation": (
            "Start with tied lm_head / shallow NEXTN block. Do not duplicate a full lm_head "
            "unless accept-rate evidence justifies the extra ~1 GiB BF16 head."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28, 36])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--distributions", nargs="+", default=["gaussian", "wide", "outlier"])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fmt", default="e4m3", choices=["e4m3", "e5m2"])
    args = parser.parse_args()

    model_dir = Path(args.model)
    config = _read_json(model_dir / "config.json")
    weight_map = _load_weight_map(model_dir)
    if not config or not weight_map:
        raise SystemExit(f"model config/index missing under {model_dir}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this preflight")

    torch.manual_seed(0)
    random.seed(0)
    start = time.time()
    layer_results = [
        _layer_probe(
            model_dir,
            weight_map,
            layer=layer,
            seeds=args.seeds,
            distributions=args.distributions,
            top_k=args.top_k,
            device=args.device,
            fmt=args.fmt,
        )
        for layer in args.layers
    ]
    summaries = [x["summary"] for x in layer_results]
    all_gateup = all(x["all_gateup_relaxed"] for x in summaries)
    all_full = all(x["all_full_relaxed"] for x in summaries)
    max_full = max(x["max_full_rel_l2"] for x in summaries)
    max_gateup = max(x["max_gateup_rel_l2"] for x in summaries)
    if all_gateup and all_full:
        decision = "GREEN: BF16 artifact is locally W4A8-friendly on sampled active-MoE layers; start short Recovery and generation gates."
        code = 0
    elif all_gateup and max_full <= 0.05:
        decision = "AMBER: gate/up W4A8 is stable and full-active is near gate; start W4A8 Recovery with down/intermediate activation as the primary repair target."
        code = 1
    else:
        decision = "RED: sampled W4A8 active-MoE drift is large; run Recovery only as a diagnostic and keep W4A8 runtime disabled."
        code = 2

    result = {
        "schema_version": "lynn-a100-w4a8-mtp-preflight-v1",
        "model": str(model_dir),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "device_name": torch.cuda.get_device_name(torch.device(args.device)),
        },
        "probe": {
            "format": args.fmt,
            "layers": args.layers,
            "seeds": args.seeds,
            "distributions": args.distributions,
            "top_k": args.top_k,
        },
        "layer_results": layer_results,
        "aggregate": {
            "all_gateup_relaxed": all_gateup,
            "all_full_relaxed": all_full,
            "max_gateup_rel_l2": max_gateup,
            "max_full_rel_l2": max_full,
            "min_gateup_cosine": min(x["min_gateup_cosine"] for x in summaries),
            "min_full_cosine": min(x["min_full_cosine"] for x in summaries),
        },
        "mtp_preflight": _mtp_preflight(model_dir, config, weight_map),
        "decision": decision,
        "elapsed_seconds": time.time() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
