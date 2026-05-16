#!/usr/bin/env python3
"""Validate a folded W4A8 alpha overlay against the original BF16 artifact.

This is the correct validation口径 for a folded artifact:

* reference: original BF16 model active-MoE output;
* candidate: folded artifact active-MoE output with full W4A8 fake quant.

It intentionally does not compare the folded artifact to itself.
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

from benchmarks.p10e_packed_active_expert_probe import _prefill_to_layer_input  # noqa: E402
from engine.full_forward import _rms_norm  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_w4a8_real_prompt_gate import PROMPTS, _diff, _fake_quant_fp8  # noqa: E402


def _active_moe(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    *,
    top_k: int,
    mode: str,
    fmt: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    h_flat = hidden.view(1, -1)
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1, sorted=False)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)[0]
    expert_ids = expert_indices[0].to(torch.long)
    h_gate = _fake_quant_fp8(hidden, fmt=fmt) if mode in {"gateup", "full"} else hidden
    outs: list[torch.Tensor] = []
    for slot, expert_id_t in enumerate(expert_ids):
        expert_id = int(expert_id_t.item())
        gate_up = F.linear(h_gate, w["mlp.experts.gate_up_proj"][expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate.float()) * up.float()
        if mode == "full":
            inter = _fake_quant_fp8(inter.to(torch.bfloat16), fmt=fmt).float()
        out = F.linear(inter.to(torch.bfloat16), w["mlp.experts.down_proj"][expert_id]).float()
        outs.append(out * routing_weights[slot])
    return torch.stack(outs, dim=0).sum(dim=0), {
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.detach().cpu().tolist()],
    }


def _collect_refs(
    original_model: str,
    *,
    layers: list[int],
    prompts: list[str],
    fmt: str,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner = LynnIncrementalRunner(
        original_model,
        device=device,
        dtype=torch.bfloat16,
        max_seq_len=4096,
        verbose=True,
    )
    refs: list[dict[str, Any]] = []
    for layer in layers:
        w = runner.layer_weights[layer]
        cfg = runner.layer_cfgs[layer]
        top_k = int(cfg["num_experts_per_tok"])
        for prompt_id, prompt in enumerate(prompts):
            h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
            h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
            hidden = h_moe.reshape(-1, h_moe.shape[-1])[0].contiguous()
            ref, meta = _active_moe(hidden, w, top_k=top_k, mode="off", fmt=fmt)
            refs.append(
                {
                    "layer": layer,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "top_k": top_k,
                    "hidden": hidden.detach().cpu(),
                    "ref": ref.detach().cpu(),
                    "ref_expert_ids": meta["expert_ids"],
                }
            )
            del h_layer, h_moe, hidden, ref
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    del runner
    torch.cuda.empty_cache()
    return refs, {"original_peak_memory_gib": peak}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-model", required=True)
    parser.add_argument("--folded-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--fmt", default="e4m3", choices=["e4m3", "e5m2"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    start = time.time()
    refs, load_info = _collect_refs(
        args.original_model,
        layers=args.layers,
        prompts=args.prompts,
        fmt=args.fmt,
        device=args.device,
    )
    cases = []
    loaded_layers: dict[int, tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}
    for rec in refs:
        layer = int(rec["layer"])
        if layer not in loaded_layers:
            loaded_layers[layer] = load_qwen36_layer(
                args.folded_model,
                layer,
                num_experts=256,
                device=args.device,
                dequant_dtype=torch.bfloat16,
            )
        w, cfg = loaded_layers[layer]
        top_k = int(cfg.get("num_experts_per_tok", rec["top_k"]))
        hidden = rec["hidden"].to(args.device, dtype=torch.bfloat16)
        ref = rec["ref"].to(args.device)
        folded_bf16, meta_bf16 = _active_moe(hidden, w, top_k=top_k, mode="off", fmt=args.fmt)
        folded_w4a8, meta_w4a8 = _active_moe(hidden, w, top_k=top_k, mode="full", fmt=args.fmt)
        cases.append(
            {
                "layer": layer,
                "prompt_id": rec["prompt_id"],
                "prompt": rec["prompt"],
                "ref_expert_ids": rec["ref_expert_ids"],
                "folded_bf16_expert_ids": meta_bf16["expert_ids"],
                "folded_w4a8_expert_ids": meta_w4a8["expert_ids"],
                "folded_bf16_vs_original": _diff(ref, folded_bf16),
                "folded_w4a8_vs_original": _diff(ref, folded_w4a8),
            }
        )
    by_layer: dict[str, dict[str, Any]] = {}
    for layer in args.layers:
        rows = [c for c in cases if c["layer"] == layer]
        by_layer[str(layer)] = {
            "case_count": len(rows),
            "max_folded_bf16_rel_l2": max(c["folded_bf16_vs_original"]["rel_l2"] for c in rows),
            "max_folded_w4a8_rel_l2": max(c["folded_w4a8_vs_original"]["rel_l2"] for c in rows),
            "min_folded_bf16_cosine": min(c["folded_bf16_vs_original"]["cosine"] for c in rows),
            "min_folded_w4a8_cosine": min(c["folded_w4a8_vs_original"]["cosine"] for c in rows),
        }
    max_w4a8 = max(v["max_folded_w4a8_rel_l2"] for v in by_layer.values())
    max_bf16 = max(v["max_folded_bf16_rel_l2"] for v in by_layer.values())
    decision = (
        "GREEN: folded artifact W4A8 active-MoE stays under 3% local drift vs original BF16."
        if max_w4a8 <= 0.03
        else "AMBER: folded artifact improves local drift but still needs broader Recovery/generation validation."
    )
    result = {
        "schema_version": "lynn-a100-w4a8-folded-vs-original-gate-v1",
        "original_model": args.original_model,
        "folded_model": args.folded_model,
        "format": args.fmt,
        "layers": args.layers,
        "prompt_count": len(args.prompts),
        "cases": cases,
        "summary_by_layer": by_layer,
        "aggregate": {
            "max_folded_bf16_rel_l2": max_bf16,
            "max_folded_w4a8_rel_l2": max_w4a8,
            "min_folded_bf16_cosine": min(v["min_folded_bf16_cosine"] for v in by_layer.values()),
            "min_folded_w4a8_cosine": min(v["min_folded_w4a8_cosine"] for v in by_layer.values()),
        },
        "load_info": load_info,
        "decision": decision,
        "elapsed_seconds": time.time() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if decision.startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
