#!/usr/bin/env python3
"""A100 real-prompt W4A8 active-MoE gate for BF16 artifacts.

`a100_w4a8_mtp_preflight.py` uses synthetic hidden vectors so it can run without
loading the full model. This companion gate loads the resident BF16 model and
uses real prompt hidden states at selected layers. It is slower, but it answers
the decision question that matters before starting Recovery: does W4A8
activation rounding drift on the actual Lynn prompt distribution?
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
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


PROMPTS = [
    "用一句话解释 W4A8 为什么可能比 W4A4 更稳。",
    "请写一个 Python 函数,判断字符串是否是回文。",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请给出一个 JSON: {\"city\":\"Tokyo\",\"unit\":\"celsius\"}",
    "总结一下 MoE 模型里 router 和 expert 的分工。",
    "解释长上下文推理里 linear attention 的优势。",
]


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
    inter_diffs: list[dict[str, float]] = []
    for slot, expert_id_t in enumerate(expert_ids):
        expert_id = int(expert_id_t.item())
        gate_up = F.linear(h_gate, w["mlp.experts.gate_up_proj"][expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = F.silu(gate.float()) * up.float()
        if mode == "full":
            inter_q = _fake_quant_fp8(inter.to(torch.bfloat16), fmt=fmt).float()
            inter_diffs.append(_diff(inter, inter_q))
            inter = inter_q
        out = F.linear(inter.to(torch.bfloat16), w["mlp.experts.down_proj"][expert_id]).float()
        outs.append(out * routing_weights[slot])
    return torch.stack(outs, dim=0).sum(dim=0), {
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.detach().cpu().tolist()],
        "inter_quant_diffs": inter_diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 16, 28, 36])
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--fmt", default="e4m3", choices=["e4m3", "e5m2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    start = time.time()
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )
    load_elapsed = time.time() - start
    cases: list[dict[str, Any]] = []
    for layer in args.layers:
        w = runner.layer_weights[layer]
        cfg = runner.layer_cfgs[layer]
        top_k = int(cfg["num_experts_per_tok"])
        for idx, prompt in enumerate(args.prompts):
            h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
            h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
            hidden = h_moe.reshape(-1, h_moe.shape[-1])[0].contiguous()
            ref, ref_meta = _active_moe(hidden, w, top_k=top_k, mode="off", fmt=args.fmt)
            gateup, gate_meta = _active_moe(hidden, w, top_k=top_k, mode="gateup", fmt=args.fmt)
            full, full_meta = _active_moe(hidden, w, top_k=top_k, mode="full", fmt=args.fmt)
            cases.append(
                {
                    "layer": layer,
                    "prompt_id": idx,
                    "prompt": prompt,
                    "top_k": top_k,
                    "expert_ids": ref_meta["expert_ids"],
                    "gateup_same_experts": ref_meta["expert_ids"] == gate_meta["expert_ids"],
                    "full_same_experts": ref_meta["expert_ids"] == full_meta["expert_ids"],
                    "hidden_quant_diff": _diff(hidden, _fake_quant_fp8(hidden, fmt=args.fmt)),
                    "gateup_diff": _diff(ref, gateup),
                    "full_diff": _diff(ref, full),
                    "max_inter_rel_l2": max(
                        [d["rel_l2"] for d in full_meta["inter_quant_diffs"]],
                        default=0.0,
                    ),
                }
            )
            del h_layer, h_moe, hidden, ref, gateup, full
            torch.cuda.empty_cache()

    by_layer: dict[str, dict[str, Any]] = {}
    for layer in args.layers:
        rows = [c for c in cases if c["layer"] == layer]
        by_layer[str(layer)] = {
            "case_count": len(rows),
            "all_gateup_relaxed": all(c["gateup_diff"]["cosine"] >= 0.999 and c["gateup_diff"]["rel_l2"] <= 0.03 for c in rows),
            "all_full_relaxed": all(c["full_diff"]["cosine"] >= 0.999 and c["full_diff"]["rel_l2"] <= 0.03 for c in rows),
            "min_gateup_cosine": min(c["gateup_diff"]["cosine"] for c in rows),
            "max_gateup_rel_l2": max(c["gateup_diff"]["rel_l2"] for c in rows),
            "min_full_cosine": min(c["full_diff"]["cosine"] for c in rows),
            "max_full_rel_l2": max(c["full_diff"]["rel_l2"] for c in rows),
            "max_hidden_rel_l2": max(c["hidden_quant_diff"]["rel_l2"] for c in rows),
            "max_inter_rel_l2": max(c["max_inter_rel_l2"] for c in rows),
        }

    all_gateup = all(v["all_gateup_relaxed"] for v in by_layer.values())
    all_full = all(v["all_full_relaxed"] for v in by_layer.values())
    max_full = max(v["max_full_rel_l2"] for v in by_layer.values())
    if all_gateup and all_full:
        decision = "GREEN: real-prompt W4A8 active-MoE drift is within relaxed gate; start Recovery and generation gates."
        code = 0
    elif all_gateup and max_full <= 0.04:
        decision = "AMBER: gate/up is clean on real prompts, full-active is near gate; train Recovery on down/intermediate activation."
        code = 1
    elif max_full <= 0.05:
        decision = "AMBER: real-prompt W4A8 drift is repairable but gate/up also needs Recovery margin."
        code = 1
    else:
        decision = "RED: real-prompt W4A8 drift is too large for short Recovery assumptions."
        code = 2

    result = {
        "schema_version": "lynn-a100-w4a8-real-prompt-gate-v1",
        "model": args.model,
        "torch": {
            "version": torch.__version__,
            "device": args.device,
            "device_name": torch.cuda.get_device_name(torch.device(args.device)),
            "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "load_elapsed_seconds": load_elapsed,
        "format": args.fmt,
        "layers": args.layers,
        "prompt_count": len(args.prompts),
        "cases": cases,
        "summary_by_layer": by_layer,
        "aggregate": {
            "all_gateup_relaxed": all_gateup,
            "all_full_relaxed": all_full,
            "max_gateup_rel_l2": max(v["max_gateup_rel_l2"] for v in by_layer.values()),
            "max_full_rel_l2": max_full,
            "min_gateup_cosine": min(v["min_gateup_cosine"] for v in by_layer.values()),
            "min_full_cosine": min(v["min_full_cosine"] for v in by_layer.values()),
        },
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
