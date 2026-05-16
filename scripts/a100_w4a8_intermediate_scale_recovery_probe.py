#!/usr/bin/env python3
"""Probe a foldable W4A8 intermediate-scale Recovery adapter.

The A100 real-prompt gate showed:

* gate/up W4A8 is clean across sampled real prompts;
* full-active drift is near the gate and mostly comes from the intermediate
  activation before `down_proj`.

This script tests a cheap Recovery candidate before any expensive fine-tune:
learn one per-layer intermediate-channel scale vector `alpha[512]` applied
after FP8-rounding the expert intermediate and before `down_proj`.

If it helps, the scale can be folded into down weights:

    down(inter_q * alpha) == linear(inter_q, down_weight * alpha)

So the runtime does not need an extra multiply in the final artifact.
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
from scripts.a100_w4a8_real_prompt_gate import PROMPTS, _diff, _fake_quant_fp8  # noqa: E402


def _active_moe_with_alpha(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    *,
    top_k: int,
    alpha: torch.Tensor | None,
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
        if alpha is not None:
            if alpha.ndim == 1:
                inter = inter * alpha.float()
            elif alpha.ndim == 2:
                inter = inter * alpha[expert_id].float()
            else:
                raise ValueError(f"alpha must be [I] or [E, I], got {tuple(alpha.shape)}")
        out = F.linear(inter.to(torch.bfloat16), w["mlp.experts.down_proj"][expert_id]).float()
        outs.append(out * routing_weights[slot])
    return torch.stack(outs, dim=0).sum(dim=0), {
        "expert_ids": [int(x) for x in expert_ids.tolist()],
        "routing_weights": [float(x) for x in routing_weights.detach().cpu().tolist()],
        "inter_quant_diffs": inter_diffs,
    }


def _collect_layer_cases(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompts: list[str],
    fmt: str,
) -> list[dict[str, Any]]:
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    top_k = int(cfg["num_experts_per_tok"])
    cases = []
    for prompt_id, prompt in enumerate(prompts):
        h_layer, _ = _prefill_to_layer_input(runner, layer, prompt)
        h_moe = _rms_norm(h_layer, w["post_attention_layernorm.weight"])
        hidden = h_moe.reshape(-1, h_moe.shape[-1])[0].contiguous()
        ref, _ = _active_moe_with_alpha(hidden, w, top_k=top_k, alpha=None, mode="off", fmt=fmt)
        base, _ = _active_moe_with_alpha(hidden, w, top_k=top_k, alpha=None, mode="full", fmt=fmt)
        cases.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "hidden": hidden.detach(),
                "ref": ref.detach(),
                "baseline": base.detach(),
                "baseline_diff": _diff(ref, base),
            }
        )
        del h_layer, h_moe, hidden, ref, base
    return cases


def _train_layer_alpha(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    prompts: list[str],
    fmt: str,
    steps: int,
    lr: float,
    reg: float,
    alpha_mode: str,
) -> dict[str, Any]:
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    top_k = int(cfg["num_experts_per_tok"])
    cases = _collect_layer_cases(runner, layer=layer, prompts=prompts, fmt=fmt)
    n_experts = int(w["mlp.experts.down_proj"].shape[0])
    inter_size = int(w["mlp.experts.down_proj"].shape[-1])
    if alpha_mode == "shared":
        alpha_shape = (inter_size,)
    elif alpha_mode == "expert":
        alpha_shape = (n_experts, inter_size)
    else:
        raise ValueError("alpha_mode must be shared or expert")
    log_alpha = torch.zeros(alpha_shape, device=runner.device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([log_alpha], lr=lr)

    history = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        alpha = torch.exp(log_alpha).clamp(0.75, 1.25)
        losses = []
        for c in cases:
            out, _ = _active_moe_with_alpha(
                c["hidden"],
                w,
                top_k=top_k,
                alpha=alpha,
                mode="full",
                fmt=fmt,
            )
            ref = c["ref"].float()
            denom = ref.pow(2).mean().clamp_min(1e-8)
            losses.append((out.float() - ref).pow(2).mean() / denom)
        loss = torch.stack(losses).mean() + reg * (log_alpha.pow(2).mean())
        loss.backward()
        opt.step()
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 5) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().item())})

    alpha = torch.exp(log_alpha.detach()).clamp(0.75, 1.25)
    eval_cases = []
    for c in cases:
        out, _ = _active_moe_with_alpha(c["hidden"], w, top_k=top_k, alpha=alpha, mode="full", fmt=fmt)
        eval_cases.append(
            {
                "prompt_id": c["prompt_id"],
                "prompt": c["prompt"],
                "before": c["baseline_diff"],
                "after": _diff(c["ref"], out),
            }
        )
    before_max = max(c["before"]["rel_l2"] for c in eval_cases)
    after_max = max(c["after"]["rel_l2"] for c in eval_cases)
    result = {
        "layer": layer,
        "alpha_mode": alpha_mode,
        "prompt_count": len(cases),
        "history": history,
        "before": {
            "max_rel_l2": before_max,
            "min_cosine": min(c["before"]["cosine"] for c in eval_cases),
        },
        "after": {
            "max_rel_l2": after_max,
            "min_cosine": min(c["after"]["cosine"] for c in eval_cases),
        },
        "improvement": {
            "max_rel_l2_delta": before_max - after_max,
            "max_rel_l2_ratio": after_max / max(before_max, 1e-12),
        },
        "alpha_stats": {
            "min": float(alpha.min().item()),
            "max": float(alpha.max().item()),
            "mean": float(alpha.mean().item()),
            "std": float(alpha.std().item()),
        },
        "cases": eval_cases,
    }
    del cases, log_alpha, alpha
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 26, 24, 12, 23, 32])
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--fmt", default="e4m3", choices=["e4m3", "e5m2"])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--alpha-mode", default="shared", choices=["shared", "expert"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    start = time.time()
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        max_seq_len=4096,
        verbose=True,
    )
    layer_results = [
        _train_layer_alpha(
            runner,
            layer=layer,
            prompts=args.prompts,
            fmt=args.fmt,
            steps=args.steps,
            lr=args.lr,
            reg=args.reg,
            alpha_mode=args.alpha_mode,
        )
        for layer in args.layers
    ]
    before_max = max(r["before"]["max_rel_l2"] for r in layer_results)
    after_max = max(r["after"]["max_rel_l2"] for r in layer_results)
    improved = after_max < before_max
    decision = (
        "GREEN: foldable intermediate alpha improves the worst W4A8 down/intermediate drift; test folding into down weights."
        if improved and after_max <= 0.03
        else (
            "AMBER: foldable alpha improves drift but does not fully clear the gate; use it as initialization or expand to expert-wise alpha."
            if improved
            else "RED: shared foldable alpha does not improve drift; move to LoRA/weight Recovery."
        )
    )
    result = {
        "schema_version": "lynn-a100-w4a8-intermediate-scale-recovery-probe-v1",
        "model": args.model,
        "format": args.fmt,
        "layers": args.layers,
        "prompt_count": len(args.prompts),
        "steps": args.steps,
        "lr": args.lr,
        "reg": args.reg,
        "alpha_mode": args.alpha_mode,
        "layer_results": layer_results,
        "aggregate": {
            "before_max_rel_l2": before_max,
            "after_max_rel_l2": after_max,
            "delta": before_max - after_max,
            "ratio": after_max / max(before_max, 1e-12),
        },
        "decision": decision,
        "elapsed_seconds": time.time() - start,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if decision.startswith("GREEN") else (1 if decision.startswith("AMBER") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
