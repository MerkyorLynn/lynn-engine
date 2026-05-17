#!/usr/bin/env python3
"""Prefix-margin W4A8 Recovery probe for structured first-token drift.

The regular intermediate-scale Recovery probe trains alpha on prompt-end local
MoE activations. The current generation failures are narrower: structured
prompts often enter the wrong output format in the first few decode tokens.

This probe collects active-MoE cases from the teacher path over the first
`prefix_steps` generated tokens, optionally feeding an explicit format prefix.
It then trains the same foldable alpha vectors:

    down(inter_q * alpha) == linear(inter_q, down_weight * alpha)

The output alpha directory can be folded with `a100_fold_w4a8_alpha_overlay.py`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from scripts.a100_w4a8_intermediate_scale_recovery_probe import _active_moe_with_alpha  # noqa: E402
from scripts.a100_w4a8_real_prompt_gate import _diff  # noqa: E402


def _load_specs(path: str | None, prompts: list[str]) -> list[dict[str, Any]]:
    if not path:
        return [{"id": str(i), "prompt": prompt} for i, prompt in enumerate(prompts)]
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            specs.append({"id": str(i), "prompt": item})
        elif isinstance(item, dict):
            specs.append({"id": str(item.get("id", i)), **item})
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return specs


def _alpha_stats(alpha: torch.Tensor) -> dict[str, float]:
    a = alpha.float()
    return {
        "min": float(a.min().item()),
        "max": float(a.max().item()),
        "mean": float(a.mean().item()),
        "std": float(a.std().item()),
    }


def _append_layer_case(
    cases_by_layer: dict[int, list[dict[str, Any]]],
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    hidden_in: torch.Tensor,
    prompt_id: str,
    prompt: str,
    decode_index: int,
    token_id: int | None,
    fmt: str,
) -> None:
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    top_k = int(cfg["num_experts_per_tok"])
    h_moe = _rms_norm(hidden_in.reshape(1, 1, -1), w["post_attention_layernorm.weight"])
    hidden = h_moe.reshape(-1, h_moe.shape[-1])[0].contiguous().detach()
    ref, route = _active_moe_with_alpha(hidden, w, top_k=top_k, alpha=None, mode="off", fmt=fmt)
    base, _ = _active_moe_with_alpha(hidden, w, top_k=top_k, alpha=None, mode="full", fmt=fmt)
    cases_by_layer[layer].append(
        {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "decode_index": decode_index,
            "input_token_id": token_id,
            "hidden": hidden.detach(),
            "ref": ref.detach(),
            "baseline": base.detach(),
            "baseline_diff": _diff(ref, base),
            "route": route,
        }
    )


def _collect_prefix_cases(
    runner: LynnIncrementalRunner,
    *,
    specs: list[dict[str, Any]],
    layers: list[int],
    prefix_steps: int,
    fmt: str,
    force_prefix: bool,
    use_chat_template: bool,
) -> dict[int, list[dict[str, Any]]]:
    tok = runner.tokenizer
    layer_set = set(layers)
    cases_by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    old_active = os.environ.get("LYNN_W4A8_FAKE_QUANT_ACTIVE")
    os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = "off"
    try:
        for spec in specs:
            prompt = str(spec["prompt"])
            prompt_id = str(spec.get("id", len(cases_by_layer)))
            forced_prefix = str(spec.get("forced_prefix", "")) if force_prefix and spec.get("forced_prefix") else ""
            forced_ids = tok(forced_prefix, add_special_tokens=False).input_ids if forced_prefix else []

            ids = _encode_prompt(tok, prompt, runner.device, use_chat_template=use_chat_template)
            state = LynnInferenceState(
                batch=1,
                max_seq_len=runner.max_seq_len,
                device=runner.device,
                dtype=runner.dtype,
            )
            h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
            pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
            for i in range(runner.n_layers):
                if i in layer_set:
                    _append_layer_case(
                        cases_by_layer,
                        runner,
                        layer=i,
                        hidden_in=h[:, -1, :],
                        prompt_id=prompt_id,
                        prompt=prompt,
                        decode_index=0,
                        token_id=None,
                        fmt=fmt,
                    )
                h = _prefill_layer(h, pos, LAYER_TYPES[i], runner.layer_weights[i], runner.layer_cfgs[i], state, i)
            state.seq_len = int(ids.shape[1])
            h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
            logits = runner._lm_head_logits(h_final)
            raw_next = int(logits[0].argmax().item())
            next_id = int(forced_ids[0]) if forced_ids else raw_next

            for step in range(1, prefix_steps):
                token = torch.tensor([[next_id]], device=runner.device, dtype=torch.long)
                pos_tensor = torch.tensor([[state.seq_len]], device=runner.device, dtype=torch.long)
                h = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
                for i in range(runner.n_layers):
                    if i in layer_set:
                        _append_layer_case(
                            cases_by_layer,
                            runner,
                            layer=i,
                            hidden_in=h[:, 0, :],
                            prompt_id=prompt_id,
                            prompt=prompt,
                            decode_index=step,
                            token_id=next_id,
                            fmt=fmt,
                        )
                    h = runner._decode_layer_fast(h, pos_tensor, state, i)
                state.seq_len += 1
                h_final = _rms_norm(h, runner.outside["model.language_model.norm.weight"])
                logits = runner._lm_head_logits(h_final)
                raw_next = int(logits[0].argmax().item())
                next_id = int(forced_ids[step]) if step < len(forced_ids) else raw_next
            del state, h, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if old_active is None:
            os.environ.pop("LYNN_W4A8_FAKE_QUANT_ACTIVE", None)
        else:
            os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = old_active
    return cases_by_layer


def _train_layer_alpha(
    runner: LynnIncrementalRunner,
    *,
    layer: int,
    cases: list[dict[str, Any]],
    fmt: str,
    steps: int,
    lr: float,
    reg: float,
    alpha_mode: str,
    alpha_min: float,
    alpha_max: float,
    save_alpha_dir: Path | None,
) -> dict[str, Any]:
    w = runner.layer_weights[layer]
    cfg = runner.layer_cfgs[layer]
    top_k = int(cfg["num_experts_per_tok"])
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
    history: list[dict[str, float]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        alpha = torch.exp(log_alpha).clamp(alpha_min, alpha_max)
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
        loss = torch.stack(losses).mean() + reg * log_alpha.pow(2).mean()
        loss.backward()
        opt.step()
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 5) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().item())})

    alpha = torch.exp(log_alpha.detach()).clamp(alpha_min, alpha_max)
    alpha_path = None
    if save_alpha_dir is not None:
        save_alpha_dir.mkdir(parents=True, exist_ok=True)
        alpha_path = save_alpha_dir / f"layer_{layer:02d}_{alpha_mode}_alpha.pt"
        torch.save(
            {
                "schema_version": "lynn-a100-w4a8-prefix-margin-alpha-overlay-v1",
                "layer": layer,
                "alpha_mode": alpha_mode,
                "shape": list(alpha.shape),
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "alpha": alpha.detach().cpu(),
                "folding_rule": "down_proj[expert, :, channel] *= alpha[expert, channel] for expert mode; down_proj[:, :, channel] *= alpha[channel] for shared mode",
            },
            alpha_path,
        )

    eval_cases = []
    for c in cases:
        out, _ = _active_moe_with_alpha(c["hidden"], w, top_k=top_k, alpha=alpha, mode="full", fmt=fmt)
        eval_cases.append(
            {
                "prompt_id": c["prompt_id"],
                "decode_index": c["decode_index"],
                "input_token_id": c["input_token_id"],
                "before": c["baseline_diff"],
                "after": _diff(c["ref"], out),
                "route": c["route"],
            }
        )
    before_max = max(c["before"]["rel_l2"] for c in eval_cases)
    after_max = max(c["after"]["rel_l2"] for c in eval_cases)
    result = {
        "layer": layer,
        "alpha_mode": alpha_mode,
        "case_count": len(cases),
        "history": history,
        "before": {
            "max_rel_l2": before_max,
            "mean_rel_l2": sum(c["before"]["rel_l2"] for c in eval_cases) / len(eval_cases),
            "min_cosine": min(c["before"]["cosine"] for c in eval_cases),
        },
        "after": {
            "max_rel_l2": after_max,
            "mean_rel_l2": sum(c["after"]["rel_l2"] for c in eval_cases) / len(eval_cases),
            "min_cosine": min(c["after"]["cosine"] for c in eval_cases),
        },
        "improvement": {
            "max_rel_l2_delta": before_max - after_max,
            "max_rel_l2_ratio": after_max / max(before_max, 1e-12),
        },
        "alpha_stats": _alpha_stats(alpha),
        "alpha_path": None if alpha_path is None else str(alpha_path),
        "cases": eval_cases,
    }
    del log_alpha, alpha
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 24, 26, 29, 30])
    parser.add_argument("--prompts", nargs="*", default=[])
    parser.add_argument("--prompt-specs-file")
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--force-prefix", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--fmt", default="e4m3", choices=["e4m3", "e5m2"])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--alpha-mode", default="shared", choices=["shared", "expert"])
    parser.add_argument("--alpha-min", type=float, default=0.75)
    parser.add_argument("--alpha-max", type=float, default=1.25)
    parser.add_argument("--save-alpha-dir")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    specs = _load_specs(args.prompt_specs_file, args.prompts)
    if not specs:
        raise ValueError("provide --prompt-specs-file or --prompts")

    start = time.time()
    runner = LynnIncrementalRunner(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
        max_seq_len=4096,
        verbose=True,
    )
    cases_by_layer = _collect_prefix_cases(
        runner,
        specs=specs,
        layers=args.layers,
        prefix_steps=args.prefix_steps,
        fmt=args.fmt,
        force_prefix=args.force_prefix,
        use_chat_template=args.use_chat_template,
    )
    save_alpha_dir = Path(args.save_alpha_dir) if args.save_alpha_dir else None
    layer_results = [
        _train_layer_alpha(
            runner,
            layer=layer,
            cases=cases_by_layer[layer],
            fmt=args.fmt,
            steps=args.steps,
            lr=args.lr,
            reg=args.reg,
            alpha_mode=args.alpha_mode,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            save_alpha_dir=save_alpha_dir,
        )
        for layer in args.layers
    ]
    before_max = max(r["before"]["max_rel_l2"] for r in layer_results)
    after_max = max(r["after"]["max_rel_l2"] for r in layer_results)
    improved = after_max < before_max
    decision = (
        "GREEN: prefix-margin foldable alpha improves first-token structured drift; fold and run format/generation gates."
        if improved and after_max <= 0.03
        else (
            "AMBER: prefix-margin alpha improves drift but does not fully clear local threshold; fold only as a research variant."
            if improved
            else "RED: prefix-margin alpha did not improve local drift; change layer set or alpha mode."
        )
    )
    result = {
        "schema_version": "lynn-a100-w4a8-prefix-margin-recovery-probe-v1",
        "model": args.model,
        "format": args.fmt,
        "layers": args.layers,
        "prompt_specs_file": args.prompt_specs_file,
        "prompt_count": len(specs),
        "prefix_steps": args.prefix_steps,
        "force_prefix": args.force_prefix,
        "use_chat_template": args.use_chat_template,
        "steps": args.steps,
        "lr": args.lr,
        "reg": args.reg,
        "alpha_mode": args.alpha_mode,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "save_alpha_dir": args.save_alpha_dir,
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
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if decision.startswith("GREEN") else (1 if decision.startswith("AMBER") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
