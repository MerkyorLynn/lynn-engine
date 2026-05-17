#!/usr/bin/env python3
"""Train a Lynn MTP sidecar on iterative base-greedy token positions.

`a100_mtp_fc_calibration_train.py` proves prompt-boundary next-token learning.
`a100_mtp_iterative_accept_probe.py` then showed the boundary: a sidecar can be
8/8 on first-token heldout and still collapse to low accept rate once the base
greedy path advances.

This trainer collects token-position cases along the frozen base model's greedy
path, then trains selected MTP sidecar tensors to predict the base next token at
each position. It is still a quality gate, not a serving TPS claim.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights, _topk  # noqa: E402


def _load_prompts(path: str | None, inline: list[str]) -> list[dict[str, Any]]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = inline
    specs: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            specs.append({"id": str(idx), "prompt": item, "weight": 1.0})
        elif isinstance(item, dict):
            specs.append(
                {
                    "id": str(item.get("id", idx)),
                    "prompt": str(item["prompt"]),
                    "weight": float(item.get("weight", 1.0)),
                }
            )
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return specs


def _case_from_hidden(
    *,
    runner: LynnIncrementalRunner,
    prompt_idx: int,
    prompt_id: str,
    prompt: str,
    prompt_weight: float,
    step: int,
    current_token_id: int,
    current_pos: int,
    base_hidden: torch.Tensor,
    first_token_weight: float,
    step1_weight: float,
    later_token_weight: float,
) -> dict[str, Any]:
    base_logits = runner._lm_head_logits(_rms_norm(base_hidden, runner.outside["model.language_model.norm.weight"]))
    label_id = int(base_logits[0].argmax().item())
    token_weight = first_token_weight if step == 0 else step1_weight if step == 1 else later_token_weight
    case_weight = prompt_weight * token_weight
    return {
        "prompt_idx": prompt_idx,
        "id": prompt_id,
        "prompt": prompt,
        "step": step,
        "weight": float(case_weight),
        "current_pos": int(current_pos),
        "current_token_id": int(current_token_id),
        "current_token_text": runner.tokenizer.decode([int(current_token_id)]),
        "base_hidden": base_hidden.detach().contiguous(),
        "input_embed": runner.outside["model.language_model.embed_tokens.weight"][int(current_token_id)]
        .view(1, 1, -1)
        .detach()
        .contiguous(),
        "label_id": label_id,
        "label_text": runner.tokenizer.decode([label_id]),
        "base_topk": _topk(runner.tokenizer, base_logits, 5),
    }


@torch.no_grad()
def _collect_prompt_cases(
    *,
    runner: LynnIncrementalRunner,
    spec: dict[str, Any],
    prompt_idx: int,
    use_chat_template: bool,
    max_new: int,
    first_token_weight: float,
    step1_weight: float,
    later_token_weight: float,
) -> list[dict[str, Any]]:
    ids = _encode_prompt(runner.tokenizer, spec["prompt"], runner.device, use_chat_template=use_chat_template)
    state = LynnInferenceState(
        batch=1,
        max_seq_len=runner.max_seq_len,
        device=runner.device,
        dtype=runner.dtype,
    )
    h = F.embedding(ids, runner.outside["model.language_model.embed_tokens.weight"])
    pos = torch.arange(ids.shape[1], device=runner.device, dtype=torch.long).unsqueeze(0)
    for layer_idx in range(runner.n_layers):
        h = _prefill_layer(
            h,
            pos,
            LAYER_TYPES[layer_idx],
            runner.layer_weights[layer_idx],
            runner.layer_cfgs[layer_idx],
            state,
            layer_idx,
        )
    state.seq_len = int(ids.shape[1])

    cases: list[dict[str, Any]] = []
    current_hidden = h[:, -1:, :].contiguous()
    current_token_id = int(ids[0, -1].item())
    current_pos = int(ids.shape[1] - 1)
    case = _case_from_hidden(
        runner=runner,
        prompt_idx=prompt_idx,
        prompt_id=str(spec["id"]),
        prompt=str(spec["prompt"]),
        prompt_weight=float(spec.get("weight", 1.0)),
        step=0,
        current_token_id=current_token_id,
        current_pos=current_pos,
        base_hidden=current_hidden,
        first_token_weight=first_token_weight,
        step1_weight=step1_weight,
        later_token_weight=later_token_weight,
    )
    cases.append(case)
    next_id = int(case["label_id"])

    new_token_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    for step in range(1, max_new):
        if next_id in runner.stop_token_ids:
            break
        new_token_tensor.fill_(next_id)
        pos_id = int(state.seq_len)
        pos_tensor.fill_(pos_id)
        h_step = F.embedding(new_token_tensor, runner.outside["model.language_model.embed_tokens.weight"])
        for layer_idx in range(runner.n_layers):
            h_step = runner._decode_layer_fast(h_step, pos_tensor, state, layer_idx)
        state.seq_len += 1
        case = _case_from_hidden(
            runner=runner,
            prompt_idx=prompt_idx,
            prompt_id=str(spec["id"]),
            prompt=str(spec["prompt"]),
            prompt_weight=float(spec.get("weight", 1.0)),
            step=step,
            current_token_id=next_id,
            current_pos=pos_id,
            base_hidden=h_step.contiguous(),
            first_token_weight=first_token_weight,
            step1_weight=step1_weight,
            later_token_weight=later_token_weight,
        )
        cases.append(case)
        next_id = int(case["label_id"])
    return cases


def _collect_cases(
    *,
    runner: LynnIncrementalRunner,
    specs: list[dict[str, Any]],
    use_chat_template: bool,
    max_new: int,
    first_token_weight: float,
    step1_weight: float,
    later_token_weight: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for prompt_idx, spec in enumerate(specs):
        cases.extend(
            _collect_prompt_cases(
                runner=runner,
                spec=spec,
                prompt_idx=prompt_idx,
                use_chat_template=use_chat_template,
                max_new=max_new,
                first_token_weight=first_token_weight,
                step1_weight=step1_weight,
                later_token_weight=later_token_weight,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cases


def _filter_cases_by_step(cases: list[dict[str, Any]], steps: set[int] | None) -> list[dict[str, Any]]:
    if steps is None:
        return cases
    return [case for case in cases if int(case["step"]) in steps]


def _mtp_logits_for_case(
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    case: dict[str, Any],
) -> torch.Tensor:
    hidden_part = _rms_norm(case["base_hidden"], sidecar["mtp.pre_fc_norm_hidden.weight"])
    embed_part = _rms_norm(case["input_embed"], sidecar["mtp.pre_fc_norm_embedding.weight"])
    mtp_hidden = F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])
    pos = torch.tensor([[int(case["current_pos"])]], device=runner.device, dtype=torch.long)
    mtp_out = _layer_forward(mtp_hidden, pos, "full_attention", mtp_w, cfg)
    mtp_normed = _rms_norm(mtp_out, sidecar["mtp.norm.weight"])
    return runner._lm_head_logits(mtp_normed)


def _evaluate(
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for case_idx, case in enumerate(cases):
            logits = _mtp_logits_for_case(runner, sidecar, mtp_w, cfg, case)
            label = torch.tensor([case["label_id"]], device=runner.device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), label)
            pred_id = int(logits[0].argmax().item())
            losses.append(float(loss.item()))
            rows.append(
                {
                    "case_idx": case_idx,
                    "prompt_idx": case["prompt_idx"],
                    "id": case["id"],
                    "step": case["step"],
                    "weight": case["weight"],
                    "current_pos": case["current_pos"],
                    "current_token_id": case["current_token_id"],
                    "current_token_text": case["current_token_text"],
                    "label_id": case["label_id"],
                    "label_text": case["label_text"],
                    "draft_id": pred_id,
                    "draft_text": runner.tokenizer.decode([pred_id]),
                    "accepted": pred_id == case["label_id"],
                    "loss": float(loss.item()),
                    "mtp_topk": _topk(runner.tokenizer, logits, 5),
                }
            )
            del logits, label, loss
    accepted = sum(1 for row in rows if row["accepted"])
    by_step: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["step"])
        item = by_step.setdefault(key, {"events": 0, "accepted": 0})
        item["events"] += 1
        item["accepted"] += int(row["accepted"])
    for item in by_step.values():
        item["accept_rate"] = item["accepted"] / item["events"] if item["events"] else 0.0
    return {
        "case_count": len(rows),
        "accepted": accepted,
        "accept_rate": accepted / len(rows) if rows else 0.0,
        "mean_loss": statistics.fmean(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "by_step": by_step,
        "rows": rows,
    }


def _select_trainable(
    sidecar: dict[str, torch.Tensor],
    mode: str,
) -> tuple[list[str], list[torch.Tensor]]:
    for tensor in sidecar.values():
        tensor.requires_grad_(False)
    if mode == "fc":
        keys = ["mtp.fc.weight"]
    elif mode == "fc_norms":
        keys = [
            "mtp.fc.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]
    elif mode == "fc_mtp_layer":
        keys = ["mtp.fc.weight", "mtp.norm.weight"]
        keys.extend(sorted(key for key in sidecar if key.startswith("mtp.layers.0.")))
    else:
        raise ValueError(f"unknown trainable mode: {mode}")
    params = []
    for key in keys:
        if key not in sidecar:
            raise KeyError(f"sidecar missing trainable tensor {key}")
        sidecar[key].requires_grad_(True)
        params.append(sidecar[key])
    return keys, params


def _train(
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    steps: int,
    lr: float,
    weight_decay: float,
    trainable: str,
) -> tuple[list[str], list[dict[str, float]]]:
    keys, params = _select_trainable(sidecar, trainable)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_weight = sum(float(case.get("weight", 1.0)) for case in cases) or 1.0
    history: list[dict[str, float]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        accepted = 0
        for case in cases:
            logits = _mtp_logits_for_case(runner, sidecar, mtp_w, cfg, case)
            label = torch.tensor([case["label_id"]], device=runner.device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), label)
            (loss * (float(case.get("weight", 1.0)) / total_weight)).backward()
            total_loss += float(loss.detach().item())
            accepted += int(int(logits[0].argmax().item()) == case["label_id"])
            del logits, label, loss
        opt.step()
        history.append(
            {
                "step": step + 1,
                "mean_loss": total_loss / max(1, len(cases)),
                "accept_rate": accepted / max(1, len(cases)),
            }
        )
    for tensor in sidecar.values():
        tensor.requires_grad_(False)
    return keys, history


def _save_sidecar(sidecar: dict[str, torch.Tensor], out_dir: Path, source_sidecar: Path, report: dict[str, Any]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "mtp.safetensors"
    save_file(
        {key: tensor.detach().cpu() for key, tensor in sidecar.items()},
        str(out_file),
        metadata={
            "lynn_mtp_iterative_train": "true",
            "source_sidecar": str(source_sidecar),
            "train_accept_rate_after": str(report["train_after"]["accept_rate"]),
            "eval_accept_rate_after": str(report.get("eval_after", {}).get("accept_rate")),
        },
    )
    return str(out_file)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--train-prompts-file", required=True)
    ap.add_argument("--eval-prompts-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-sidecar-dir")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new-train", type=int, default=8)
    ap.add_argument("--max-new-eval", type=int, default=16)
    ap.add_argument("--first-token-weight", type=float, default=2.0)
    ap.add_argument("--step1-weight", type=float, default=1.0)
    ap.add_argument("--later-token-weight", type=float, default=1.0)
    ap.add_argument("--train-steps", type=int, nargs="*", default=None)
    ap.add_argument("--trainable", default="fc", choices=["fc", "fc_norms", "fc_mtp_layer"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    train_specs = _load_prompts(args.train_prompts_file, [])
    eval_specs = _load_prompts(args.eval_prompts_file, []) if args.eval_prompts_file else []

    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    sidecar, sidecar_inventory = _load_sidecar(Path(args.sidecar_file), args.device, dtype)
    mtp_w = _mtp_layer_weights(sidecar)
    cfg = _mtp_cfg(runner, mtp_w)

    train_cases_all = _collect_cases(
        runner=runner,
        specs=train_specs,
        use_chat_template=args.use_chat_template,
        max_new=args.max_new_train,
        first_token_weight=args.first_token_weight,
        step1_weight=args.step1_weight,
        later_token_weight=args.later_token_weight,
    )
    train_cases = _filter_cases_by_step(train_cases_all, set(args.train_steps) if args.train_steps else None)
    if not train_cases:
        raise ValueError("no training cases left after --train-steps filter")
    eval_cases = (
        _collect_cases(
            runner=runner,
            specs=eval_specs,
            use_chat_template=args.use_chat_template,
            max_new=args.max_new_eval,
            first_token_weight=args.first_token_weight,
            step1_weight=args.step1_weight,
            later_token_weight=args.later_token_weight,
        )
        if eval_specs
        else []
    )

    train_before = _evaluate(runner, sidecar, mtp_w, cfg, train_cases)
    eval_before = _evaluate(runner, sidecar, mtp_w, cfg, eval_cases) if eval_cases else None
    trainable_keys, history = _train(
        runner,
        sidecar,
        mtp_w,
        cfg,
        train_cases,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        trainable=args.trainable,
    )
    train_after = _evaluate(runner, sidecar, mtp_w, cfg, train_cases)
    eval_after = _evaluate(runner, sidecar, mtp_w, cfg, eval_cases) if eval_cases else None

    report: dict[str, Any] = {
        "schema_version": "lynn-a100-mtp-iterative-train-v1",
        "decision": (
            "GREEN: iterative MTP training improved eval accept rate above 70%."
            if eval_after and eval_after["accept_rate"] >= 0.70 and eval_after["accept_rate"] > (eval_before or {}).get("accept_rate", 0.0)
            else "AMBER: iterative MTP training completed; inspect train/eval accept rates."
        ),
        "base_model": args.base_model,
        "source_sidecar_file": args.sidecar_file,
        "use_chat_template": args.use_chat_template,
        "dtype": args.dtype,
        "trainable": args.trainable,
        "trainable_tensors": trainable_keys,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_new_train": args.max_new_train,
        "max_new_eval": args.max_new_eval,
        "first_token_weight": args.first_token_weight,
        "step1_weight": args.step1_weight,
        "later_token_weight": args.later_token_weight,
        "train_steps": args.train_steps,
        "weights_saved": bool(args.out_sidecar_dir),
        "train_case_count_before_filter": len(train_cases_all),
        "train_case_count": len(train_cases),
        "eval_case_count": len(eval_cases),
        "train_before": train_before,
        "eval_before": eval_before,
        "history": history,
        "train_after": train_after,
        "eval_after": eval_after,
        "train_cases": [
            {
                "case_idx": idx,
                "prompt_idx": case["prompt_idx"],
                "id": case["id"],
                "step": case["step"],
                "weight": case["weight"],
                "current_pos": case["current_pos"],
                "current_token_id": case["current_token_id"],
                "current_token_text": case["current_token_text"],
                "label_id": case["label_id"],
                "label_text": case["label_text"],
                "base_topk": case["base_topk"],
            }
            for idx, case in enumerate(train_cases)
        ],
        "sidecar": sidecar_inventory,
    }
    if args.out_sidecar_dir:
        report["trained_sidecar_file"] = _save_sidecar(sidecar, Path(args.out_sidecar_dir), Path(args.sidecar_file), report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "train_before": {
                    key: train_before[key]
                    for key in ("case_count", "accepted", "accept_rate", "mean_loss")
                },
                "train_after": {
                    key: train_after[key]
                    for key in ("case_count", "accepted", "accept_rate", "mean_loss")
                },
                "eval_before": None
                if not eval_before
                else {key: eval_before[key] for key in ("case_count", "accepted", "accept_rate", "mean_loss")},
                "eval_after": None
                if not eval_after
                else {key: eval_after[key] for key in ("case_count", "accepted", "accept_rate", "mean_loss")},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del runner, sidecar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
