#!/usr/bin/env python3
"""Train/evaluate the Lynn MTP `fc` bridge on a small calibration prompt set.

This is the first Stage-1 MTP gate after the single-prompt smoke:

* frozen base model;
* frozen aligned MTP sidecar except `mtp.fc.weight`;
* target is the base model greedy next token for each prompt;
* metric is one-token draft accept rate: MTP argmax == base argmax.

It is intentionally small and honest. A GREEN here does not claim speculative
TPS; it says the MTP head can learn the immediate base-token manifold enough to
justify a larger head-only trainer and iterative accept-rate evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _rms_norm  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_forward_smoke import (  # noqa: E402
    _base_prefill_last_hidden,
    _load_sidecar,
    _mtp_layer_weights,
    _topk,
)


def _load_prompts(path: str | None, inline: list[str]) -> list[dict[str, Any]]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = inline
    prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            prompts.append({"id": str(idx), "prompt": item})
        elif isinstance(item, dict):
            if "prompt" not in item:
                raise KeyError(f"prompt spec {idx} is missing `prompt`")
            prompts.append({"id": str(item.get("id", idx)), "prompt": str(item["prompt"])})
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return prompts


def _mtp_cfg(runner: LynnIncrementalRunner, mtp_w: dict[str, torch.Tensor]) -> dict[str, Any]:
    text_cfg = runner.cfg.get("text_config") or runner.cfg
    cfg = dict(text_cfg)
    cfg["num_experts"] = int(mtp_w["mlp.experts.gate_up_proj"].shape[0])
    cfg["num_experts_per_tok"] = int(text_cfg.get("num_experts_per_tok", 8))
    cfg["expert_intermediate"] = int(mtp_w["mlp.experts.down_proj"].shape[-1])
    cfg["layer_idx"] = 0
    return cfg


def _collect_cases(
    runner: LynnIncrementalRunner,
    specs: list[dict[str, Any]],
    *,
    use_chat_template: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for prompt_idx, spec in enumerate(specs):
        base_hidden, input_embed, ids, last_token_id, last_pos = _base_prefill_last_hidden(
            runner,
            str(spec["prompt"]),
            use_chat_template=use_chat_template,
        )
        base_logits = runner._lm_head_logits(
            _rms_norm(base_hidden, runner.outside["model.language_model.norm.weight"])
        )
        label_id = int(base_logits[0].argmax().item())
        cases.append(
            {
                "prompt_idx": prompt_idx,
                "id": spec["id"],
                "prompt": spec["prompt"],
                "prompt_tokens": int(ids.numel()),
                "last_token_id": last_token_id,
                "last_token_text": runner.tokenizer.decode([last_token_id]),
                "last_pos": last_pos,
                "base_hidden": base_hidden.detach(),
                "input_embed": input_embed.detach(),
                "label_id": label_id,
                "label_text": runner.tokenizer.decode([label_id]),
                "base_topk": _topk(runner.tokenizer, base_logits, 5),
            }
        )
        del ids, base_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cases


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
    pos = torch.tensor([[int(case["last_pos"])]], device=runner.device, dtype=torch.long)
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
        for case in cases:
            logits = _mtp_logits_for_case(runner, sidecar, mtp_w, cfg, case)
            label = torch.tensor([case["label_id"]], device=runner.device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), label)
            pred_id = int(logits[0].argmax().item())
            losses.append(float(loss.item()))
            rows.append(
                {
                    "prompt_idx": case["prompt_idx"],
                    "id": case["id"],
                    "prompt_tokens": case["prompt_tokens"],
                    "label_id": case["label_id"],
                    "label_text": case["label_text"],
                    "draft_id": pred_id,
                    "draft_text": runner.tokenizer.decode([pred_id]),
                    "accepted": pred_id == case["label_id"],
                    "loss": float(loss.item()),
                    "mtp_topk": _topk(runner.tokenizer, logits, 5),
                }
            )
    accept = sum(1 for row in rows if row["accepted"])
    return {
        "case_count": len(rows),
        "accepted": accept,
        "accept_rate": accept / len(rows) if rows else 0.0,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "rows": rows,
    }


def _train_fc(
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    steps: int,
    lr: float,
    weight_decay: float,
) -> list[dict[str, float]]:
    for tensor in sidecar.values():
        tensor.requires_grad_(False)
    sidecar["mtp.fc.weight"].requires_grad_(True)
    opt = torch.optim.AdamW([sidecar["mtp.fc.weight"]], lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        accepted = 0
        for case in cases:
            logits = _mtp_logits_for_case(runner, sidecar, mtp_w, cfg, case)
            label = torch.tensor([case["label_id"]], device=runner.device, dtype=torch.long)
            loss = F.cross_entropy(logits.float(), label)
            (loss / max(1, len(cases))).backward()
            total_loss += float(loss.detach().item())
            accepted += int(int(logits[0].argmax().item()) == case["label_id"])
            del logits, loss, label
        opt.step()
        history.append(
            {
                "step": step + 1,
                "mean_loss": total_loss / max(1, len(cases)),
                "accept_rate": accepted / max(1, len(cases)),
            }
        )
    sidecar["mtp.fc.weight"].requires_grad_(False)
    return history


def _save_sidecar(sidecar: dict[str, torch.Tensor], out_dir: Path, source_sidecar: Path, report: dict[str, Any]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "mtp.safetensors"
    tensors = {key: tensor.detach().cpu() for key, tensor in sidecar.items()}
    save_file(
        tensors,
        str(out_file),
        metadata={
            "lynn_mtp_fc_calibration": "true",
            "source_sidecar": str(source_sidecar),
            "accept_rate_after": str(report["after"]["accept_rate"]),
        },
    )
    return str(out_file)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-sidecar-dir")
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=[
        "Return one JSON object with keys city and unit for Tokyo in celsius. No markdown.",
        "Return one JSON object with key status and value ok. No markdown.",
        "Output exactly one JSON arguments object for get_weather with city Tokyo and unit celsius. No markdown.",
        "Answer only with the distance and unit: a train travels 60 mph for 2.5 hours.",
    ])
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    sidecar, sidecar_inventory = _load_sidecar(Path(args.sidecar_file), args.device, dtype)
    mtp_w = _mtp_layer_weights(sidecar)
    cfg = _mtp_cfg(runner, mtp_w)
    cases = _collect_cases(runner, specs, use_chat_template=args.use_chat_template)
    before = _evaluate(runner, sidecar, mtp_w, cfg, cases)
    history = _train_fc(
        runner,
        sidecar,
        mtp_w,
        cfg,
        cases,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    after = _evaluate(runner, sidecar, mtp_w, cfg, cases)
    report: dict[str, Any] = {
        "schema_version": "lynn-a100-mtp-fc-calibration-train-v1",
        "decision": (
            "GREEN: fc-only MTP calibration improved one-token accept rate."
            if after["accept_rate"] > before["accept_rate"]
            else "AMBER: fc-only MTP calibration completed without accept-rate improvement."
        ),
        "base_model": args.base_model,
        "source_sidecar_file": args.sidecar_file,
        "use_chat_template": args.use_chat_template,
        "dtype": args.dtype,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "trainable_tensors": ["mtp.fc.weight"],
        "weights_saved": bool(args.out_sidecar_dir),
        "before": before,
        "history": history,
        "after": after,
        "cases": [
            {
                "prompt_idx": case["prompt_idx"],
                "id": case["id"],
                "prompt": case["prompt"],
                "prompt_tokens": case["prompt_tokens"],
                "label_id": case["label_id"],
                "label_text": case["label_text"],
                "base_topk": case["base_topk"],
            }
            for case in cases
        ],
        "sidecar": sidecar_inventory,
    }
    if args.out_sidecar_dir:
        report["trained_sidecar_file"] = _save_sidecar(sidecar, Path(args.out_sidecar_dir), Path(args.sidecar_file), report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("decision", "before", "after", "history")}, ensure_ascii=False, indent=2))
    del runner, sidecar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
