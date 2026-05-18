#!/usr/bin/env python3
"""P120: sweep MTP sidecar alignment hypotheses.

Official Qwen3.6-35B MTP shape/forward can be GREEN while iterative accept is
0%. This probe checks whether the miss is a simple serving-contract shift:

* position id: current position versus next position;
* embedding input: current token embedding versus oracle next-token embedding;
* target: immediate base next token versus the token after that.

The oracle variants are not deployable. They are a diagnostic: if an oracle
variant lights up, the serving-side proposer contract is shifted. If all
variants stay dead, the sidecar is not aligned with the local base runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _rms_norm  # noqa: E402
from engine.mtp_sidecar import load_mtp_sidecar, mtp_layer_config, mtp_layer_forward, mtp_layer_weights  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_forward_smoke import _topk  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _load_prompts  # noqa: E402


DEFAULT_PROMPTS = [
    "Return one JSON object with keys city and unit for Berlin in metric units. No markdown.",
    "Output exactly one JSON arguments object for translate_text with text hello and target_language Japanese. No markdown.",
    "用一句中文短句说明 MoE router 的作用。必须以 router 开头。",
]


def _variant_logits(
    *,
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    mtp_cfg: dict[str, Any],
    case: dict[str, Any],
    embed_token_id: int,
    pos_offset: int,
) -> torch.Tensor:
    input_embed = runner.outside["model.language_model.embed_tokens.weight"][int(embed_token_id)].view(1, 1, -1)
    hidden_part = _rms_norm(case["base_hidden"], sidecar["mtp.pre_fc_norm_hidden.weight"])
    embed_part = _rms_norm(input_embed, sidecar["mtp.pre_fc_norm_embedding.weight"])
    mtp_hidden = F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])
    pos = torch.tensor([[int(case["current_pos"]) + int(pos_offset)]], device=runner.device, dtype=torch.long)
    mtp_out = mtp_layer_forward(mtp_hidden, pos, mtp_w, mtp_cfg)
    mtp_normed = _rms_norm(mtp_out, sidecar["mtp.norm.weight"])
    return runner._lm_head_logits(mtp_normed)


def _rank(logits: torch.Tensor, token_id: int) -> int:
    scores = logits[0].float()
    score = scores[int(token_id)]
    return int(torch.count_nonzero(scores > score).item()) + 1


def _loss(logits: torch.Tensor, token_id: int) -> float:
    label = torch.tensor([int(token_id)], device=logits.device, dtype=torch.long)
    return float(F.cross_entropy(logits.float(), label).item())


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"events": 0}
    ranks = [int(r["target_rank"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    return {
        "events": len(rows),
        "top1": sum(1 for r in rows if int(r["target_rank"]) == 1),
        "top5": sum(1 for r in rows if int(r["target_rank"]) <= 5),
        "top8": sum(1 for r in rows if int(r["target_rank"]) <= 8),
        "top32": sum(1 for r in rows if int(r["target_rank"]) <= 32),
        "top1_rate": sum(1 for r in rows if int(r["target_rank"]) == 1) / len(rows),
        "top8_rate": sum(1 for r in rows if int(r["target_rank"]) <= 8) / len(rows),
        "median_rank": statistics.median(ranks),
        "mean_loss": statistics.fmean(losses),
        "min_loss": min(losses),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    sidecar, inventory = load_mtp_sidecar(args.sidecar_file, device=args.device, dtype=dtype)
    mtp_w = mtp_layer_weights(sidecar)
    mtp_cfg = mtp_layer_config(runner.cfg, mtp_w)

    cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=args.use_chat_template,
        force_prefix_from_spec=False,
        skip_forced_prefix_cases=False,
        max_new=args.max_new,
        first_token_weight=1.0,
        step1_weight=1.0,
        later_token_weight=1.0,
    )
    next_case_by_key = {
        (int(case["prompt_idx"]), int(case["step"])): case
        for case in cases
    }

    variant_rows: dict[str, list[dict[str, Any]]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    embed_modes = ("current", "label_oracle")
    target_modes = ("same_label", "next_label")
    pos_offsets = (-1, 0, 1, 2)
    with torch.no_grad():
        for case_idx, case in enumerate(cases):
            next_case = next_case_by_key.get((int(case["prompt_idx"]), int(case["step"]) + 1))
            for embed_mode in embed_modes:
                embed_token_id = int(case["current_token_id"]) if embed_mode == "current" else int(case["label_id"])
                for pos_offset in pos_offsets:
                    logits = _variant_logits(
                        runner=runner,
                        sidecar=sidecar,
                        mtp_w=mtp_w,
                        mtp_cfg=mtp_cfg,
                        case=case,
                        embed_token_id=embed_token_id,
                        pos_offset=pos_offset,
                    )
                    draft_id = int(logits[0].argmax().item())
                    for target_mode in target_modes:
                        if target_mode == "same_label":
                            target = case
                        else:
                            if next_case is None:
                                continue
                            target = next_case
                        target_id = int(target["label_id"])
                        name = f"embed={embed_mode};pos_offset={pos_offset};target={target_mode}"
                        row = {
                            "case_idx": case_idx,
                            "prompt_idx": int(case["prompt_idx"]),
                            "step": int(case["step"]),
                            "current_pos": int(case["current_pos"]),
                            "embed_token_id": embed_token_id,
                            "embed_text": runner.tokenizer.decode([embed_token_id]),
                            "target_id": target_id,
                            "target_text": runner.tokenizer.decode([target_id]),
                            "draft_id": draft_id,
                            "draft_text": runner.tokenizer.decode([draft_id]),
                            "target_rank": _rank(logits, target_id),
                            "loss": _loss(logits, target_id),
                            "topk": _topk(runner.tokenizer, logits, args.top_k),
                        }
                        variant_rows.setdefault(name, []).append(row)
                        if len(examples.setdefault(name, [])) < 4:
                            examples[name].append(row)

    summaries = {name: _summarize(rows) for name, rows in sorted(variant_rows.items())}
    best = sorted(
        summaries.items(),
        key=lambda kv: (
            -float(kv[1].get("top1_rate", 0.0)),
            -float(kv[1].get("top8_rate", 0.0)),
            float(kv[1].get("mean_loss", 1e9)),
        ),
    )[:8]
    report = {
        "schema_version": "lynn-p120-mtp-alignment-sweep-v1",
        "decision": (
            "AMBER: at least one MTP alignment variant has nonzero top1."
            if any(s.get("top1", 0) for s in summaries.values())
            else "RED: no swept MTP alignment variant produced top1 acceptance."
        ),
        "model": args.model,
        "sidecar_file": args.sidecar_file,
        "dtype": args.dtype,
        "use_chat_template": args.use_chat_template,
        "mtp_layer_moe": os.environ.get("LYNN_MTP_LAYER_MOE", "decode_slot_sorted"),
        "case_count": len(cases),
        "sidecar_tensor_count": len(inventory.get("tensors", {})),
        "summaries": summaries,
        "best": [{"variant": name, **summary} for name, summary in best],
        "examples": {name: examples[name] for name, _ in best},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "best": report["best"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
