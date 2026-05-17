#!/usr/bin/env python3
"""P108: compare BF16, native FP4, and fake-native-FP4 lm_head boundaries.

MTP v27-v31 showed that a sidecar can cross the credit bar with BF16 lm_head
but stay below it with `LYNN_NATIVE_FP4_LM_HEAD=1`. This probe isolates the
projection boundary: along the resident runner's greedy path, compare top-1 and
top-k agreement between the real native FP4 `_scaled_mm` head and a fixed
fake-quantized BF16 surrogate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from scripts.a100_mtp_iterative_train import (  # noqa: E402
    _fake_native_fp4_activation,
    _fake_native_fp4_lm_head_weight,
)


DEFAULT_PROMPTS = [
    "Return one JSON object with keys city and unit for Berlin in metric units. No markdown.",
    "Output exactly one JSON arguments object for translate_text with text hello and target_language Japanese. No markdown.",
    "Only output a Python code block defining slugify(text: str) -> str. It must lowercase and replace spaces with hyphens. No explanation.",
    "用一句中文短句说明 MoE router 的作用。必须以 router 开头。",
]


def _load_prompt_specs(path: str | None, inline: list[str]) -> list[dict[str, str]]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = inline
    specs: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, str):
            specs.append({"id": str(idx), "prompt": item})
        elif isinstance(item, dict):
            specs.append({"id": str(item.get("id", idx)), "prompt": str(item["prompt"])})
        else:
            raise TypeError(f"prompt spec must be string or object, got {type(item)}")
    return specs


def _top_ids(logits: torch.Tensor, k: int) -> list[int]:
    return [int(x) for x in torch.topk(logits.float(), k=k, dim=-1).indices[0].tolist()]


def _rank_of(logits: torch.Tensor, token_id: int) -> int:
    score = logits[0, int(token_id)]
    return int((logits[0] > score).sum().item()) + 1


def _row(
    *,
    runner: LynnIncrementalRunner,
    hidden: torch.Tensor,
    fake_weight: torch.Tensor,
    step: int,
    current_token_id: int,
    current_pos: int,
    top_k: int,
) -> dict[str, Any]:
    h2d = hidden[:, -1, :] if hidden.ndim == 3 else hidden
    runner.native_fp4_lm_head_enabled = False
    bf16_logits = runner._lm_head_logits(hidden)
    fake_logits = F.linear(h2d, fake_weight)
    fake_act_logits = F.linear(_fake_native_fp4_activation(h2d), fake_weight)
    runner.native_fp4_lm_head_enabled = True
    native_logits = runner._lm_head_logits(hidden)
    native_id = int(native_logits[0].argmax().item())
    bf16_id = int(bf16_logits[0].argmax().item())
    fake_id = int(fake_logits[0].argmax().item())
    fake_act_id = int(fake_act_logits[0].argmax().item())
    native_top = set(_top_ids(native_logits, top_k))
    fake_top = set(_top_ids(fake_logits, top_k))
    fake_act_top = set(_top_ids(fake_act_logits, top_k))
    bf16_top = set(_top_ids(bf16_logits, top_k))
    return {
        "step": int(step),
        "current_pos": int(current_pos),
        "current_token_id": int(current_token_id),
        "current_token_text": runner.tokenizer.decode([int(current_token_id)]),
        "native_id": native_id,
        "native_text": runner.tokenizer.decode([native_id]),
        "bf16_id": bf16_id,
        "bf16_text": runner.tokenizer.decode([bf16_id]),
        "fake_id": fake_id,
        "fake_text": runner.tokenizer.decode([fake_id]),
        "fake_act_id": fake_act_id,
        "fake_act_text": runner.tokenizer.decode([fake_act_id]),
        "bf16_matches_native": bf16_id == native_id,
        "fake_matches_native": fake_id == native_id,
        "fake_act_matches_native": fake_act_id == native_id,
        "native_in_fake_topk": native_id in fake_top,
        "native_in_fake_act_topk": native_id in fake_act_top,
        "native_in_bf16_topk": native_id in bf16_top,
        "fake_rank_of_native": _rank_of(fake_logits, native_id),
        "fake_act_rank_of_native": _rank_of(fake_act_logits, native_id),
        "bf16_rank_of_native": _rank_of(bf16_logits, native_id),
        "topk_overlap_fake_native": len(fake_top & native_top),
        "topk_overlap_fake_act_native": len(fake_act_top & native_top),
        "topk_overlap_bf16_native": len(bf16_top & native_top),
    }


@torch.no_grad()
def _probe_prompt(
    *,
    runner: LynnIncrementalRunner,
    fake_weight: torch.Tensor,
    prompt_id: str,
    prompt: str,
    use_chat_template: bool,
    max_new: int,
    top_k: int,
) -> dict[str, Any]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
    state = LynnInferenceState(batch=1, max_seq_len=runner.max_seq_len, device=runner.device, dtype=runner.dtype)
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

    rows = []
    current_hidden = h[:, -1:, :].contiguous()
    current_token_id = int(ids[0, -1].item())
    current_pos = int(ids.shape[1] - 1)
    row = _row(
        runner=runner,
        hidden=current_hidden,
        fake_weight=fake_weight,
        step=0,
        current_token_id=current_token_id,
        current_pos=current_pos,
        top_k=top_k,
    )
    rows.append(row)
    next_id = int(row["native_id"])

    token = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    pos_tensor = torch.empty((1, 1), device=runner.device, dtype=torch.long)
    for step in range(1, max_new):
        if next_id in runner.stop_token_ids:
            break
        token.fill_(next_id)
        pos_id = int(state.seq_len)
        pos_tensor.fill_(pos_id)
        h_step = F.embedding(token, runner.outside["model.language_model.embed_tokens.weight"])
        for layer_idx in range(runner.n_layers):
            h_step = runner._decode_layer_fast(h_step, pos_tensor, state, layer_idx)
        state.seq_len += 1
        row = _row(
            runner=runner,
            hidden=h_step.contiguous(),
            fake_weight=fake_weight,
            step=step,
            current_token_id=next_id,
            current_pos=pos_id,
            top_k=top_k,
        )
        rows.append(row)
        next_id = int(row["native_id"])

    return {"id": prompt_id, "prompt": prompt, "events": len(rows), "rows": rows}


def _summary(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for prompt in prompts for row in prompt["rows"]]
    total = len(rows)
    fake_matches = sum(1 for row in rows if row["fake_matches_native"])
    fake_act_matches = sum(1 for row in rows if row["fake_act_matches_native"])
    bf16_matches = sum(1 for row in rows if row["bf16_matches_native"])
    fake_topk = sum(1 for row in rows if row["native_in_fake_topk"])
    fake_act_topk = sum(1 for row in rows if row["native_in_fake_act_topk"])
    bf16_topk = sum(1 for row in rows if row["native_in_bf16_topk"])
    return {
        "events": total,
        "fake_top1_matches_native": fake_matches,
        "fake_top1_match_rate": fake_matches / total if total else None,
        "fake_act_top1_matches_native": fake_act_matches,
        "fake_act_top1_match_rate": fake_act_matches / total if total else None,
        "bf16_top1_matches_native": bf16_matches,
        "bf16_top1_match_rate": bf16_matches / total if total else None,
        "native_in_fake_topk": fake_topk,
        "native_in_fake_topk_rate": fake_topk / total if total else None,
        "native_in_fake_act_topk": fake_act_topk,
        "native_in_fake_act_topk_rate": fake_act_topk / total if total else None,
        "native_in_bf16_topk": bf16_topk,
        "native_in_bf16_topk_rate": bf16_topk / total if total else None,
        "mean_fake_rank_of_native": statistics.fmean(row["fake_rank_of_native"] for row in rows) if rows else None,
        "mean_fake_act_rank_of_native": (
            statistics.fmean(row["fake_act_rank_of_native"] for row in rows) if rows else None
        ),
        "mean_bf16_rank_of_native": statistics.fmean(row["bf16_rank_of_native"] for row in rows) if rows else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompt_specs(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    if not runner.native_fp4_lm_head_enabled:
        runner._prepare_native_fp4_lm_head()
    fake_weight = _fake_native_fp4_lm_head_weight(runner.outside["lm_head.weight"])

    prompts = [
        _probe_prompt(
            runner=runner,
            fake_weight=fake_weight,
            prompt_id=spec["id"],
            prompt=spec["prompt"],
            use_chat_template=args.use_chat_template,
            max_new=args.max_new,
            top_k=args.top_k,
        )
        for spec in specs
    ]
    summary = _summary(prompts)
    report = {
        "schema_version": "lynn-p108-lm-head-native-surrogate-parity-v2",
        "decision": (
            "GREEN: activation-aware fake-native lm_head top-1 parity is at least 95%."
            if summary["fake_act_top1_match_rate"] is not None and summary["fake_act_top1_match_rate"] >= 0.95
            else "AMBER: activation-aware fake-native lm_head top-1 parity is below promotion threshold."
        ),
        "model": args.model,
        "use_chat_template": args.use_chat_template,
        "max_new": args.max_new,
        "top_k": args.top_k,
        "summary": summary,
        "prompts": prompts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
