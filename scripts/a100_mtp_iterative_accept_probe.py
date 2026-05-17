#!/usr/bin/env python3
"""Probe iterative one-token MTP accept rate along the base greedy path.

The fc calibration gate checks only the first token after each prompt. This
probe is the next step toward real speculative decode: it advances the frozen
base model greedily and asks the MTP sidecar to draft the next token at every
position. A draft is accepted when MTP argmax equals the base greedy argmax.

This is still a quality/accept-rate probe, not a TPS claim. It deliberately
uses the base token path as the authority so it can run before serving-side MTP
integration exists.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _layer_forward, _prefill_layer, _rms_norm  # noqa: E402
from engine.inference_state import LAYER_TYPES, LynnInferenceState  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner, _encode_prompt  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights, _topk  # noqa: E402


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


def _mtp_logits(
    *,
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    mtp_cfg: dict[str, Any],
    base_hidden: torch.Tensor,
    current_token_id: int,
    current_pos: int,
) -> torch.Tensor:
    input_embed = runner.outside["model.language_model.embed_tokens.weight"][current_token_id].view(1, 1, -1)
    hidden_part = _rms_norm(base_hidden, sidecar["mtp.pre_fc_norm_hidden.weight"])
    embed_part = _rms_norm(input_embed, sidecar["mtp.pre_fc_norm_embedding.weight"])
    mtp_hidden = F.linear(torch.cat([hidden_part, embed_part], dim=-1), sidecar["mtp.fc.weight"])
    pos = torch.tensor([[current_pos]], device=runner.device, dtype=torch.long)
    mtp_out = _layer_forward(mtp_hidden, pos, "full_attention", mtp_w, mtp_cfg)
    mtp_normed = _rms_norm(mtp_out, sidecar["mtp.norm.weight"])
    return runner._lm_head_logits(mtp_normed)


def _row(
    *,
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    mtp_cfg: dict[str, Any],
    prompt_id: str,
    step: int,
    current_token_id: int,
    current_pos: int,
    base_hidden: torch.Tensor,
    base_logits: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    label_id = int(base_logits[0].argmax().item())
    mtp_logits = _mtp_logits(
        runner=runner,
        sidecar=sidecar,
        mtp_w=mtp_w,
        mtp_cfg=mtp_cfg,
        base_hidden=base_hidden,
        current_token_id=current_token_id,
        current_pos=current_pos,
    )
    draft_id = int(mtp_logits[0].argmax().item())
    label = torch.tensor([label_id], device=runner.device, dtype=torch.long)
    loss = F.cross_entropy(mtp_logits.float(), label)
    return {
        "prompt_id": prompt_id,
        "step": step,
        "current_pos": current_pos,
        "current_token_id": current_token_id,
        "current_token_text": runner.tokenizer.decode([current_token_id]),
        "label_id": label_id,
        "label_text": runner.tokenizer.decode([label_id]),
        "draft_id": draft_id,
        "draft_text": runner.tokenizer.decode([draft_id]),
        "accepted": draft_id == label_id,
        "loss": float(loss.item()),
        "base_topk": _topk(runner.tokenizer, base_logits, top_k),
        "mtp_topk": _topk(runner.tokenizer, mtp_logits, top_k),
    }


@torch.no_grad()
def _probe_prompt(
    *,
    runner: LynnIncrementalRunner,
    sidecar: dict[str, torch.Tensor],
    mtp_w: dict[str, torch.Tensor],
    mtp_cfg: dict[str, Any],
    prompt_id: str,
    prompt: str,
    use_chat_template: bool,
    max_new: int,
    top_k: int,
) -> dict[str, Any]:
    ids = _encode_prompt(runner.tokenizer, prompt, runner.device, use_chat_template=use_chat_template)
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

    rows: list[dict[str, Any]] = []
    current_hidden = h[:, -1:, :].contiguous()
    current_token_id = int(ids[0, -1].item())
    current_pos = int(ids.shape[1] - 1)
    base_logits = runner._lm_head_logits(_rms_norm(current_hidden, runner.outside["model.language_model.norm.weight"]))
    row = _row(
        runner=runner,
        sidecar=sidecar,
        mtp_w=mtp_w,
        mtp_cfg=mtp_cfg,
        prompt_id=prompt_id,
        step=0,
        current_token_id=current_token_id,
        current_pos=current_pos,
        base_hidden=current_hidden,
        base_logits=base_logits,
        top_k=top_k,
    )
    rows.append(row)
    next_id = int(row["label_id"])

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
        base_logits = runner._lm_head_logits(
            _rms_norm(h_step, runner.outside["model.language_model.norm.weight"])
        )
        row = _row(
            runner=runner,
            sidecar=sidecar,
            mtp_w=mtp_w,
            mtp_cfg=mtp_cfg,
            prompt_id=prompt_id,
            step=step,
            current_token_id=next_id,
            current_pos=pos_id,
            base_hidden=h_step.contiguous(),
            base_logits=base_logits,
            top_k=top_k,
        )
        rows.append(row)
        next_id = int(row["label_id"])

    accepted = sum(1 for row in rows if row["accepted"])
    losses = [float(row["loss"]) for row in rows]
    return {
        "id": prompt_id,
        "prompt": prompt,
        "prompt_tokens": int(ids.numel()),
        "events": len(rows),
        "accepted": accepted,
        "accept_rate": accepted / len(rows) if rows else 0.0,
        "mean_loss": statistics.fmean(losses) if losses else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=[
        "Return one JSON object with keys city and unit for Berlin in metric units. No markdown.",
        "Output exactly one JSON arguments object for translate_text with text hello and target_language Japanese. No markdown.",
        "用一句中文短句说明 MoE router 的作用。必须以 router 开头。",
    ])
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompt_specs(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    sidecar, sidecar_inventory = _load_sidecar(Path(args.sidecar_file), args.device, dtype)
    for tensor in sidecar.values():
        tensor.requires_grad_(False)
    mtp_w = _mtp_layer_weights(sidecar)
    mtp_cfg = _mtp_cfg(runner, mtp_w)

    prompt_reports = [
        _probe_prompt(
            runner=runner,
            sidecar=sidecar,
            mtp_w=mtp_w,
            mtp_cfg=mtp_cfg,
            prompt_id=spec["id"],
            prompt=spec["prompt"],
            use_chat_template=args.use_chat_template,
            max_new=args.max_new,
            top_k=args.top_k,
        )
        for spec in specs
    ]
    total_events = sum(int(p["events"]) for p in prompt_reports)
    total_accepted = sum(int(p["accepted"]) for p in prompt_reports)
    losses = [float(row["loss"]) for p in prompt_reports for row in p["rows"]]
    report = {
        "schema_version": "lynn-a100-mtp-iterative-accept-probe-v1",
        "decision": (
            "GREEN: iterative one-token MTP accept rate is at least 70%."
            if total_events and (total_accepted / total_events) >= 0.70
            else "RED: iterative one-token MTP accept rate is below 70%."
        ),
        "base_model": args.base_model,
        "sidecar_file": args.sidecar_file,
        "use_chat_template": args.use_chat_template,
        "dtype": args.dtype,
        "max_new": args.max_new,
        "prompt_count": len(prompt_reports),
        "summary": {
            "events": total_events,
            "accepted": total_accepted,
            "accept_rate": total_accepted / total_events if total_events else 0.0,
            "mean_loss": statistics.fmean(losses) if losses else None,
            "max_loss": max(losses) if losses else None,
        },
        "prompts": prompt_reports,
        "sidecar": sidecar_inventory,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "summary": report["summary"]}, ensure_ascii=False, indent=2))
    del runner, sidecar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
