#!/usr/bin/env python3
"""Diagnose saved MTP sidecar misses with label ranks and logit margins.

The saved-sidecar gate tells us whether a draft token exactly matches the base
greedy token. Once the ladder plateaus, the next useful signal is what the MTP
head is preferring instead, how far the teacher token is from the top, and
whether misses cluster into reusable failure modes such as premature stop,
generic JSON keys, or punctuation splits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_mtp_fc_calibration_train import _mtp_cfg  # noqa: E402
from scripts.a100_mtp_forward_smoke import _load_sidecar, _mtp_layer_weights, _topk  # noqa: E402
from scripts.a100_mtp_iterative_train import _collect_cases, _load_prompts, _mtp_logits_for_case  # noqa: E402


def _parse_sidecar(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, value
    label, path = value.split("=", 1)
    return label, path


def _bucket_text(text: str) -> str:
    if text in {"<|im_end|>", "<|endoftext|>"}:
        return "stop_token"
    if text in {"action", "city", "tool", "type", "body", "unit", "metric", "router"}:
        return "generic_structured_key"
    if text.strip() in {"", "\\n", "\\n\\n"} or text in {"\n", "\n\n"}:
        return "whitespace"
    if text in {'"', '":', '":"', '",', '","', '"}', "{", '{"', " {"}:
        return "json_punctuation"
    if text.startswith("<") and text.endswith(">"):
        return "special_token"
    return "semantic_token"


def _margin_row(
    *,
    runner: LynnIncrementalRunner,
    logits: torch.Tensor,
    case: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    logits_f = logits.float()[0]
    label_id = int(case["label_id"])
    draft_id = int(torch.argmax(logits_f).item())
    label_score = float(logits_f[label_id].item())
    draft_score = float(logits_f[draft_id].item())
    label_rank = int(torch.count_nonzero(logits_f > logits_f[label_id]).item()) + 1
    loss = float(F.cross_entropy(logits.float(), torch.tensor([label_id], device=logits.device)).item())
    draft_text = runner.tokenizer.decode([draft_id])
    accepted = draft_id == label_id
    return {
        "case_idx": int(case["case_idx"]) if "case_idx" in case else None,
        "prompt_idx": int(case["prompt_idx"]),
        "id": str(case["id"]),
        "step": int(case["step"]),
        "current_pos": int(case["current_pos"]),
        "current_token_id": int(case["current_token_id"]),
        "current_token_text": str(case["current_token_text"]),
        "label_id": label_id,
        "label_text": str(case["label_text"]),
        "draft_id": draft_id,
        "draft_text": draft_text,
        "accepted": accepted,
        "loss": loss,
        "label_rank": label_rank,
        "label_score": label_score,
        "draft_score": draft_score,
        "draft_minus_label": draft_score - label_score,
        "miss_bucket": "accepted" if accepted else _bucket_text(draft_text),
        "mtp_topk": _topk(runner.tokenizer, logits, top_k),
    }


def _compact_by_step(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_step: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = by_step.setdefault(
            str(row["step"]),
            {
                "events": 0,
                "accepted": 0,
                "misses": 0,
                "mean_label_rank": None,
                "mean_draft_minus_label": None,
                "miss_buckets": {},
            },
        )
        item["events"] += 1
        item["accepted"] += int(row["accepted"])
        item["misses"] += int(not row["accepted"])
    for key, item in by_step.items():
        step_rows = [row for row in rows if str(row["step"]) == key]
        miss_rows = [row for row in step_rows if not row["accepted"]]
        item["accept_rate"] = item["accepted"] / item["events"] if item["events"] else 0.0
        item["mean_label_rank"] = statistics.fmean(float(row["label_rank"]) for row in step_rows)
        item["mean_draft_minus_label"] = statistics.fmean(float(row["draft_minus_label"]) for row in step_rows)
        item["miss_buckets"] = dict(Counter(str(row["miss_bucket"]) for row in miss_rows))
    return by_step


def _summarize(label: str, sidecar_file: str, inventory: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = sum(1 for row in rows if row["accepted"])
    misses = [row for row in rows if not row["accepted"]]
    near_misses = [row for row in misses if int(row["label_rank"]) <= 5]
    hard_misses = [row for row in misses if int(row["label_rank"]) > 20]
    bucket_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses:
        bucket_to_rows[str(row["miss_bucket"])].append(row)
    bucket_summary = {
        bucket: {
            "count": len(items),
            "mean_label_rank": statistics.fmean(float(row["label_rank"]) for row in items),
            "mean_draft_minus_label": statistics.fmean(float(row["draft_minus_label"]) for row in items),
        }
        for bucket, items in sorted(bucket_to_rows.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }
    return {
        "label": label,
        "sidecar_file": sidecar_file,
        "inventory": inventory,
        "case_count": len(rows),
        "accepted": accepted,
        "accept_rate": accepted / len(rows) if rows else 0.0,
        "miss_count": len(misses),
        "near_miss_count_label_rank_le_5": len(near_misses),
        "hard_miss_count_label_rank_gt_20": len(hard_misses),
        "by_step": _compact_by_step(rows),
        "miss_buckets": bucket_summary,
        "worst_misses": sorted(
            misses,
            key=lambda row: (float(row["draft_minus_label"]), int(row["label_rank"])),
            reverse=True,
        )[:24],
        "near_misses": sorted(near_misses, key=lambda row: (int(row["label_rank"]), float(row["draft_minus_label"])))[:24],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--eval-prompts-file", required=True)
    ap.add_argument("--sidecar", action="append", required=True, help="label=/path/to/mtp.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-new-eval", type=int, default=16)
    ap.add_argument("--first-token-weight", type=float, default=1.0)
    ap.add_argument("--step1-weight", type=float, default=1.0)
    ap.add_argument("--later-token-weight", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompts(args.eval_prompts_file, [])
    runner = LynnIncrementalRunner(args.base_model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)
    eval_cases = _collect_cases(
        runner=runner,
        specs=specs,
        use_chat_template=args.use_chat_template,
        max_new=args.max_new_eval,
        first_token_weight=args.first_token_weight,
        step1_weight=args.step1_weight,
        later_token_weight=args.later_token_weight,
    )
    for idx, case in enumerate(eval_cases):
        case["case_idx"] = idx

    rows: list[dict[str, Any]] = []
    for item in args.sidecar:
        label, sidecar_path = _parse_sidecar(item)
        sidecar, inventory = _load_sidecar(Path(sidecar_path), args.device, dtype)
        mtp_w = _mtp_layer_weights(sidecar)
        cfg = _mtp_cfg(runner, mtp_w)
        sidecar_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for case in eval_cases:
                logits = _mtp_logits_for_case(runner, sidecar, mtp_w, cfg, case)
                sidecar_rows.append(_margin_row(runner=runner, logits=logits, case=case, top_k=args.top_k))
                del logits
        rows.append(_summarize(label, sidecar_path, inventory, sidecar_rows))
        del sidecar, mtp_w
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(rows, key=lambda row: float(row["accept_rate"]), default=None)
    result = {
        "schema_version": "lynn-a100-mtp-saved-sidecar-diagnostic-v1",
        "base_model": args.base_model,
        "eval_prompts_file": args.eval_prompts_file,
        "max_new_eval": args.max_new_eval,
        "case_count": len(eval_cases),
        "rows": rows,
        "best_label": None if best is None else best["label"],
        "best_accept_rate": None if best is None else best["accept_rate"],
        "decision": (
            "GREEN: at least one saved sidecar clears 55% heldout accept."
            if best and float(best["accept_rate"]) >= 0.55
            else "AMBER: sidecar remains below serving threshold; use miss buckets for next target construction."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "best_label": result["best_label"],
                "best_accept_rate": result["best_accept_rate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
