#!/usr/bin/env python3
"""P107: measure MTP one-token shadow credit through the resident runner.

This is the serving-shaped bridge between A100 MTP training reports and R6000
decode integration. It loads a normal `LynnIncrementalRunner`, enables the
opt-in MTP shadow verifier, and records whether the sidecar draft argmax matches
the base greedy argmax at each generated token boundary.

The probe does not change emitted tokens and does not claim TPS acceleration.
It reports the acceptance credit and draft-head cost that a future speculative
decode path would have to turn into real throughput.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_PROMPTS = [
    "Return one JSON object with keys city and unit for Berlin in metric units. No markdown.",
    "Output exactly one JSON arguments object for translate_text with text hello and target_language Japanese. No markdown.",
    "Write a Python function is_palindrome(s) in exactly six lines.",
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


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = sum(int(row["mtp_shadow"]["events"]) for row in rows)
    accepted = sum(int(row["mtp_shadow"]["accepted"]) for row in rows)
    draft_seconds = [
        float(x)
        for row in rows
        for x in row["mtp_shadow"].get("draft_step_seconds", [])
    ]
    decode_step_seconds = [
        float(x)
        for row in rows
        for x in row["timings"].get("decode_step_seconds", [])
    ]
    prompt_rates = [
        float(row["mtp_shadow"]["accept_rate"])
        for row in rows
        if row["mtp_shadow"].get("accept_rate") is not None
    ]
    return {
        "prompt_count": len(rows),
        "events": events,
        "accepted": accepted,
        "accept_rate": accepted / events if events else None,
        "mean_prompt_accept_rate": statistics.fmean(prompt_rates) if prompt_rates else None,
        "decode_tps": (
            len(decode_step_seconds) / sum(decode_step_seconds)
            if decode_step_seconds
            else None
        ),
        "draft_tps": (
            len(draft_seconds) / sum(draft_seconds)
            if draft_seconds
            else None
        ),
        "mean_draft_seconds": statistics.fmean(draft_seconds) if draft_seconds else None,
        "max_one_token_speculative_multiplier": (
            (events + accepted) / events
            if events
            else None
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts-file")
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    os.environ["LYNN_MTP_SIDECAR"] = args.sidecar_file
    os.environ["LYNN_MTP_SHADOW_VERIFY"] = "1"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    specs = _load_prompt_specs(args.prompts_file, args.prompts)
    runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, max_seq_len=4096, verbose=True)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        out = runner.generate(
            spec["prompt"],
            max_new=args.max_new,
            top_k=args.top_k,
            use_chat_template=args.use_chat_template,
        )
        rows.append(
            {
                "id": spec["id"],
                "prompt": spec["prompt"],
                "completion_text": out["completion_text"],
                "new_ids": out["new_ids"],
                "timings": out["timings"],
                "mtp_shadow": out["mtp_shadow"],
                "stopped_reason": out["stopped_reason"],
            }
        )

    summary = _summarize(rows)
    report = {
        "schema_version": "lynn-p107-mtp-shadow-serving-credit-v1",
        "decision": (
            "GREEN-CREDIT: MTP shadow accept is at least 55%."
            if summary["accept_rate"] is not None and summary["accept_rate"] >= 0.55
            else "AMBER: MTP shadow accept is below serving-credit threshold."
        ),
        "model": args.model,
        "sidecar_file": args.sidecar_file,
        "use_chat_template": args.use_chat_template,
        "dtype": args.dtype,
        "max_new": args.max_new,
        "summary": summary,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
