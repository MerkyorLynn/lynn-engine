#!/usr/bin/env python3
"""Generation-level W4A8 fake-quant gate for BF16 and folded artifacts.

This is the slow but honest gate after local active-MoE tensor drift improves:
does greedy generation stay on the same token path?
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from scripts.a100_w4a8_real_prompt_gate import PROMPTS  # noqa: E402


def _same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _run_model(
    model_dir: str,
    *,
    prompts: list[str],
    max_new: int,
    top_k: int,
    label: str,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    runner = LynnIncrementalRunner(model_dir, device=device, dtype=dtype, max_seq_len=4096, verbose=True)
    rows: list[dict[str, Any]] = []
    try:
        for prompt_id, prompt in enumerate(prompts):
            per_prompt: dict[str, Any] = {"prompt_id": prompt_id, "prompt": prompt}
            for mode in ("off", "full"):
                os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = mode
                out = runner.generate(prompt, max_new=max_new, top_k=top_k, use_chat_template=False)
                per_prompt[mode] = {
                    "new_ids": out["new_ids"],
                    "completion_text": out["completion_text"],
                    "completion_text_raw": out["completion_text_raw"],
                    "stopped_reason": out["stopped_reason"],
                    "decode_tps": out["timings"].get("decode_tps"),
                    "topk_trace": out.get("topk_trace", []),
                }
            prefix = _same_prefix(per_prompt["off"]["new_ids"], per_prompt["full"]["new_ids"])
            per_prompt["self_compare"] = {
                "exact": per_prompt["off"]["new_ids"] == per_prompt["full"]["new_ids"],
                "same_prefix_tokens": prefix,
                "first_diff_index": None if prefix == max_new else prefix,
            }
            rows.append(per_prompt)
    finally:
        os.environ["LYNN_W4A8_FAKE_QUANT_ACTIVE"] = "off"
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    exact = sum(1 for row in rows if row["self_compare"]["exact"])
    prefixes = [row["self_compare"]["same_prefix_tokens"] for row in rows]
    return {
        "label": label,
        "model_dir": model_dir,
        "prompt_count": len(rows),
        "exact": exact,
        "min_same_prefix_tokens": min(prefixes) if prefixes else None,
        "mean_same_prefix_tokens": sum(prefixes) / len(prefixes) if prefixes else None,
        "rows": rows,
    }


def _cross_compare(reference: dict[str, Any], candidate: dict[str, Any], max_new: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ref, cand in zip(reference["rows"], candidate["rows"]):
        prefix = _same_prefix(ref["off"]["new_ids"], cand["full"]["new_ids"])
        rows.append(
            {
                "prompt_id": ref["prompt_id"],
                "prompt": ref["prompt"],
                "exact": ref["off"]["new_ids"] == cand["full"]["new_ids"],
                "same_prefix_tokens": prefix,
                "first_diff_index": None if prefix == max_new else prefix,
                "reference_off_text": ref["off"]["completion_text"],
                "candidate_full_text": cand["full"]["completion_text"],
            }
        )
    prefixes = [row["same_prefix_tokens"] for row in rows]
    return {
        "reference": reference["label"] + ":off",
        "candidate": candidate["label"] + ":full",
        "exact": sum(1 for row in rows if row["exact"]),
        "min_same_prefix_tokens": min(prefixes) if prefixes else None,
        "mean_same_prefix_tokens": sum(prefixes) / len(prefixes) if prefixes else None,
        "rows": rows,
    }


def _decision(exact: int, total: int, min_prefix: int | None, max_new: int) -> str:
    if total and exact == total:
        return "GREEN: W4A8 generation is token-exact on this gate."
    if min_prefix is not None and min_prefix >= max_new * 0.75:
        return "AMBER: W4A8 diverges late; continue Recovery and run larger gate."
    return "RED: W4A8 changes greedy decode early; train/adapt before runtime promotion."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--folded-model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "bmm")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16")

    dtype = torch.bfloat16
    original = _run_model(
        args.model,
        prompts=args.prompts,
        max_new=args.max_new,
        top_k=args.top_k,
        label="original",
        device=args.device,
        dtype=dtype,
    )
    model_results = [original]
    cross = None
    if args.folded_model:
        folded = _run_model(
            args.folded_model,
            prompts=args.prompts,
            max_new=args.max_new,
            top_k=args.top_k,
            label="folded",
            device=args.device,
            dtype=dtype,
        )
        model_results.append(folded)
        cross = _cross_compare(original, folded, args.max_new)

    primary = cross or original
    total = len(primary["rows"]) if "rows" in primary else primary["prompt_count"]
    decision = _decision(
        int(primary["exact"]),
        total,
        primary["min_same_prefix_tokens"],
        args.max_new,
    )
    result = {
        "schema_version": "lynn-a100-w4a8-generation-gate-v1",
        "decision": decision,
        "max_new": args.max_new,
        "top_k": args.top_k,
        "env": {
            "LYNN_MOE_IMPL": os.environ.get("LYNN_MOE_IMPL"),
            "LYNN_W4A8_FAKE_QUANT_FORMAT": os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT"),
            "LYNN_W4A8_FAKE_QUANT_GRANULARITY": os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY"),
        },
        "models": model_results,
        "cross_model_compare": cross,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
