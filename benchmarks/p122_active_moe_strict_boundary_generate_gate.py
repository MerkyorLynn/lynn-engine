#!/usr/bin/env python3
"""P122: exact-greedy generate gate for strict active-MoE boundary backend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p37_moe_config_generate_gate import BASE_ENV, PROMPTS  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


STRICT_ENV = {
    **BASE_ENV,
    "LYNN_MOE_FAST_FIXED": "0",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "strict_fused_boundary",
}

TRITON_ENV = {
    **BASE_ENV,
    "LYNN_MOE_FAST_FIXED": "0",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
}


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["timings"]["decode_tps"]) for r in rows if r["timings"].get("decode_tps")]
    return {
        "count": len(rows),
        "decode_tps_mean": mean(vals) if vals else None,
        "decode_tps_median": median(vals) if vals else None,
        "decode_tps_min": min(vals) if vals else None,
        "decode_tps_max": max(vals) if vals else None,
    }


def _run_mode(
    model: str,
    *,
    label: str,
    env: dict[str, str],
    max_new: int,
    prompts: list[str],
) -> list[dict[str, Any]]:
    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "label": label,
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "timings": out["timings"],
                "stopped_reason": out["stopped_reason"],
            })
        return rows
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prompts-jsonl")
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts_jsonl:
        prompts = []
        with open(args.prompts_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("input") or item.get("question") or item.get("problem")
                if prompt:
                    prompts.append(str(prompt))
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_jsonl}")

    triton_rows = _run_mode(
        args.model,
        label="triton",
        env=TRITON_ENV,
        max_new=args.max_new,
        prompts=prompts,
    )
    strict_rows = _run_mode(
        args.model,
        label="strict_fused_boundary",
        env=STRICT_ENV,
        max_new=args.max_new,
        prompts=prompts,
    )
    for ref, cand in zip(triton_rows, strict_rows, strict=True):
        cand["new_ids_match_reference"] = cand["new_ids"] == ref["new_ids"]
        cand["reference_new_ids"] = ref["new_ids"]

    all_match = all(row["new_ids_match_reference"] for row in strict_rows)
    triton_summary = _summary(triton_rows)
    strict_summary = _summary(strict_rows)
    speedup = None
    if triton_summary["decode_tps_median"] and strict_summary["decode_tps_median"]:
        speedup = strict_summary["decode_tps_median"] / triton_summary["decode_tps_median"]

    result = {
        "schema_version": "lynn-engine-p122-active-moe-strict-boundary-generate-gate-v1",
        "model": args.model,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "triton": {"rows": triton_rows, "summary": triton_summary},
        "strict_fused_boundary": {"rows": strict_rows, "summary": strict_summary},
        "new_ids_all_match": all_match,
        "median_speedup": speedup,
        "promote_default": False,
        "decision": (
            "Strict fused boundary must first match Triton exact-greedy output. "
            "Use P25 and structured gates separately before any runtime promotion."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
