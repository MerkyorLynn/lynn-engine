#!/usr/bin/env python3
"""P105: generation-level W4A8 fake-quant gate.

P104 measured local active-MoE drift and found the near-term target:
E4M3 per16 is AMBER. P105 checks whether that local drift actually changes
greedy decode behavior over normal prompts.

The runner stays on the current packed NVFP4 decode path. We only enable the
research-only `LYNN_W4A8_FAKE_QUANT_ACTIVE` switch for decode active experts:

* off:    current BF16 activation semantics
* gateup: FP8-round hidden before active expert gate/up
* full:   gateup + FP8-round intermediate before down

Prefill remains BF16 in all modes. This keeps P105 fast and targeted at the
decode runtime route we are optimizing for 155+ TPS.
"""
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


PROMPTS = [
    "用一句话解释 W4A8 为什么可能比 W4A4 更稳。",
    "请写一个 Python 函数,判断字符串是否是回文。",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请给出一个 JSON: {\"city\":\"Tokyo\",\"unit\":\"celsius\"}",
    "总结一下 MoE 模型里 router 和 expert 的分工。",
    "解释长上下文推理里 linear attention 的优势。",
]


@contextmanager
def _temporary_env(values: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _same_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _run_mode(
    runner: LynnIncrementalRunner,
    prompt: str,
    *,
    mode: str,
    max_new: int,
    top_k: int,
) -> dict:
    with _temporary_env({"LYNN_W4A8_FAKE_QUANT_ACTIVE": mode}):
        out = runner.generate(prompt, max_new=max_new, top_k=top_k, use_chat_template=True)
    return {
        "mode": mode,
        "new_ids": out["new_ids"],
        "completion_text": out["completion_text"],
        "completion_text_raw": out["completion_text_raw"],
        "stopped_reason": out["stopped_reason"],
        "decode_tps": out["timings"]["decode_tps"],
        "topk_trace": out.get("topk_trace", []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--prompts", nargs="*", default=PROMPTS)
    ap.add_argument("--modes", nargs="+", default=["gateup", "full"], choices=["gateup", "full"])
    args = ap.parse_args()

    os.environ.setdefault("LYNN_MOE_IMPL", "packed_nvfp4")
    os.environ.setdefault("LYNN_MOE_FAST_FIXED", "1")
    os.environ.setdefault("LYNN_NATIVE_GATEUP_BACKEND", "triton_fast_decode")
    os.environ.setdefault("LYNN_NATIVE_DOWN_BACKEND", "triton")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_FORMAT", "e4m3")
    os.environ.setdefault("LYNN_W4A8_FAKE_QUANT_GRANULARITY", "per16")
    os.environ.setdefault("LYNN_LINEAR_ATTN_RECURRENT_INPLACE", "1")
    os.environ.setdefault("LYNN_LINEAR_STATE_UPDATE", "inplace")

    runner = LynnIncrementalRunner(args.model, device="cuda", verbose=False)
    cases = []
    for idx, prompt in enumerate(args.prompts):
        baseline = _run_mode(runner, prompt, mode="off", max_new=args.max_new, top_k=args.top_k)
        variants = []
        for mode in args.modes:
            got = _run_mode(runner, prompt, mode=mode, max_new=args.max_new, top_k=args.top_k)
            prefix = _same_prefix(baseline["new_ids"], got["new_ids"])
            variants.append(
                {
                    **got,
                    "exact_new_ids_match": got["new_ids"] == baseline["new_ids"],
                    "same_prefix_tokens": prefix,
                    "first_diff_index": None if prefix == min(len(baseline["new_ids"]), len(got["new_ids"])) else prefix,
                }
            )
        cases.append(
            {
                "id": idx,
                "prompt": prompt,
                "baseline": baseline,
                "variants": variants,
            }
        )

    summary_by_mode = {}
    for mode in args.modes:
        mode_cases = [v for c in cases for v in c["variants"] if v["mode"] == mode]
        summary_by_mode[mode] = {
            "exact_match_count": sum(1 for v in mode_cases if v["exact_new_ids_match"]),
            "total": len(mode_cases),
            "min_same_prefix_tokens": min(v["same_prefix_tokens"] for v in mode_cases),
            "mean_same_prefix_tokens": sum(v["same_prefix_tokens"] for v in mode_cases) / len(mode_cases),
            "mean_decode_tps": sum((v["decode_tps"] or 0.0) for v in mode_cases) / len(mode_cases),
        }

    gateup = summary_by_mode.get("gateup")
    full = summary_by_mode.get("full")

    def _all_exact(summary: dict | None) -> bool:
        return bool(summary and summary["exact_match_count"] == summary["total"])

    def _near_boundary(summary: dict | None) -> bool:
        if not summary:
            return False
        # Generation-level greedy exactness is intentionally strict. For W4A8
        # training triage, a late divergence with long same-prefix stability is
        # an adaptation target, not a route-killing failure.
        return (
            summary["min_same_prefix_tokens"] >= min(12, args.max_new)
            and summary["mean_same_prefix_tokens"] >= max(1.0, args.max_new * 0.5)
        )

    if gateup and full and _all_exact(gateup) and _all_exact(full):
        decision = "GREEN: gateup and full W4A8 fake-quant match all greedy decode samples; proceed to longer eval."
    elif (gateup and _all_exact(gateup) and full and _near_boundary(full)) or (
        full and _all_exact(full) and gateup and _near_boundary(gateup)
    ):
        decision = "AMBER: one W4A8 mode is exact and the other is near-boundary; train/adapt before runtime promotion."
    elif (gateup and _near_boundary(gateup)) or (full and _near_boundary(full)):
        decision = "AMBER: W4A8 changes some greedy tokens but keeps long same-prefix stability; use A100 QAT-lite/Recovery."
    else:
        decision = "RED: W4A8 changes greedy decode early; train/adapt before runtime promotion."
    result = {
        "schema_version": "lynn-engine-p105-w4a8-generate-gate-v1",
        "model": args.model,
        "max_new": args.max_new,
        "top_k": args.top_k,
        "fake_quant_format": os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT"),
        "fake_quant_granularity": os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY"),
        "cases": cases,
        "summary_by_mode": summary_by_mode,
        "decision": decision,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"].startswith("GREEN") else (1 if result["decision"].startswith("AMBER") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
