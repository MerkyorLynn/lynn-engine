#!/usr/bin/env python3
"""P36: gate runner-fixed decode dispatch.

This compares the legacy env/import dispatch path with the runner-fixed
dispatch plan introduced after P35. It does not change kernels or numerical
contracts; a promotion requires identical greedy token IDs and non-regressed
decode TPS.
"""
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

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


PROMPTS = [
    "用一句话解释 MoE active parameters",
    "Python 写一个递归阶乘函数",
    "比较 RoPE 与 ALiBi 的优缺点",
]


PRODUCTION_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_MOE_GATE_BLOCK_INTER": "8",
    "LYNN_MOE_GATE_BLOCK_HIDDEN": "256",
    "LYNN_MOE_DOWN_BLOCK_HIDDEN": "8",
    "LYNN_MOE_DOWN_BLOCK_INTER": "512",
    "LYNN_MOE_GATE_NUM_WARPS": "4",
    "LYNN_MOE_DOWN_NUM_WARPS": "8",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_LINEAR_BLOCK_GRAPH": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
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


def _run_mode(model: str, *, fast_dispatch: bool, max_new: int, prompts: list[str]) -> list[dict[str, Any]]:
    env = dict(PRODUCTION_ENV)
    env["LYNN_DECODE_FAST_DISPATCH"] = "1" if fast_dispatch else "0"
    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "fast_dispatch": fast_dispatch,
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "timings": out["timings"],
                "stopped_reason": out["stopped_reason"],
            })
        return rows
    finally:
        _restore_env(old)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["timings"]["decode_tps"]) for r in rows if r["timings"].get("decode_tps")]
    return {
        "count": len(rows),
        "decode_tps_mean": mean(vals) if vals else None,
        "decode_tps_median": median(vals) if vals else None,
        "decode_tps_min": min(vals) if vals else None,
        "decode_tps_max": max(vals) if vals else None,
    }


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

    legacy_rows = _run_mode(args.model, fast_dispatch=False, max_new=args.max_new, prompts=prompts)
    fast_rows = _run_mode(args.model, fast_dispatch=True, max_new=args.max_new, prompts=prompts)
    for ref, cand in zip(legacy_rows, fast_rows, strict=True):
        cand["new_ids_match_legacy"] = cand["new_ids"] == ref["new_ids"]
        cand["legacy_new_ids"] = ref["new_ids"]

    legacy_summary = _summary(legacy_rows)
    fast_summary = _summary(fast_rows)
    speedup = None
    if legacy_summary["decode_tps_median"] and fast_summary["decode_tps_median"]:
        speedup = fast_summary["decode_tps_median"] / legacy_summary["decode_tps_median"]

    result = {
        "schema_version": "lynn-engine-p36-decode-fast-dispatch-gate-v1",
        "model": args.model,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "legacy": {"rows": legacy_rows, "summary": legacy_summary},
        "fast": {"rows": fast_rows, "summary": fast_summary},
        "new_ids_all_match": all(row["new_ids_match_legacy"] for row in fast_rows),
        "median_speedup": speedup,
        "promote_default": bool(
            all(row["new_ids_match_legacy"] for row in fast_rows)
            and speedup is not None
            and speedup >= 0.995
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["new_ids_all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
