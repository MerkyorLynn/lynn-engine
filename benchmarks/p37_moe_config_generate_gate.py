#!/usr/bin/env python3
"""P37: production MoE config generate gate.

P32 compares Triton against diagnostic `cuda_scalar`, which is useful for native
kernel research but awkward for safe production retunes because the diagnostic
backend is intentionally guarded. This gate compares two *production* packed
NVFP4 Triton configurations and requires exact greedy IDs before considering a
config change.
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


BASE_ENV = {
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


def _parse_overrides(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def _run_mode(
    model: str,
    *,
    label: str,
    overrides: dict[str, str],
    max_new: int,
    prompts: list[str],
) -> list[dict[str, Any]]:
    env = dict(BASE_ENV)
    env.update(overrides)
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
                "overrides": overrides,
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
    ap.add_argument("--baseline", action="append", default=[], help="Baseline env override KEY=VALUE")
    ap.add_argument("--candidate", action="append", default=[], help="Candidate env override KEY=VALUE")
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

    baseline_overrides = _parse_overrides(args.baseline)
    candidate_overrides = _parse_overrides(args.candidate)
    baseline_rows = _run_mode(
        args.model,
        label="baseline",
        overrides=baseline_overrides,
        max_new=args.max_new,
        prompts=prompts,
    )
    candidate_rows = _run_mode(
        args.model,
        label="candidate",
        overrides=candidate_overrides,
        max_new=args.max_new,
        prompts=prompts,
    )
    for ref, cand in zip(baseline_rows, candidate_rows, strict=True):
        cand["new_ids_match_baseline"] = cand["new_ids"] == ref["new_ids"]
        cand["baseline_new_ids"] = ref["new_ids"]

    baseline_summary = _summary(baseline_rows)
    candidate_summary = _summary(candidate_rows)
    speedup = None
    if baseline_summary["decode_tps_median"] and candidate_summary["decode_tps_median"]:
        speedup = candidate_summary["decode_tps_median"] / baseline_summary["decode_tps_median"]

    all_match = all(row["new_ids_match_baseline"] for row in candidate_rows)
    result = {
        "schema_version": "lynn-engine-p37-moe-config-generate-gate-v1",
        "model": args.model,
        "baseline_overrides": baseline_overrides,
        "candidate_overrides": candidate_overrides,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "baseline": {"rows": baseline_rows, "summary": baseline_summary},
        "candidate": {"rows": candidate_rows, "summary": candidate_summary},
        "new_ids_all_match": all_match,
        "median_speedup": speedup,
        "promote_default": bool(all_match and speedup is not None and speedup >= 1.005),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
