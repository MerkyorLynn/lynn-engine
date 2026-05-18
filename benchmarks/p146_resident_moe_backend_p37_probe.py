#!/usr/bin/env python3
"""P146: generic resident Native MoE backend P37 probe.

This probe compares one candidate ``LYNN_NATIVE_ACTIVE_MOE_BACKEND`` against the
known-good Triton active-MoE backend in the real resident decode path. It is
intentionally fixture-free: fixture probes are useful for kernel iteration, but
resident P37 is the first admission gate before P25 or structured tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PROMPTS = [
    "用一句话解释 MoE active parameters",
    "Python 写一个递归阶乘函数",
    "比较 RoPE 与 ALiBi 的优缺点",
]


BASE_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
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
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "0",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton",
    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
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


def _run_mode(
    *,
    model: str,
    label: str,
    moe_backend: str,
    max_new: int,
    graph_on: bool,
    extra_env: dict[str, str],
) -> list[dict[str, Any]]:
    from engine.resident_runner import LynnIncrementalRunner

    env = dict(BASE_ENV)
    env.update(extra_env)
    env["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = moe_backend
    if moe_backend != "triton":
        env["LYNN_MOE_FAST_FIXED"] = "0"
    if graph_on:
        env["LYNN_LINEAR_BLOCK_GRAPH"] = "1"
        env["LYNN_LINEAR_BLOCK_GRAPH_REUSE"] = "1"
        env["LYNN_LINEAR_BLOCK_GRAPH_PREWARM"] = "1"

    old = _set_env(env)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(PROMPTS):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append(
                {
                    "prompt_id": idx,
                    "prompt": prompt,
                    "label": label,
                    "new_ids": out["new_ids"],
                    "completion_text": out["completion_text"],
                    "decode_tps": out["timings"].get("decode_tps"),
                }
            )
        return rows
    finally:
        _restore_env(old)


def _parse_env_items(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env item must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env item has empty key: {item!r}")
        out[key] = value
    return out


def _compare(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    exact_count = 0
    collapse_detected = False
    for b, c in zip(baseline, candidate):
        exact = b["new_ids"] == c["new_ids"]
        exact_count += int(exact)
        collapse = len(set(c["new_ids"])) == 1 if c["new_ids"] else False
        collapse_detected = collapse_detected or collapse
        drift_index = None
        for idx, (b_id, c_id) in enumerate(zip(b["new_ids"], c["new_ids"])):
            if b_id != c_id:
                drift_index = idx
                break
        rows.append(
            {
                "prompt_id": b["prompt_id"],
                "exact": exact,
                "drift_token_index": drift_index,
                "collapse": collapse,
                "baseline_ids": b["new_ids"],
                "candidate_ids": c["new_ids"],
                "baseline_tps": b["decode_tps"],
                "candidate_tps": c["decode_tps"],
            }
        )
    if collapse_detected:
        verdict = "CLOSED_GRAPH_COLLAPSE"
    elif exact_count == len(baseline):
        verdict = "P37_EXACT"
    else:
        verdict = "CLOSED_P37_DRIFT"
    return {
        "exact_count": exact_count,
        "total_prompts": len(baseline),
        "collapse_detected": collapse_detected,
        "verdict": verdict,
        "results": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate-backend", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--graph-on", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="Extra candidate and baseline env item, KEY=VALUE")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    extra_env = _parse_env_items(args.env)
    print("[p146] resident Native MoE backend P37 probe")
    print(f"[p146] model={args.model}")
    print(f"[p146] candidate_backend={args.candidate_backend}")
    print(f"[p146] graph_on={args.graph_on}")
    print(f"[p146] max_new={args.max_new}")
    if extra_env:
        print(f"[p146] extra_env={extra_env}")
    print()

    print("[p146] Running BASELINE (triton)...")
    baseline = _run_mode(
        model=args.model,
        label="triton_baseline",
        moe_backend="triton",
        max_new=args.max_new,
        graph_on=args.graph_on,
        extra_env=extra_env,
    )
    for row in baseline:
        tps = row["decode_tps"]
        print(f"  P{row['prompt_id']}: {row['new_ids'][:8]} tps={tps:.2f}" if tps else f"  P{row['prompt_id']}: {row['new_ids'][:8]}")

    print(f"\n[p146] Running CANDIDATE ({args.candidate_backend})...")
    candidate = _run_mode(
        model=args.model,
        label=args.candidate_backend,
        moe_backend=args.candidate_backend,
        max_new=args.max_new,
        graph_on=args.graph_on,
        extra_env=extra_env,
    )
    for row in candidate:
        tps = row["decode_tps"]
        print(f"  P{row['prompt_id']}: {row['new_ids'][:8]} tps={tps:.2f}" if tps else f"  P{row['prompt_id']}: {row['new_ids'][:8]}")

    comparison = _compare(baseline, candidate)
    print()
    print("=" * 60)
    print("P146 RESULT")
    print(f"  Exact: {comparison['exact_count']}/{comparison['total_prompts']}")
    print(f"  Collapse: {comparison['collapse_detected']}")
    print(f"  Verdict: {comparison['verdict']}")
    print("=" * 60)

    report = {
        "probe": "p146_resident_moe_backend_p37",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "candidate_backend": args.candidate_backend,
        "graph_on": args.graph_on,
        "max_new": args.max_new,
        "extra_env": extra_env,
        **comparison,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p146] report={out}")
    return 0 if comparison["verdict"] == "P37_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
