#!/usr/bin/env python3
"""P143: V3.1 graph-safe resident MoE P37 exact probe.

Runs LynnIncrementalRunner with LYNN_NATIVE_ACTIVE_MOE_BACKEND=packed_pretransposed_graphsafe_v32_ordered
and compares output token IDs against the Triton baseline (LYNN_NATIVE_ACTIVE_MOE_BACKEND=triton).

P37 exact = all 3 prompts produce identical greedy token IDs.
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


def _run_mode(model: str, label: str, moe_backend: str, max_new: int, graph_on: bool) -> list[dict]:
    from engine.resident_runner import LynnIncrementalRunner

    env = dict(BASE_ENV)
    env["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = moe_backend
    if moe_backend != "triton":
        # V3.1 path uses the non-fast-fixed dispatch; fast-fixed rejects non-triton
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
            rows.append({
                "prompt_id": idx,
                "prompt": prompt,
                "label": label,
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "decode_tps": out["timings"].get("decode_tps"),
            })
        return rows
    finally:
        _restore_env(old)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--graph-on", action="store_true", default=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[p145] V3.1 resident P37 probe")
    print(f"[p145] Model: {args.model}")
    print(f"[p145] Graph-on: {args.graph_on}")
    print(f"[p145] Max new tokens: {args.max_new}")
    print()

    # Baseline: Triton (known-good)
    print("[p145] Running BASELINE (triton)...")
    baseline = _run_mode(args.model, "triton_baseline", "triton", args.max_new, graph_on=args.graph_on)
    for r in baseline:
        print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}... tps={r['decode_tps']:.1f}" if r['decode_tps'] else f"  P{r['prompt_id']}: {r['new_ids'][:8]}...")

    # Candidate: V3.1
    print("\n[p145] Running CANDIDATE (packed_pretransposed_graphsafe_v32_ordered)...")
    candidate = _run_mode(args.model, "graphsafe_v32_ordered", "packed_pretransposed_graphsafe_v32_ordered", args.max_new, graph_on=args.graph_on)
    for r in candidate:
        print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}... tps={r['decode_tps']:.1f}" if r['decode_tps'] else f"  P{r['prompt_id']}: {r['new_ids'][:8]}...")

    # Compare
    exact_count = 0
    collapse_detected = False
    results = []
    for b, c in zip(baseline, candidate):
        match = b["new_ids"] == c["new_ids"]
        if match:
            exact_count += 1
        # Token-0 collapse: all tokens identical
        if len(set(c["new_ids"])) == 1:
            collapse_detected = True
        results.append({
            "prompt_id": b["prompt_id"],
            "exact": match,
            "baseline_ids": b["new_ids"],
            "candidate_ids": c["new_ids"],
            "baseline_tps": b["decode_tps"],
            "candidate_tps": c["decode_tps"],
            "collapse": len(set(c["new_ids"])) == 1,
        })

    # Verdict
    if collapse_detected:
        verdict = "CLOSED_GRAPH_COLLAPSE"
    elif exact_count == len(PROMPTS):
        verdict = "P37_EXACT"
    else:
        verdict = "CLOSED_P37_DRIFT"

    print(f"\n{'='*60}")
    print(f"P143 P37 RESULT")
    print(f"  Exact: {exact_count}/{len(PROMPTS)}")
    print(f"  Collapse: {collapse_detected}")
    print(f"  Graph-on: {args.graph_on}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")

    report = {
        "probe": "p145_resident_moe_graphsafe_p37",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "graph_on": args.graph_on,
        "max_new": args.max_new,
        "exact_count": exact_count,
        "total_prompts": len(PROMPTS),
        "collapse_detected": collapse_detected,
        "verdict": verdict,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p145] Report: {args.out}")

    return 0 if verdict == "P37_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
