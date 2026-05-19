#!/usr/bin/env python3
"""P197 · Qwen3.5-9B W4A16 vs W4A8 per-step token drift probe.

Compares CONVSTRICT_W4A16 (reference) against either the true resident
FP4xFP8 path or the fake-quant W4A8 approximation at each decoding step using
top-5 logit ID agreement + shared-cosine.

Metric: combined = 0.5 * jaccard(top5_ids) + 0.5 * shared_cosine(shared_ids).
  - jaccard  = |A ∩ B| / |A ∪ B|
  - shared_cosine = cosine similarity over the intersection of IDs only

Decision rules:
  STRICT  — 0 drift steps across all prompts
  AMBER   — combined ≥ 0.80 AND drift_ratio ≤ 0.25
  CLOSED  — everything else
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p148_qwen35_9b_nvfp4_fast_profile import BASELINE_ENV, _restore_env, _set_env

PROMPTS = [
    "用一句话解释 CUDA graph 对推理速度的帮助。",
    "Python 写一个函数判断字符串是否为回文。",
    "Return a compact JSON object with keys city and country for Paris.",
    "What are the first 10 Fibonacci numbers? List them concisely.",
    "解释 MoE 路由器中 top-k 和 softmax 的执行顺序为什么重要。",
]


def _merge(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(updates)
    return out


CONVSTRICT_ENV = _merge(
    BASELINE_ENV,
    {
        "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    },
)

W4A8_FAKE_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_W4A8_FAKE_QUANT_ACTIVE": "full",
        "LYNN_W4A8_FAKE_QUANT_FORMAT": "FP4_E2M1xFP8_E4M3",
        "LYNN_W4A8_FAKE_QUANT_GRANULARITY": "128",
    },
)


def _true_fp4xfp8_env(sidecar_dir: str) -> dict[str, str]:
    return _merge(
        CONVSTRICT_ENV,
        {
            "LYNN_DENSE_FFN_TRUE_FP8": "1",
            "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": sidecar_dir,
        },
    )


def _topk_similarity(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, float]:
    """Compute jaccard, shared_cosine, and combined between two top-k results."""
    ref_ids = set(ref["ids"])
    cand_ids = set(cand["ids"])

    inter = ref_ids & cand_ids
    union = ref_ids | cand_ids
    jaccard = len(inter) / len(union) if union else 0.0

    if inter:
        ref_map = dict(zip(ref["ids"], ref["values"]))
        cand_map = dict(zip(cand["ids"], cand["values"]))
        dot = sum(ref_map[i] * cand_map[i] for i in inter)
        norm_r = math.sqrt(sum(ref_map[i] * ref_map[i] for i in inter))
        norm_c = math.sqrt(sum(cand_map[i] * cand_map[i] for i in inter))
        shared_cosine = dot / (norm_r * norm_c) if norm_r > 0 and norm_c > 0 else 0.0
    else:
        shared_cosine = 0.0

    combined = 0.5 * jaccard + 0.5 * shared_cosine
    return {"jaccard": jaccard, "shared_cosine": shared_cosine, "combined": combined}


def _generate_with_topk(runner: Any, prompt: str, max_new: int, top_k: int) -> dict[str, Any]:
    """Generate tokens and collect per-step top-k logits."""
    result = runner.generate(
        prompt,
        max_new=max_new,
        use_chat_template=False,
        top_k=top_k,
    )
    return result.get("topk_trace", {})


def _compare_drift(
    model: str,
    prompts: list[str],
    max_new: int,
    max_seq_len: int,
    top_k: int,
    candidate_env: dict[str, str],
) -> dict[str, Any]:
    """Run both modes and collect per-step drift data."""
    from engine.resident_runner import LynnIncrementalRunner

    per_prompt: list[dict[str, Any]] = []

    for pidx, prompt in enumerate(prompts):
        # --- reference (W4A16) ---
        old_ref = _set_env(CONVSTRICT_ENV)
        try:
            runner_ref = LynnIncrementalRunner(
                model, device="cuda", dtype=torch.bfloat16,
                max_seq_len=max_seq_len, verbose=False,
            )
            trace_ref = _generate_with_topk(runner_ref, prompt, max_new, top_k)
            del runner_ref
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            _restore_env(old_ref)

        # --- candidate (W4A8/FP4xFP8) ---
        old_cand = _set_env(candidate_env)
        try:
            runner_cand = LynnIncrementalRunner(
                model, device="cuda", dtype=torch.bfloat16,
                max_seq_len=max_seq_len, verbose=False,
            )
            trace_cand = _generate_with_topk(runner_cand, prompt, max_new, top_k)
            del runner_cand
            torch.cuda.empty_cache()
            gc.collect()
        finally:
            _restore_env(old_cand)

        # --- compare ---
        steps: list[dict[str, Any]] = []
        for si in range(len(trace_ref)):
            ref_tk = trace_ref[si]
            cand_tk = trace_cand[si] if si < len(trace_cand) else None
            if cand_tk is None:
                step_entry = {
                    "step": si,
                    "ref_ids": ref_tk["ids"],
                    "cand_ids": [],
                    "exact_match": False,
                    "jaccard": 0.0,
                    "shared_cosine": 0.0,
                    "combined": 0.0,
                }
            else:
                sim = _topk_similarity(ref_tk, cand_tk)
                step_entry = {
                    "step": si,
                    "ref_ids": ref_tk["ids"],
                    "cand_ids": cand_tk["ids"],
                    "exact_match": ref_tk["ids"] == cand_tk["ids"],
                    **sim,
                }
            steps.append(step_entry)

        per_prompt.append({
            "prompt_index": pidx,
            "prompt": prompt,
            "steps": steps,
        })

    return per_prompt


def _compute_decision(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-prompt drift into global verdict."""
    all_jaccards: list[float] = []
    all_cosines: list[float] = []
    all_combineds: list[float] = []
    drift_steps = 0
    total_steps = 0
    first_drift_step: int | None = None

    for pp in per_prompt:
        for s in pp["steps"]:
            total_steps += 1
            all_jaccards.append(s["jaccard"])
            all_cosines.append(s["shared_cosine"])
            all_combineds.append(s["combined"])
            if not s["exact_match"]:
                drift_steps += 1
                if first_drift_step is None or s["step"] < first_drift_step:
                    first_drift_step = s["step"]

    exact_match_count = total_steps - drift_steps
    drift_ratio = drift_steps / total_steps if total_steps else 0.0
    jaccard_mean = sum(all_jaccards) / len(all_jaccards) if all_jaccards else 0.0
    cosine_mean = sum(all_cosines) / len(all_cosines) if all_cosines else 0.0
    combined_mean = sum(all_combineds) / len(all_combineds) if all_combineds else 0.0

    if drift_steps == 0:
        decision = "STRICT"
    elif combined_mean >= 0.80 and drift_ratio <= 0.25:
        decision = "AMBER"
    else:
        decision = "CLOSED"

    return {
        "total_steps": total_steps,
        "drift_steps": drift_steps,
        "exact_match_count": exact_match_count,
        "first_drift_step": first_drift_step,
        "drift_ratio": drift_ratio,
        "top5_jaccard_mean": round(jaccard_mean, 4),
        "shared_cosine_mean": round(cosine_mean, 4),
        "combined_score": round(combined_mean, 4),
        "decision": decision,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P197 W4A16 vs W4A8 token drift probe.")
    ap.add_argument("--model", required=True, help="Path to NVFP4 model dir.")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=5, help="Max prompts to probe.")
    ap.add_argument(
        "--candidate-profile",
        choices=("true_fp4xfp8", "fake_w4a8"),
        default="true_fp4xfp8",
        help="Candidate path to compare against W4A16 reference.",
    )
    ap.add_argument(
        "--sidecar-dir",
        default="/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar",
        help="Packed FP4xFP8 sidecar dir for --candidate-profile true_fp4xfp8.",
    )
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    prompts = PROMPTS[: args.limit]
    candidate_env = (
        _true_fp4xfp8_env(args.sidecar_dir)
        if args.candidate_profile == "true_fp4xfp8"
        else W4A8_FAKE_ENV
    )
    candidate_label = (
        "convstrict_true_fp4xfp8_dense_ffn"
        if args.candidate_profile == "true_fp4xfp8"
        else "convstrict_fake_w4a8"
    )

    print(f"[p197] model={args.model}")
    print(f"[p197] prompts={len(prompts)} max_new={args.max_new} max_seq_len={args.max_seq_len}")
    print(f"[p197] reference=CONVSTRICT_W4A16  candidate={candidate_label}")
    print(f"[p197] metric: combined = 0.5*jaccard + 0.5*shared_cosine")
    print(f"[p197] thresholds: STRICT=0 drift; AMBER=combined≥0.80 & drift_ratio≤0.25; else CLOSED")
    print()

    t0 = time.time()
    per_prompt = _compare_drift(
        model=args.model,
        prompts=prompts,
        max_new=args.max_new,
        max_seq_len=args.max_seq_len,
        top_k=5,
        candidate_env=candidate_env,
    )
    elapsed = round(time.time() - t0, 1)
    print(f"[p197] generation + comparison done in {elapsed}s")

    result = _compute_decision(per_prompt)

    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-fp4xfp8-token-drift-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "reference_profile": "convstrict_w4a16",
        "candidate_profile": candidate_label,
        "candidate_mode": args.candidate_profile,
        "sidecar_dir": args.sidecar_dir if args.candidate_profile == "true_fp4xfp8" else None,
        "max_seq_len": args.max_seq_len,
        "n_prompts": len(prompts),
        "elapsed_s": elapsed,
        "steps": per_prompt,
        "first_drift_step": result["first_drift_step"],
        "exact_match_count": result["exact_match_count"],
        "top5_jaccard_mean": result["top5_jaccard_mean"],
        "shared_cosine_mean": result["shared_cosine_mean"],
        "combined_score": result["combined_score"],
        "decision": result["decision"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p197] report saved to {out_path}")

    summary = {
        "decision": result["decision"],
        "exact_match_count": result["exact_match_count"],
        "total_steps": result["total_steps"],
        "first_drift_step": result["first_drift_step"],
        "drift_ratio": result["drift_ratio"],
        "combined_score": result["combined_score"],
        "top5_jaccard_mean": result["top5_jaccard_mean"],
        "shared_cosine_mean": result["shared_cosine_mean"],
    }
    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
