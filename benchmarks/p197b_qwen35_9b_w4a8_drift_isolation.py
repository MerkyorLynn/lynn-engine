#!/usr/bin/env python3
"""P197b · Qwen3.5-9B W4A8 drift source isolation probe.

Variant of P197 that:
1. Tests TRUE FP4xFP8 with the SCALAR REFERENCE kernel (known correct, P191 GREEN)
   to separate MMA fragment layout drift from quantization drift.
2. Tests individual layer boundaries:
   - gate_proj+up_proj FP8 only (down_proj stays BF16 matmul)
   - down_proj FP8 only (gate/up stay BF16 matmul)
   - full FP8 (all three projections)
3. Tests with/without per-16 act scale vs tensor-level act scale.

This isolates whether drift comes from:
  A) MMA fragment layout bug (→ fix fragment, not quantization)
  B) FP8 activation quantization error accumulating through residual
  C) Per-16 vs per-tensor scale mismatch
  D) silu/multiply intermediate re-quantization to FP8 for down_proj

Usage:
  python benchmarks/p197b_qwen35_9b_w4a8_drift_isolation.py \\
    --model /root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0 \\
    --out /root/autodl-tmp/reports/qwen35_9b/p197b_w4a8_drift_isolation.json
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p148_qwen35_9b_nvfp4_fast_profile import (
    CONVSTRICT_ENV,
    _restore_env,
    _set_env,
)
from benchmarks.p184_qwen35_9b_nvfp4_convstrict_exact_gate import _merge

# ─────────────────────────────────────────────────────────────
# Env configs for isolation modes
# ─────────────────────────────────────────────────────────────
# Mode A: MMA path (as deployed, known fragment-layout broken per P191)
TRUE_FP8_MMA_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": (
            "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar"
        ),
        "LYNN_DENSE_FFN_TRUE_FP8_KERNEL": "mma",  # default
    },
)

# Mode B: Scalar reference path (P191 GREEN: correct dequant, slow)
TRUE_FP8_SCALAR_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": (
            "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar"
        ),
        "LYNN_DENSE_FFN_TRUE_FP8_KERNEL": "scalar",
    },
)

# Mode C: gate/up only FP8, down stays BF16 matmul
TRUE_FP8_GATEUP_ONLY_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": (
            "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar"
        ),
        "LYNN_DENSE_FFN_TRUE_FP8_KERNEL": "scalar",
        "LYNN_DENSE_FFN_TRUE_FP8_SCOPE": "gateup",
    },
)

# Mode D: down only FP8, gate/up stay BF16 matmul
TRUE_FP8_DOWN_ONLY_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": (
            "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar"
        ),
        "LYNN_DENSE_FFN_TRUE_FP8_KERNEL": "scalar",
        "LYNN_DENSE_FFN_TRUE_FP8_SCOPE": "down",
    },
)


# ─────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    "用一句话解释 CUDA graph 对推理速度的帮助。",
    "Write a Python function to compute the Fibonacci sequence up to n.",
    "Return a compact JSON object with keys city and country for Paris.",
    "Explain the difference between TCP and UDP in exactly 3 sentences.",
    "List the first 5 prime numbers as a JSON array.",
]


# ─────────────────────────────────────────────────────────────
# Helpers (same as P197)
# ─────────────────────────────────────────────────────────────
def _topk_similarity(
    a_ids: list[int], a_vals: list[float],
    b_ids: list[int], b_vals: list[float],
) -> tuple[float, float, float]:
    a_set = set(a_ids)
    b_set = set(b_ids)
    inter = a_set & b_set
    union = a_set | b_set
    jaccard = len(inter) / len(union) if union else 1.0
    if inter:
        a_map = dict(zip(a_ids, a_vals))
        b_map = dict(zip(b_ids, b_vals))
        dot = sum(a_map[i] * b_map[i] for i in inter)
        na = math.sqrt(sum(a_map[i] ** 2 for i in inter))
        nb = math.sqrt(sum(b_map[i] ** 2 for i in inter))
        shared_cosine = dot / (na * nb) if na > 1e-12 and nb > 1e-12 else 0.0
    else:
        shared_cosine = 0.0
    combined = jaccard * 0.5 + shared_cosine * 0.5
    return jaccard, shared_cosine, combined


def _run_one(
    model: str,
    label: str,
    env: dict[str, str],
    prompts: list[str],
    max_new: int,
    max_seq_len: int,
) -> dict[str, Any]:
    from engine.resident_runner import LynnIncrementalRunner

    print(f"[p197b] loading mode={label}", flush=True)
    old = _set_env(env)
    try:
        t0 = time.time()
        runner = LynnIncrementalRunner(
            model,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=max_seq_len,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_seconds = time.time() - t0

        rows: list[dict[str, Any]] = []
        for pid, prompt in enumerate(prompts):
            print(f"[p197b] {label} prompt={pid}/{len(prompts)}", flush=True)
            out = runner.generate(
                prompt,
                max_new=max_new,
                use_chat_template=False,
                top_k=5,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings = out.get("timings", {})
            topk_trace = out.get("topk_trace", [])
            new_ids = out.get("new_ids", [])
            tok = runner.tokenizer
            rows.append({
                "prompt_id": pid,
                "prompt": prompt,
                "new_ids": new_ids,
                "decoded_tokens": [tok.decode([tid]) for tid in new_ids],
                "topk_trace": topk_trace,
                "completion_text": out.get("completion_text", ""),
                "decode_tps": timings.get("decode_tps"),
            })

        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"label": label, "env": env, "load_seconds": load_seconds, "rows": rows}
    finally:
        _restore_env(old)


def _compare(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    cand_by_pid = {r["prompt_id"]: r for r in cand["rows"]}
    total_steps = 0
    drift_steps = 0
    first_drift_prompt = None
    first_drift_step = None
    topology_combineds: list[float] = []

    for ref_row in ref["rows"]:
        pid = ref_row["prompt_id"]
        cand_row = cand_by_pid.get(pid)
        if cand_row is None:
            continue

        ref_ids = ref_row["new_ids"]
        cand_ids = cand_row["new_ids"]
        ref_topk = ref_row.get("topk_trace", [])
        cand_topk = cand_row.get("topk_trace", [])
        n_steps = min(len(ref_ids), len(cand_ids), len(ref_topk), len(cand_topk))

        for s in range(n_steps):
            total_steps += 1
            if ref_ids[s] != cand_ids[s]:
                drift_steps += 1
                if first_drift_prompt is None:
                    first_drift_prompt = pid
                    first_drift_step = s

            rtk = ref_topk[s]
            ctk = cand_topk[s]
            _, _, combined = _topk_similarity(
                rtk["ids"], rtk["values"], ctk["ids"], ctk["values"],
            )
            topology_combineds.append(combined)

    return {
        "total_steps": total_steps,
        "drift_steps": drift_steps,
        "drift_ratio": drift_steps / total_steps if total_steps else 0.0,
        "first_drift_prompt": first_drift_prompt,
        "first_drift_step": first_drift_step,
        "topk_combined_min": min(topology_combineds) if topology_combineds else 1.0,
        "topk_combined_mean": (
            sum(topology_combineds) / len(topology_combineds)
            if topology_combineds else 1.0
        ),
    }


# ─────────────────────────────────────────────────────────────
# Isolation modes to test
# ─────────────────────────────────────────────────────────────
ISOLATION_MODES = [
    {
        "id": "scalar_full",
        "label": "FP4xFP8 scalar reference (all 3 projections)",
        "env": TRUE_FP8_SCALAR_ENV,
        "hypothesis": "If drift persists, FP8 quant error causes it (not MMA layout)",
    },
    {
        "id": "scalar_gateup_only",
        "label": "FP4xFP8 scalar gate+up only (down stays BF16)",
        "env": TRUE_FP8_GATEUP_ONLY_ENV,
        "hypothesis": "If drift < full, down_proj re-quant is a major contributor",
    },
    {
        "id": "scalar_down_only",
        "label": "FP4xFP8 scalar down only (gate+up stay BF16)",
        "env": TRUE_FP8_DOWN_ONLY_ENV,
        "hypothesis": "If drift < full, gate/up quant is the major contributor",
    },
    {
        "id": "mma_full",
        "label": "FP4xFP8 MMA path (all 3 projections, production)",
        "env": TRUE_FP8_MMA_ENV,
        "hypothesis": "Baseline from P197 — fragment layout still broken (P191)",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description="P197b drift source isolation probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--modes", default="scalar_full,scalar_gateup_only,scalar_down_only,mma_full",
                    help="Comma-separated list of isolation modes to run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS[:args.limit]
    assert len(prompts) >= 3, f"Need ≥3 prompts, got {len(prompts)}"
    requested_modes = [m.strip() for m in args.modes.split(",")]

    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-p197b-drift-isolation-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "max_new": args.max_new,
        "prompts_count": len(prompts),
        "reference": "convstrict_w4a16",
        "modes_requested": requested_modes,
    }

    # Step 1: Run W4A16 reference (always)
    ref = _run_one(
        model=args.model,
        label="convstrict_w4a16_reference",
        env=CONVSTRICT_ENV,
        prompts=prompts,
        max_new=args.max_new,
        max_seq_len=args.max_seq_len,
    )
    report["reference_load_seconds"] = ref["load_seconds"]

    # Step 2: Run each isolation mode
    results: list[dict[str, Any]] = []
    for mode in ISOLATION_MODES:
        if mode["id"] not in requested_modes:
            continue

        print(f"\n{'=' * 60}", flush=True)
        print(f"[p197b] ISOLATION MODE: {mode['id']}", flush=True)
        print(f"[p197b] {mode['hypothesis']}", flush=True)
        print(f"{'=' * 60}\n", flush=True)

        cand = _run_one(
            model=args.model,
            label=mode["label"],
            env=mode["env"],
            prompts=prompts,
            max_new=args.max_new,
            max_seq_len=args.max_seq_len,
        )

        comp = _compare(ref, cand)
        drift_ratio = comp["drift_ratio"]
        combined_min = comp["topk_combined_min"]

        # Classify
        if comp["drift_steps"] == 0:
            verdict = "NO_DRIFT"
        elif combined_min >= 0.80 and drift_ratio <= 0.25:
            verdict = "AMBER"
        else:
            verdict = "RED"

        result = {
            "mode_id": mode["id"],
            "label": mode["label"],
            "hypothesis": mode["hypothesis"],
            "verdict": verdict,
            "total_steps": comp["total_steps"],
            "drift_steps": comp["drift_steps"],
            "drift_ratio": drift_ratio,
            "topk_combined_min": combined_min,
            "topk_combined_mean": comp["topk_combined_mean"],
            "first_drift_prompt": comp["first_drift_prompt"],
            "first_drift_step": comp["first_drift_step"],
            "candidate_load_seconds": cand["load_seconds"],
        }
        results.append(result)

        print(f"[p197b] {mode['id']}: verdict={verdict} "
              f"drift={comp['drift_steps']}/{comp['total_steps']} "
              f"combined_min={combined_min:.6f}", flush=True)

    report["results"] = results

    # Step 3: Diagnosis
    diagnosis = _diagnose(results)
    report["diagnosis"] = diagnosis

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write MD
    md_path = out_path.with_suffix(".md")
    _write_markdown(md_path, report)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"P197b ISOLATION SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['mode_id']:25s} verdict={r['verdict']:10s} "
              f"drift={r['drift_steps']}/{r['total_steps']} "
              f"combined_min={r['topk_combined_min']:.6f}")
    print(f"\nDiagnosis: {diagnosis['root_cause']}")
    print(f"Next step: {diagnosis['next_step']}")
    print(f"\nJSON: {out_path}")
    print(f"MD:   {md_path}")
    return 0


def _diagnose(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Determine root cause from isolation results."""
    by_id = {r["mode_id"]: r for r in results}

    scalar_full = by_id.get("scalar_full")
    gateup_only = by_id.get("scalar_gateup_only")
    down_only = by_id.get("scalar_down_only")
    mma_full = by_id.get("mma_full")

    # Case 1: Scalar full has NO drift → MMA fragment layout is sole cause
    if scalar_full and scalar_full["verdict"] == "NO_DRIFT":
        return {
            "root_cause": "MMA_FRAGMENT_LAYOUT",
            "confidence": "HIGH",
            "explanation": (
                "Scalar reference path (known correct per P191) produces no drift. "
                "All drift originates from the broken SM120a MMA fragment register packing. "
                "The FP8 quantization itself is not a drift source."
            ),
            "next_step": (
                "Fix the MMA fragment layout (P191 AMBER blocker). Once fragments are "
                "correct, the W4A8 path should pass P197 STRICT."
            ),
            "severity": "FIXABLE",
        }

    # Case 2: Scalar full drifts, but gateup_only doesn't → down_proj re-quant is the cause
    if (scalar_full and scalar_full["drift_steps"] > 0
            and gateup_only and gateup_only["verdict"] == "NO_DRIFT"):
        return {
            "root_cause": "DOWN_PROJ_REQUANT",
            "confidence": "HIGH",
            "explanation": (
                "gate/up FP8 path alone doesn't drift, but adding down_proj FP8 "
                "causes drift. The silu*up intermediate is high-dynamic-range and "
                "re-quantizing it to E4M3 for down_proj loses critical precision."
            ),
            "next_step": (
                "Keep down_proj in BF16 matmul (skip FP8 quant for intermediate). "
                "Or use per-channel (row) scale instead of per-16 for the intermediate."
            ),
            "severity": "FIXABLE",
        }

    # Case 3: Scalar full drifts, down_only doesn't → gate/up quant is the cause
    if (scalar_full and scalar_full["drift_steps"] > 0
            and down_only and down_only["verdict"] == "NO_DRIFT"):
        return {
            "root_cause": "GATEUP_ACTIVATION_QUANT",
            "confidence": "HIGH",
            "explanation": (
                "down_proj FP8 alone doesn't drift, but gate/up FP8 does. "
                "The hidden-state quantization to E4M3 per-16 loses precision in "
                "early-layer activations that propagates through residual connections."
            ),
            "next_step": (
                "Try per-tensor scale for gate/up activation (less granular but fewer "
                "outlier truncation events). Or increase to per-8 grouping."
            ),
            "severity": "FIXABLE",
        }

    # Case 4: Both gateup and down contribute
    if (scalar_full and scalar_full["drift_steps"] > 0
            and gateup_only and gateup_only["drift_steps"] > 0
            and down_only and down_only["drift_steps"] > 0):
        # Compare magnitudes
        gu_ratio = gateup_only["drift_ratio"]
        dn_ratio = down_only["drift_ratio"]
        full_ratio = scalar_full["drift_ratio"]
        return {
            "root_cause": "COMPOUND_QUANT_ERROR",
            "confidence": "MEDIUM",
            "explanation": (
                f"Both gate/up (drift_ratio={gu_ratio:.3f}) and down "
                f"(drift_ratio={dn_ratio:.3f}) contribute to full drift "
                f"(drift_ratio={full_ratio:.3f}). Residual accumulation "
                f"compounds per-layer FP8 quantization noise."
            ),
            "next_step": (
                "1) Try selective layer bypass: FP8 only on layers 0-15, BF16 on 16+. "
                "2) Try per-row activation scale. "
                "3) Consider W4A8 only for prefill (batch>1) where error averages out."
            ),
            "severity": "HARD",
        }

    # Case 5: Only scalar_full ran, and it drifts
    if scalar_full and scalar_full["drift_steps"] > 0:
        return {
            "root_cause": "FP8_QUANTIZATION_GENERAL",
            "confidence": "LOW",
            "explanation": (
                "Scalar reference path still drifts. This is fundamental FP8 E4M3 "
                "precision loss in the decode path. Need more isolation (gateup/down split)."
            ),
            "next_step": (
                "Run with --modes scalar_full,scalar_gateup_only,scalar_down_only "
                "to isolate which projection causes drift."
            ),
            "severity": "NEEDS_MORE_DATA",
        }

    return {
        "root_cause": "UNKNOWN",
        "confidence": "LOW",
        "explanation": "Insufficient data to determine root cause.",
        "next_step": "Run all isolation modes.",
        "severity": "UNKNOWN",
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# P197b W4A8 Drift Source Isolation Report",
        "",
        f"**Model:** {report['model']}",
        f"**Created:** {report['created']}",
        f"**Max new tokens:** {report['max_new']}",
        f"**Prompts:** {report['prompts_count']}",
        "",
        "## Isolation Results",
        "",
        "| Mode | Verdict | Drift Steps | Drift Ratio | Combined Min | Combined Mean |",
        "|------|---------|-------------|-------------|--------------|---------------|",
    ]
    for r in report.get("results", []):
        lines.append(
            f"| {r['mode_id']} | {r['verdict']} | "
            f"{r['drift_steps']}/{r['total_steps']} | "
            f"{r['drift_ratio']:.3f} | "
            f"{r['topk_combined_min']:.6f} | "
            f"{r['topk_combined_mean']:.6f} |"
        )

    diag = report.get("diagnosis", {})
    lines += [
        "",
        "## Diagnosis",
        "",
        f"**Root cause:** `{diag.get('root_cause', 'UNKNOWN')}`",
        f"**Confidence:** {diag.get('confidence', '?')}",
        f"**Severity:** {diag.get('severity', '?')}",
        "",
        f"**Explanation:** {diag.get('explanation', 'N/A')}",
        "",
        f"**Next step:** {diag.get('next_step', 'N/A')}",
        "",
        "## Mode Hypotheses",
        "",
    ]
    for r in report.get("results", []):
        lines.append(f"- **{r['mode_id']}**: {r['hypothesis']}")
        if r.get("first_drift_prompt") is not None:
            lines.append(
                f"  - First drift: prompt={r['first_drift_prompt']} "
                f"step={r['first_drift_step']}"
            )
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
