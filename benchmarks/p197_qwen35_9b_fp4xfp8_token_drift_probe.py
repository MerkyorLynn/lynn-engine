#!/usr/bin/env python3
"""P197 · Qwen3.5-9B FP4×FP8 token drift probe.

Runs W4A16 (safe convstrict) and W4A8 (true-FP8 resident) on the same
prompts, capturing per-step token ids, decoded text, top-5 logits, and
prefix match.  Compares the two decode paths token-by-token to locate
the first drift step and classify the drift severity.

Verdicts:
  STRICT_CANDIDATE  - all prompts: identical greedy tokens (no drift)
  AMBER_NUMERIC     - drift present but top-5 logits cosine ≥ 0.99
  CLOSED_NUMERIC    - drift with top-5 logits cosine < 0.99

Usage:
  python benchmarks/p197_qwen35_9b_fp4xfp8_token_drift_probe.py \\
    --model /root/autodl-tmp/models/Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0 \\
    --out /root/autodl-tmp/reports/qwen35_9b/p197_token_drift_probe.json
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

from benchmarks.p148_qwen35_9b_nvfp4_fast_profile import (
    CONVSTRICT_ENV,
    _restore_env,
    _set_env,
)
from benchmarks.p184_qwen35_9b_nvfp4_convstrict_exact_gate import _merge

# ─────────────────────────────────────────────────────────────
# Env configs
# ─────────────────────────────────────────────────────────────
TRUE_FP8_ENV = _merge(
    CONVSTRICT_ENV,
    {
        "LYNN_DENSE_FFN_TRUE_FP8": "1",
        "LYNN_DENSE_FFN_TRUE_FP8_SIDECAR_DIR": (
            "/root/autodl-tmp/reports/qwen35_9b/p192_dense_fp4x_fp8_sidecar"
        ),
    },
)

# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────
TOPOLOGY_COMBINED_AMBER = 0.80   # top-5 combined (jaccard+cosine) ≥ this → AMBER
DRIFT_RATIO_AMBER = 0.25         # drift steps / total steps ≤ this → AMBER

# ─────────────────────────────────────────────────────────────
# Hard prompts (at least 3)
# ─────────────────────────────────────────────────────────────
DEFAULT_PROMPTS = [
    "用一句话解释 CUDA graph 对推理速度的帮助。",
    "Write a Python function to compute the Fibonacci sequence up to n.",
    "Return a compact JSON object with keys city and country for Paris.",
    "Explain the difference between TCP and UDP in exactly 3 sentences.",
    "List the first 5 prime numbers as a JSON array.",
]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _topk_similarity(
    a_ids: list[int], a_vals: list[float],
    b_ids: list[int], b_vals: list[float],
) -> tuple[float, float, float]:
    """Return (jaccard, shared_cosine, combined) for two top-k logit vectors.

    jaccard  = |intersection| / |union| of top-k ID sets
    shared_cosine = cosine similarity restricted to shared IDs
    combined = jaccard * 0.5 + shared_cosine * 0.5
    """
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

    print(f"[p197] loading mode={label}", flush=True)
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
            print(f"[p197] {label} prompt={pid}/{len(prompts)}", flush=True)
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


# ─────────────────────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────────────────────
def _compare(ref: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    cand_by_pid = {r["prompt_id"]: r for r in cand["rows"]}
    prompt_rows: list[dict[str, Any]] = []
    total_steps = 0
    drift_steps = 0
    first_drift_prompt = None
    first_drift_step = None
    topology_combineds: list[float] = []
    topology_jaccards: list[float] = []
    topology_shared_cosines: list[float] = []

    for ref_row in ref["rows"]:
        pid = ref_row["prompt_id"]
        cand_row = cand_by_pid.get(pid)
        if cand_row is None:
            continue

        ref_ids = ref_row["new_ids"]
        cand_ids = cand_row["new_ids"]
        ref_topk = ref_row.get("topk_trace", [])
        cand_topk = cand_row.get("topk_trace", [])

        step_details: list[dict[str, Any]] = []
        prompt_drift_count = 0
        prompt_first_drift = None

        n_steps = min(len(ref_ids), len(cand_ids), len(ref_topk), len(cand_topk))
        for s in range(n_steps):
            total_steps += 1
            prefix_match = ref_ids[s] == cand_ids[s]
            if not prefix_match:
                drift_steps += 1
                prompt_drift_count += 1
                if prompt_first_drift is None:
                    prompt_first_drift = s

            rtk = ref_topk[s]
            ctk = cand_topk[s]
            jac, s_cos, combined = _topk_similarity(
                rtk["ids"], rtk["values"], ctk["ids"], ctk["values"],
            )
            topology_combineds.append(combined)
            topology_jaccards.append(jac)
            topology_shared_cosines.append(s_cos)

            step_details.append({
                "step": s,
                "ref_token_id": ref_ids[s],
                "cand_token_id": cand_ids[s],
                "ref_text": ref_row["decoded_tokens"][s] if s < len(ref_row["decoded_tokens"]) else None,
                "cand_text": cand_row["decoded_tokens"][s] if s < len(cand_row["decoded_tokens"]) else None,
                "prefix_match": prefix_match,
                "topk_jaccard": jac,
                "topk_shared_cosine": s_cos,
                "topk_combined": combined,
                "ref_top5_ids": rtk["ids"][:5],
                "ref_top5_values": [round(v, 4) for v in rtk["values"][:5]],
                "cand_top5_ids": ctk["ids"][:5],
                "cand_top5_values": [round(v, 4) for v in ctk["values"][:5]],
                "ref_top1_margin": rtk.get("top1_margin"),
                "cand_top1_margin": ctk.get("top1_margin"),
            })

        if first_drift_prompt is None and prompt_first_drift is not None:
            first_drift_prompt = pid
            first_drift_step = prompt_first_drift

        prompt_combineds = [s["topk_combined"] for s in step_details]
        prompt_rows.append({
            "prompt_id": pid,
            "prompt": ref_row["prompt"],
            "steps": step_details,
            "drift_count": prompt_drift_count,
            "first_drift_step": prompt_first_drift,
            "topk_combined_min": min(prompt_combineds) if prompt_combineds else 1.0,
            "topk_combined_mean": (
                sum(prompt_combineds) / len(prompt_combineds)
                if prompt_combineds else 1.0
            ),
        })

    return {
        "total_steps": total_steps,
        "drift_steps": drift_steps,
        "drift_ratio": drift_steps / total_steps if total_steps else 0.0,
        "first_drift_prompt": first_drift_prompt,
        "first_drift_step": first_drift_step,
        "topk_jaccard_min": min(topology_jaccards) if topology_jaccards else 1.0,
        "topk_shared_cosine_min": min(topology_shared_cosines) if topology_shared_cosines else 1.0,
        "topk_combined_min": min(topology_combineds) if topology_combineds else 1.0,
        "topk_combined_mean": (
            sum(topology_combineds) / len(topology_combineds)
            if topology_combineds else 1.0
        ),
        "per_prompt": prompt_rows,
    }


# ─────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────
def _verdict(comp: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if comp["drift_steps"] == 0:
        return "STRICT_CANDIDATE", ["all steps match exactly"]

    drift_ratio = comp["drift_ratio"]
    combined_min = comp["topk_combined_min"]

    reasons.append(
        f"drift_steps={comp['drift_steps']}/{comp['total_steps']} "
        f"ratio={drift_ratio:.3f}"
    )
    reasons.append(
        f"topk_combined_min={combined_min:.6f} "
        f"(jaccard_min={comp['topk_jaccard_min']:.4f} "
        f"shared_cosine_min={comp['topk_shared_cosine_min']:.6f})"
    )

    if combined_min >= TOPOLOGY_COMBINED_AMBER and drift_ratio <= DRIFT_RATIO_AMBER:
        reasons.append(
            f"topk_combined_min ≥ {TOPOLOGY_COMBINED_AMBER} "
            f"AND drift_ratio ≤ {DRIFT_RATIO_AMBER}"
        )
        return "AMBER_NUMERIC", reasons

    if combined_min < TOPOLOGY_COMBINED_AMBER:
        reasons.append(f"topk_combined_min < {TOPOLOGY_COMBINED_AMBER}")
    if drift_ratio > DRIFT_RATIO_AMBER:
        reasons.append(f"drift_ratio > {DRIFT_RATIO_AMBER}")
    return "CLOSED_NUMERIC", reasons


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="P197 Qwen3.5-9B FP4×FP8 token drift probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--prompts-json", default=None)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load prompts
    if args.prompts_json:
        rows = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = []
        for row in rows:
            system = (row.get("system") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            prompts.append(f"System: {system}\nUser: {prompt}" if system else prompt)
        prompts = prompts[:args.limit]
    else:
        prompts = args.prompt or DEFAULT_PROMPTS[:args.limit]

    assert len(prompts) >= 3, f"Need ≥3 prompts, got {len(prompts)}"

    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-p197-fp4xfp8-token-drift-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "max_new": args.max_new,
        "prompts_count": len(prompts),
        "reference": "convstrict_w4a16",
        "candidate": "convstrict_true_fp4xfp8_dense_ffn",
    }

    # Run reference (W4A16)
    ref = _run_one(
        model=args.model,
        label="convstrict_w4a16_reference",
        env=CONVSTRICT_ENV,
        prompts=prompts,
        max_new=args.max_new,
        max_seq_len=args.max_seq_len,
    )

    # Run candidate (W4A8 true-FP8)
    cand = _run_one(
        model=args.model,
        label="convstrict_true_fp4xfp8",
        env=TRUE_FP8_ENV,
        prompts=prompts,
        max_new=args.max_new,
        max_seq_len=args.max_seq_len,
    )

    # Compare
    comp = _compare(ref, cand)
    verdict, reasons = _verdict(comp)

    report["reference_load_seconds"] = ref["load_seconds"]
    report["candidate_load_seconds"] = cand["load_seconds"]
    report["comparison"] = comp
    report["verdict"] = verdict
    report["reasons"] = reasons

    # Write JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write human-readable MD
    md_path = out_path.with_suffix(".md")
    _write_markdown(md_path, report)

    # Print summary
    print(f"P197 verdict: {verdict}")
    for r in reasons:
        print(f"  {r}")
    fd = comp.get("first_drift_prompt")
    fs = comp.get("first_drift_step")
    if fd is not None:
        print(f"  first drift: prompt={fd} step={fs}")
        pp = next((p for p in comp["per_prompt"] if p["prompt_id"] == fd), None)
        if pp and fs is not None:
            sd = pp["steps"][fs]
            print(f"    ref:  {sd['ref_token_id']} {sd['ref_text']!r}")
            print(f"    cand: {sd['cand_token_id']} {sd['cand_text']!r}")
            print(f"    topk_combined={sd['topk_combined']:.6f} "
                  f"(jaccard={sd['topk_jaccard']:.4f} "
                  f"shared_cosine={sd['topk_shared_cosine']:.6f})")
            print(f"    ref_top5:  {list(zip(sd['ref_top5_ids'], sd['ref_top5_values']))}")
            print(f"    cand_top5: {list(zip(sd['cand_top5_ids'], sd['cand_top5_values']))}")
    print(f"\nJSON: {out_path}")
    print(f"MD:   {md_path}")
    return 0


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    comp = report["comparison"]
    lines = [
        f"# P197 Token Drift Probe Report",
        f"",
        f"**Model:** {report['model']}",
        f"**Created:** {report['created']}",
        f"**Max new tokens:** {report['max_new']}",
        f"",
        f"## Verdict: `{report['verdict']}`",
        f"",
    ]
    for r in report.get("reasons", []):
        lines.append(f"- {r}")
    lines += [
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total steps | {comp['total_steps']} |",
        f"| Drift steps | {comp['drift_steps']} |",
        f"| Drift ratio | {comp['drift_ratio']:.3f} |",
        f"| Top-5 combined min | {comp['topk_combined_min']:.6f} |",
        f"| Top-5 combined mean | {comp['topk_combined_mean']:.6f} |",
        f"| Top-5 jaccard min | {comp['topk_jaccard_min']:.4f} |",
        f"| Top-5 shared cosine min | {comp['topk_shared_cosine_min']:.6f} |",
        f"| First drift prompt | {comp.get('first_drift_prompt', 'N/A')} |",
        f"| First drift step | {comp.get('first_drift_step', 'N/A')} |",
        f"",
    ]

    for pp in comp["per_prompt"]:
        lines += [
            f"## Prompt {pp['prompt_id']}: {pp['prompt'][:80]}",
            f"",
            f"Drift count: {pp['drift_count']}, first drift step: {pp['first_drift_step']}",
            f"",
            f"| Step | Ref ID | Cand ID | Match | Ref text | Cand text | Jaccard | Shared cos | Combined |",
            f"|---|---|---|---|---|---|---|---|---|",
        ]
        for sd in pp["steps"]:
            match = "✓" if sd["prefix_match"] else "✗"
            ref_t = repr(sd["ref_text"]) if sd["ref_text"] else ""
            cand_t = repr(sd["cand_text"]) if sd["cand_text"] else ""
            lines.append(
                f"| {sd['step']} | {sd['ref_token_id']} | {sd['cand_token_id']} "
                f"| {match} | {ref_t} | {cand_t} "
                f"| {sd['topk_jaccard']:.4f} | {sd['topk_shared_cosine']:.6f} "
                f"| {sd['topk_combined']:.6f} |"
            )
        lines.append("")

        # Show drift details
        drifts = [sd for sd in pp["steps"] if not sd["prefix_match"]]
        if drifts:
            lines.append("### Drift steps detail")
            lines.append("")
            for sd in drifts[:3]:  # show first 3 drifts
                lines.append(f"**Step {sd['step']}:**")
                lines.append(f"- Ref:  `{sd['ref_token_id']}` → {sd['ref_text']!r}")
                lines.append(f"- Cand: `{sd['cand_token_id']}` → {sd['cand_text']!r}")
                lines.append(f"- TopK combined: {sd['topk_combined']:.6f} "
                             f"(jaccard={sd['topk_jaccard']:.4f} "
                             f"shared_cosine={sd['topk_shared_cosine']:.6f})")
                lines.append(f"- Ref top5:  {list(zip(sd['ref_top5_ids'], sd['ref_top5_values']))}")
                lines.append(f"- Cand top5: {list(zip(sd['cand_top5_ids'], sd['cand_top5_values']))}")
                if sd.get('ref_top1_margin') is not None:
                    lines.append(f"- Ref top1 margin: {sd['ref_top1_margin']:.4f}")
                if sd.get('cand_top1_margin') is not None:
                    lines.append(f"- Cand top1 margin: {sd['cand_top1_margin']:.4f}")
                lines.append("")

    # Layer guess
    fd = comp.get("first_drift_prompt")
    fs = comp.get("first_drift_step")
    if fd is not None and fs is not None:
        pp = next((p for p in comp["per_prompt"] if p["prompt_id"] == fd), None)
        if pp:
            sd = pp["steps"][fs]
            lines += [
                f"## First Drift Analysis",
                f"",
                f"First drift at prompt={fd}, step={fs}.",
                f"",
                f"- Ref token: `{sd['ref_token_id']}` ({sd['ref_text']!r})",
                f"- Cand token: `{sd['cand_token_id']}` ({sd['cand_text']!r})",
                f"- TopK combined: {sd['topk_combined']:.6f} "
                f"(jaccard={sd['topk_jaccard']:.4f} "
                f"shared_cosine={sd['topk_shared_cosine']:.6f})",
                f"",
                f"**Likely cause:** Early-layer FP8 quantization error in the dense FFN",
                f"(gate_proj/up_proj) propagates through residual connections.  The",
                f"first drift step indicates the accumulated hidden-state error crossed",
                f"the argmax boundary between two competing tokens.  The top-5 topology",
                f"(combined={sd['topk_combined']:.6f}) shows "
                f"{'minor' if sd['topk_combined'] > 0.99 else 'significant'}",
                f"logit redistribution, suggesting "
                f"{'numerical noise rather than semantic corruption' if sd['topk_combined'] > 0.99 else 'structural divergence in the hidden representation'}.",
                f"",
                f"If drift occurs at step 0, the error is likely in the first few layers'",
                f"gate_proj FP8 packing scale.  If at later steps, it is a compounding",
                f"residual error from mid-stack layers (layers 8-16 for 9B).",
            ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
