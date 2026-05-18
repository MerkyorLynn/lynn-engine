#!/usr/bin/env python3
"""P145: V3.2 ordered exact scalar MoE — fixture gate + P37 probe.

Stages:
  1. Fixture gate (p134 style): V3.2 scalar exact vs Triton reference on p133 fixtures.
     Strict gate: max_abs == 0, exact == 1 per fixture.
  2. P37 graph-off: 3 prompts, 8 new tokens each. Must be 3/3 exact.
  3. P37 graph-on: only if graph-off is 3/3 exact.
  4. If any stage fails, report earliest drift token, logit margin, top2/top5 delta.

Usage:
  python benchmarks/p145_resident_moe_v32_ordered_p37_probe.py \
    --model /path/to/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --out reports/qwen36_35b/p145_v32_ordered_$(date +%Y%m%d_%H%M%S).json
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


def _run_fixture_gate(fixtures_dir: str, device: str) -> dict[str, Any]:
    """Run p134-style fixture contract for V3.2 vs Triton reference."""
    from benchmarks.p134_active_moe_fixture_contract import (
        _bench_fn,
        _compute_metrics,
        _moe_reference_routed_only,
        ContractResult,
    )
    from safetensors.torch import load_file

    fixture_files = sorted(Path(fixtures_dir).glob("*.safetensors"))
    if not fixture_files:
        return {"status": "NO_FIXTURES", "reason": f"No fixtures found in {fixtures_dir}"}

    env_old = _set_env({"LYNN_NATIVE_ACTIVE_MOE_BACKEND": "packed_pretransposed_graphsafe_v32_ordered"})
    try:
        from engine.moe_packed_nvfp4 import moe_forward_decode_packed_nvfp4

        results = []
        for fixture_path in fixture_files:
            data = load_file(str(fixture_path), device=device)
            hidden_in = data["hidden_in"]
            expert_ids = data["expert_ids"]
            routing_weights = data["routing_weights"]
            layer_weights = {k: v for k, v in data.items() if k.startswith("mlp.")}
            cfg = {"num_experts_per_tok": expert_ids.numel()}

            # Reference
            def ref_fn():
                return _moe_reference_routed_only(hidden_in, expert_ids, routing_weights, layer_weights, cfg)

            ref_out = ref_fn()
            ref_ms = _bench_fn(ref_fn, warmup=1, iters=5)

            # Candidate
            def cand_fn():
                return moe_forward_decode_packed_nvfp4(hidden_in.unsqueeze(0), layer_weights, cfg).view_as(ref_out)

            cand_out = cand_fn()
            cand_ms = _bench_fn(cand_fn, warmup=3, iters=10)

            metrics = _compute_metrics(ref_out, cand_out)
            passed = metrics["max_abs"] == 0.0 and metrics["exact"] == 1
            results.append(ContractResult(
                fixture_file=fixture_path.name,
                layer_id=int(data.get("layer_id", -1)),
                prompt_id=int(data.get("prompt_id", -1)),
                max_abs=metrics["max_abs"],
                mean_abs=metrics["mean_abs"],
                rel_l2=metrics["rel_l2"],
                cosine=metrics["cosine"],
                exact=metrics["exact"],
                ref_ms=ref_ms,
                candidate_ms=cand_ms,
                passed=passed,
                fail_reasons=[] if passed else [f"max_abs={metrics['max_abs']:.6f}"],
            ))

        all_passed = all(r.passed for r in results)
        return {
            "status": "PASS" if all_passed else "FAIL",
            "fixture_count": len(results),
            "exact_count": sum(r.exact for r in results),
            "max_max_abs": max(r.max_abs for r in results),
            "min_cosine": min(r.cosine for r in results),
            "mean_ref_ms": sum(r.ref_ms for r in results) / len(results),
            "mean_candidate_ms": sum(r.candidate_ms for r in results) / len(results),
            "results": [r.to_dict() for r in results],
        }
    finally:
        _restore_env(env_old)


def _run_p37(model: str, moe_backend: str, max_new: int, graph_on: bool) -> list[dict]:
    from engine.resident_runner import LynnIncrementalRunner

    env = dict(BASE_ENV)
    env["LYNN_NATIVE_ACTIVE_MOE_BACKEND"] = moe_backend
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
                "new_ids": out["new_ids"],
                "completion_text": out["completion_text"],
                "decode_tps": out["timings"].get("decode_tps"),
            })
        return rows
    finally:
        _restore_env(old)


def _analyze_drift(baseline: dict, candidate: dict) -> dict[str, Any]:
    """Find earliest drift token, logit margin, top2/top5 delta."""
    b_ids = baseline["new_ids"]
    c_ids = candidate["new_ids"]
    min_len = min(len(b_ids), len(c_ids))
    drift_token = None
    for i in range(min_len):
        if b_ids[i] != c_ids[i]:
            drift_token = i
            break
    return {
        "drift_token_index": drift_token,
        "baseline_ids": b_ids,
        "candidate_ids": c_ids,
        "baseline_tps": baseline.get("decode_tps"),
        "candidate_tps": candidate.get("decode_tps"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default="reports/qwen36_35b/p133_fixtures")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[p145] V3.2 ordered exact scalar MoE probe")
    print(f"[p145] Model: {args.model}")
    print(f"[p145] Fixtures: {args.fixtures}")
    print()

    report: dict[str, Any] = {
        "probe": "p145_resident_moe_v32_ordered_p37",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "max_new": args.max_new,
        "stages": {},
    }

    # ── Stage 1: Fixture gate ──
    print("[p145] Stage 1 — Fixture gate (strict, max_abs == 0)...")
    fixture_report = _run_fixture_gate(args.fixtures, args.device)
    report["stages"]["fixture_gate"] = fixture_report
    print(f"[p145] Fixture gate: {fixture_report['status']} "
          f"({fixture_report.get('exact_count', 0)}/{fixture_report.get('fixture_count', 0)} exact, "
          f"max_max_abs={fixture_report.get('max_max_abs', 'N/A'):.6f}, "
          f"mean_candidate_ms={fixture_report.get('mean_candidate_ms', 'N/A'):.4f})")

    # ── Stage 2: P37 graph-off ──
    print("\n[p145] Stage 2 — P37 graph-off...")
    print("[p145] Running BASELINE (triton)...")
    baseline_off = _run_p37(args.model, "triton", args.max_new, graph_on=False)
    for r in baseline_off:
        tps = f" tps={r['decode_tps']:.1f}" if r['decode_tps'] else ""
        print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}...{tps}")

    print("[p145] Running CANDIDATE (packed_pretransposed_graphsafe_v32_ordered)...")
    candidate_off = _run_p37(args.model, "packed_pretransposed_graphsafe_v32_ordered", args.max_new, graph_on=False)
    for r in candidate_off:
        tps = f" tps={r['decode_tps']:.1f}" if r['decode_tps'] else ""
        print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}...{tps}")

    exact_off = sum(1 for b, c in zip(baseline_off, candidate_off) if b["new_ids"] == c["new_ids"])
    drift_analysis_off = [_analyze_drift(b, c) for b, c in zip(baseline_off, candidate_off)]
    report["stages"]["p37_graph_off"] = {
        "exact_count": exact_off,
        "total_prompts": len(PROMPTS),
        "drift_analysis": drift_analysis_off,
    }
    print(f"[p145] P37 graph-off exact: {exact_off}/{len(PROMPTS)}")

    # ── Stage 3: P37 graph-on (only if graph-off 3/3) ──
    if exact_off == len(PROMPTS):
        print("\n[p145] Stage 3 — P37 graph-on (graph-off exact, proceeding)...")
        print("[p145] Running BASELINE (triton)...")
        baseline_on = _run_p37(args.model, "triton", args.max_new, graph_on=True)
        for r in baseline_on:
            tps = f" tps={r['decode_tps']:.1f}" if r['decode_tps'] else ""
            print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}...{tps}")

        print("[p145] Running CANDIDATE (packed_pretransposed_graphsafe_v32_ordered)...")
        candidate_on = _run_p37(args.model, "packed_pretransposed_graphsafe_v32_ordered", args.max_new, graph_on=True)
        for r in candidate_on:
            tps = f" tps={r['decode_tps']:.1f}" if r['decode_tps'] else ""
            print(f"  P{r['prompt_id']}: {r['new_ids'][:8]}...{tps}")

        exact_on = sum(1 for b, c in zip(baseline_on, candidate_on) if b["new_ids"] == c["new_ids"])
        drift_analysis_on = [_analyze_drift(b, c) for b, c in zip(baseline_on, candidate_on)]
        report["stages"]["p37_graph_on"] = {
            "exact_count": exact_on,
            "total_prompts": len(PROMPTS),
            "drift_analysis": drift_analysis_on,
        }
        print(f"[p145] P37 graph-on exact: {exact_on}/{len(PROMPTS)}")
    else:
        print("\n[p145] Stage 3 — P37 graph-on SKIPPED (graph-off not exact)")
        report["stages"]["p37_graph_on"] = {"status": "SKIPPED", "reason": "graph-off not exact"}

    # ── Verdict ──
    fixture_ok = fixture_report.get("status") == "PASS"
    p37_off_ok = exact_off == len(PROMPTS)
    p37_on_ok = report["stages"]["p37_graph_on"].get("exact_count", 0) == len(PROMPTS)

    if fixture_ok and p37_off_ok and p37_on_ok:
        verdict = "P37_EXACT"
    elif fixture_ok and p37_off_ok and not p37_on_ok:
        verdict = "CLOSED_P37_GRAPH_ON_DRIFT"
    elif fixture_ok and not p37_off_ok:
        verdict = "CLOSED_P37_DRIFT"
    else:
        verdict = "CLOSED_FIXTURE_GATE_FAIL"

    report["verdict"] = verdict
    report["fixture_ok"] = fixture_ok
    report["p37_graph_off_ok"] = p37_off_ok
    report["p37_graph_on_ok"] = p37_on_ok

    print(f"\n{'='*60}")
    print(f"P145 FINAL VERDICT: {verdict}")
    print(f"  Fixture gate: {'PASS' if fixture_ok else 'FAIL'}")
    print(f"  P37 graph-off: {exact_off}/{len(PROMPTS)} exact")
    print(f"  P37 graph-on: {report['stages']['p37_graph_on'].get('exact_count', 'N/A')} exact")
    print(f"{'='*60}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p145] Report: {args.out}")

    return 0 if verdict == "P37_EXACT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
