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


def _set_runtime_env(
    backend: str,
    *,
    linear_block_graph: bool,
    native_layers: str | None,
) -> dict[str, str | None]:
    graph_flag = "1" if linear_block_graph else "0"
    updates = {
        **BASE_ENV,
        "LYNN_MOE_FAST_FIXED": "0",
        "LYNN_LINEAR_BLOCK_GRAPH": graph_flag,
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": graph_flag,
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": graph_flag,
        "LYNN_NATIVE_ACTIVE_MOE_BACKEND": backend,
    }
    if native_layers is not None:
        updates["LYNN_NATIVE_ACTIVE_MOE_LAYERS"] = native_layers
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
    backend: str,
    max_new: int,
    prompts: list[str],
    linear_block_graph: bool,
    native_layers: str | None,
) -> list[dict[str, Any]]:
    old = _set_runtime_env(backend, linear_block_graph=linear_block_graph, native_layers=native_layers)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "label": label,
                "backend": backend,
                "linear_block_graph": linear_block_graph,
                "native_active_moe_layers": native_layers,
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
    ap.add_argument("--baseline-backend", default="triton")
    ap.add_argument("--candidate-backend", default="strict_fused_boundary")
    ap.add_argument(
        "--linear-block-graph",
        choices=("0", "1"),
        default="1",
        help="Enable the production reusable linear-block CUDA graph path.",
    )
    ap.add_argument(
        "--native-active-moe-layers",
        default=None,
        help="Optional native backend allowlist, e.g. 'full_attention', 'linear_attention', or comma-separated layer ids.",
    )
    args = ap.parse_args()
    linear_block_graph = args.linear_block_graph == "1"

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

    baseline_rows = _run_mode(
        args.model,
        label="baseline",
        backend=args.baseline_backend,
        max_new=args.max_new,
        prompts=prompts,
        linear_block_graph=linear_block_graph,
        native_layers=args.native_active_moe_layers,
    )
    candidate_rows = _run_mode(
        args.model,
        label="candidate",
        backend=args.candidate_backend,
        max_new=args.max_new,
        prompts=prompts,
        linear_block_graph=linear_block_graph,
        native_layers=args.native_active_moe_layers,
    )
    for ref, cand in zip(baseline_rows, candidate_rows, strict=True):
        cand["new_ids_match_reference"] = cand["new_ids"] == ref["new_ids"]
        cand["reference_new_ids"] = ref["new_ids"]

    all_match = all(row["new_ids_match_reference"] for row in candidate_rows)
    baseline_summary = _summary(baseline_rows)
    candidate_summary = _summary(candidate_rows)
    speedup = None
    if baseline_summary["decode_tps_median"] and candidate_summary["decode_tps_median"]:
        speedup = candidate_summary["decode_tps_median"] / baseline_summary["decode_tps_median"]

    result = {
        "schema_version": "lynn-engine-p122-active-moe-strict-boundary-generate-gate-v2",
        "model": args.model,
        "baseline_backend": args.baseline_backend,
        "candidate_backend": args.candidate_backend,
        "linear_block_graph": linear_block_graph,
        "native_active_moe_layers": args.native_active_moe_layers,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "baseline": {"rows": baseline_rows, "summary": baseline_summary},
        "candidate": {"rows": candidate_rows, "summary": candidate_summary},
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
