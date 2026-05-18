#!/usr/bin/env python3
"""P125: first-token parity ladder for native active-MoE backends.

This probe focuses on the known failure mode for runtime candidates: early
first-token drift. It compares Triton against one or more candidate active-MoE
backends on the same prompt set and reports exact / min-prefix / mean-prefix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.p37_moe_config_generate_gate import BASE_ENV, PROMPTS  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_CANDIDATES = [
    "strict_fused_boundary",
    "grouped_per16_nonatomic",
    "cuda_scalar_contract",
]


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


def _prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if int(x) != int(y):
            break
        n += 1
    return n


def _run_backend(
    model: str,
    *,
    backend: str,
    prompts: list[str],
    max_new: int,
    linear_block_graph: bool,
    native_layers: str | None,
) -> list[dict[str, Any]]:
    old = _set_runtime_env(backend, linear_block_graph=linear_block_graph, native_layers=native_layers)
    try:
        runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
        rows = []
        for idx, prompt in enumerate(prompts):
            out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
            rows.append(
                {
                    "prompt_id": f"prompt_{idx:03d}",
                    "prompt": prompt,
                    "backend": backend,
                    "new_ids": [int(x) for x in out["new_ids"]],
                    "completion_text": out["completion_text"],
                    "timings": out["timings"],
                    "stopped_reason": out["stopped_reason"],
                }
            )
        return rows
    finally:
        _restore_env(old)


def _summarize(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prefixes = []
    decode_tps = []
    exact = 0
    first_flip = None
    for idx, (cand, base) in enumerate(zip(candidate_rows, baseline_rows, strict=True)):
        prefix = _prefix_len(cand["new_ids"], base["new_ids"])
        prefixes.append(prefix)
        if cand["new_ids"] == base["new_ids"]:
            exact += 1
        elif first_flip is None:
            first_flip = {
                "prompt_id": cand["prompt_id"],
                "prompt_index": idx,
                "prefix_match": prefix,
                "baseline_head": base["new_ids"][:16],
                "candidate_head": cand["new_ids"][:16],
            }
        tps = cand["timings"].get("decode_tps")
        if tps is not None:
            decode_tps.append(float(tps))
    return {
        "count": len(candidate_rows),
        "exact": exact,
        "min_prefix": min(prefixes) if prefixes else None,
        "mean_prefix": statistics.fmean(prefixes) if prefixes else None,
        "decode_tps_median": statistics.median(decode_tps) if decode_tps else None,
        "first_flip": first_flip,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--prompts-jsonl")
    ap.add_argument("--baseline-backend", default="triton")
    ap.add_argument("--candidate-backends", nargs="+", default=DEFAULT_CANDIDATES)
    ap.add_argument("--linear-block-graph", choices=("0", "1"), default="1")
    ap.add_argument("--native-active-moe-layers", default=None)
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

    linear_block_graph = args.linear_block_graph == "1"
    baseline_rows = _run_backend(
        args.model,
        backend=args.baseline_backend,
        prompts=prompts,
        max_new=args.max_new,
        linear_block_graph=linear_block_graph,
        native_layers=args.native_active_moe_layers,
    )

    candidates = []
    for backend in args.candidate_backends:
        rows = _run_backend(
            args.model,
            backend=backend,
            prompts=prompts,
            max_new=args.max_new,
            linear_block_graph=linear_block_graph,
            native_layers=args.native_active_moe_layers,
        )
        summary = _summarize(rows, baseline_rows)
        baseline_tps = statistics.median(
            [float(r["timings"]["decode_tps"]) for r in baseline_rows if r["timings"].get("decode_tps") is not None]
        )
        summary["median_speedup"] = (
            summary["decode_tps_median"] / baseline_tps if summary["decode_tps_median"] and baseline_tps else None
        )
        candidates.append({
            "backend": backend,
            "rows": rows,
            "summary": summary,
        })

    result = {
        "schema_version": "lynn-engine-p125-active-moe-first-token-parity-ladder-v1",
        "model": args.model,
        "baseline_backend": args.baseline_backend,
        "candidate_backends": args.candidate_backends,
        "linear_block_graph": linear_block_graph,
        "native_active_moe_layers": args.native_active_moe_layers,
        "max_new": args.max_new,
        "prompt_count": len(prompts),
        "baseline": baseline_rows,
        "candidates": candidates,
        "decision": (
            "Treat min_prefix and first_flip as first-token/early-token safety signals. "
            "Any backend with prefix collapse stays research-only even if local parity is green."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([
        {
            "backend": item["backend"],
            **item["summary"],
        }
        for item in candidates
    ], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
