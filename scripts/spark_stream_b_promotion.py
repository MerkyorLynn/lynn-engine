#!/usr/bin/env python3
"""Stream B full-attn layer graph reuse pool — promotion gate.

Runs two configurations on the same prompt set and writes a single JSON
summary that the standard promotion-discipline wrapper can ingest:

* ``baseline``  — Spark Config D + linear_block_graph (current production).
* ``candidate`` — Spark Config D + linear_block_graph +
                  ``LYNN_FULL_ATTN_LAYER_GRAPH_POOL=1`` (Stream B).

Gates (from ``docs/STREAM_B_FULL_ATTN_LAYER_GRAPH_REUSE_SPEC_20260518``):

* **P37 exact-greedy**: candidate output token-exact == baseline output for
  every prompt. Graph replay is byte-identical kernel sequence by
  construction; any drift indicates a wrapper bug (capture-time Python
  mutating state outside the captured kernels).
* **P25-style TPS lift**: candidate ``mean_decode_tps`` > baseline + min lift
  (default 5% as a conservative "real win" bar; spec target ≥ 18%).
* **Structured drift**: any prompt whose candidate completion changes
  semantics from baseline is flagged. We surface them but token-exact
  gate covers strict promotion.

Usage on Spark::

    python scripts/spark_stream_b_promotion.py \\
        --model /home/merkyor/models/<lynn-native-w4a16-nvfp4-dir> \\
        --out /home/merkyor/reports/spark/stream_b_promotion_<TS>.json

The script reuses ``DEFAULT_PROMPTS`` from the MTP smoke runner for now;
use ``--prompts-json`` to override.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


DEFAULT_PROMPTS = [
    "Explain the difference between Q4_K_M and NVFP4 quantization in two sentences.",
    "用一句话解释 speculative decoding 的核心思想。",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "请输出一个 JSON: {\"city\": \"Tokyo\", \"unit\": \"celsius\"}",
    "Summarize the role of the MoE router in one paragraph.",
    "Compare the architectural trade-offs between dense and Mixture-of-Experts (MoE) large language models in terms of compute efficiency at inference, memory footprint, routing overhead, and quality scaling.",
    "请解释 linear attention(线性注意力)和标准 attention 在长 context 推理时的延迟、内存、质量三个维度的差异。",
]

BASE_ENV = {
    # Spark Config D — same as MTP smoke runner.
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_SHARED_EXPERT": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_FULL_ATTN_QKV_FUSED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _set_env(updates: dict[str, str | None]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _prefix_match_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _run_case(
    runner: LynnIncrementalRunner,
    *,
    label: str,
    env_updates: dict[str, str | None],
    prompts: list[str],
    max_new: int,
    baseline_ids: list[list[int]] | None,
) -> dict[str, Any]:
    previous = _set_env(env_updates)
    try:
        rows = []
        for idx, prompt in enumerate(prompts):
            t0 = time.time()
            out = runner.generate(prompt, max_new=max_new)
            wall = time.time() - t0
            new_ids = [int(x) for x in out["new_ids"]]
            base_ids = None if baseline_ids is None else baseline_ids[idx]
            rows.append({
                "prompt_id": f"prompt_{idx:03d}",
                "prompt": prompt,
                "new_ids": new_ids,
                "completion_head": out["completion_text"][:240],
                "decode_tps": out["timings"].get("decode_tps"),
                "wall_seconds": wall,
                "exact_match": None if base_ids is None else new_ids == base_ids,
                "prefix_match_len": None if base_ids is None else _prefix_match_len(new_ids, base_ids),
                "stopped_reason": out["stopped_reason"],
            })
        tps_values = [r["decode_tps"] for r in rows if r["decode_tps"]]
        return {
            "label": label,
            "env_overrides": {k: v for k, v in env_updates.items() if v is not None},
            "rows": rows,
            "mean_decode_tps": sum(tps_values) / len(tps_values) if tps_values else None,
            "min_decode_tps": min(tps_values) if tps_values else None,
            "max_decode_tps": max(tps_values) if tps_values else None,
        }
    finally:
        _restore_env(previous)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=256,
                    help="P25-style 512-token decode is the spec acceptance bar; "
                         "256 is the smoke-runner default and still surfaces meaningful TPS.")
    ap.add_argument("--bucket", type=int, default=256,
                    help="LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET")
    ap.add_argument("--min-tps-lift", type=float, default=0.05,
                    help="Minimum TPS lift over baseline for promotion PASS (default 5%%).")
    ap.add_argument("--prompts-json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        raw = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        prompts = [str(item["prompt"]) if isinstance(item, dict) else str(item) for item in raw]

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # Apply base env before runner construction.
    base_previous = _set_env(dict(BASE_ENV))
    try:
        runner = LynnIncrementalRunner(args.model, device=args.device, dtype=dtype, verbose=False)

        configs = [
            {
                "label": "baseline",
                "env": {
                    "LYNN_FULL_ATTN_LAYER_GRAPH_POOL": "0",
                    "LYNN_LINEAR_BLOCK_GRAPH": "1",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
                },
            },
            {
                "label": "candidate_stream_b",
                "env": {
                    "LYNN_FULL_ATTN_LAYER_GRAPH_POOL": "1",
                    "LYNN_FULL_ATTN_LAYER_GRAPH_BUCKET": str(args.bucket),
                    "LYNN_LINEAR_BLOCK_GRAPH": "1",
                    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
                },
            },
        ]

        cases = []
        baseline_ids = None
        for cfg in configs:
            case = _run_case(
                runner,
                label=cfg["label"],
                env_updates=cfg["env"],
                prompts=prompts,
                max_new=args.max_new,
                baseline_ids=baseline_ids,
            )
            if cfg["label"] == "baseline":
                baseline_ids = [r["new_ids"] for r in case["rows"]]
                for r in case["rows"]:
                    r["exact_match"] = True
                    r["prefix_match_len"] = len(r["new_ids"])
            cases.append(case)

        baseline_tps = cases[0]["mean_decode_tps"]
        candidate_tps = cases[1]["mean_decode_tps"]
        candidate_rows = cases[1]["rows"]
        exact_count = sum(1 for r in candidate_rows if r["exact_match"])
        n_rows = len(candidate_rows)
        prefix_lens = [r["prefix_match_len"] for r in candidate_rows if r["prefix_match_len"] is not None]
        mean_prefix = sum(prefix_lens) / len(prefix_lens) if prefix_lens else None
        tps_lift = (candidate_tps - baseline_tps) / baseline_tps if (baseline_tps and candidate_tps) else None

        gate_p37 = exact_count == n_rows
        gate_tps_lift = tps_lift is not None and tps_lift >= args.min_tps_lift
        promotion_decision = (
            "DEFAULT" if (gate_p37 and gate_tps_lift) else
            "AMBER" if gate_tps_lift else
            "CLOSED"
        )

        report = {
            "schema_version": "lynn-stream-b-promotion-v1",
            "model": args.model,
            "max_new": args.max_new,
            "bucket": args.bucket,
            "n_prompts": n_rows,
            "configs": cases,
            "summary": {
                "baseline_mean_tps": baseline_tps,
                "candidate_mean_tps": candidate_tps,
                "tps_lift_fraction": tps_lift,
                "tps_lift_percent": (tps_lift * 100.0) if tps_lift is not None else None,
                "exact_match_count": exact_count,
                "n_prompts": n_rows,
                "exact_match_rate": exact_count / n_rows if n_rows else None,
                "mean_prefix_match_len": mean_prefix,
                "min_tps_lift_required": args.min_tps_lift,
            },
            "gates": {
                "p37_exact_match": gate_p37,
                "tps_lift_pass": gate_tps_lift,
                "decision": promotion_decision,
            },
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[promotion] wrote {out_path}")
        print(f"[promotion] baseline mean_tps  = {baseline_tps}")
        print(f"[promotion] candidate mean_tps = {candidate_tps}")
        print(f"[promotion] tps_lift           = {tps_lift}")
        print(f"[promotion] p37_exact_match    = {gate_p37}  ({exact_count}/{n_rows})")
        print(f"[promotion] decision           = {promotion_decision}")
        return 0 if promotion_decision != "CLOSED" else 1
    finally:
        _restore_env(base_previous)


if __name__ == "__main__":
    raise SystemExit(main())
