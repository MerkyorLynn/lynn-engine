#!/usr/bin/env python3
"""P183 · Qwen3.5-9B NVFP4 exact fast-path isolation.

P149 found one strict baseline (`linear_graph_only`, ~60 TPS) and one much
faster service line (`fast_no_packed_decode`, ~75-77 TPS) that drifts.  This
probe narrows the drift source by comparing candidate knob groups against the
strict graph baseline, not against the slow conservative path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from p148_qwen35_9b_nvfp4_fast_profile import (
    BASELINE_ENV,
    FAST_ENV,
    PROMPTS,
    _compare_modes,
    _run_mode,
    _summarize_mode,
)


def _merge(base: dict[str, str], updates: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(updates)
    return out


def _without(env: dict[str, str], *keys: str) -> dict[str, str]:
    out = dict(env)
    for key in keys:
        out[key] = BASELINE_ENV.get(key, "0")
    return out


GRAPH_EXACT_ENV = _merge(
    BASELINE_ENV,
    {
        "LYNN_LINEAR_STATE_UPDATE": "inplace",
        "LYNN_LINEAR_BLOCK_GRAPH": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
        "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    },
)

TRITON_CORE_ENV = _merge(
    GRAPH_EXACT_ENV,
    {
        "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
        "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
        "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
        "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
        "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
        "LYNN_RMSNORM_GATED_BACKEND": "triton",
    },
)


def _candidate_modes() -> list[tuple[str, dict[str, str]]]:
    fast_no_packed = _without(
        FAST_ENV,
        "LYNN_PACKED_DECODE",
        "LYNN_PACKED_DECODE_PREPARE_NATIVE",
        "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4",
    )
    fast_no_packed_no_lm = _without(fast_no_packed, "LYNN_NATIVE_FP4_LM_HEAD")
    return [
        ("graph_exact_selfcheck", dict(GRAPH_EXACT_ENV)),
        (
            "graph_plus_recurrent_prepare",
            _merge(
                GRAPH_EXACT_ENV,
                {
                    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
                    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
                },
            ),
        ),
        (
            "graph_plus_recurrent_prepare_gqa",
            _merge(
                GRAPH_EXACT_ENV,
                {
                    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
                    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
                    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
                },
            ),
        ),
        (
            "graph_plus_conv_triton",
            _merge(GRAPH_EXACT_ENV, {"LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu"}),
        ),
        (
            "graph_plus_qk_rope_triton",
            _merge(GRAPH_EXACT_ENV, {"LYNN_QK_NORM_ROPE_BACKEND": "triton_pair"}),
        ),
        (
            "graph_plus_rmsgated_triton",
            _merge(GRAPH_EXACT_ENV, {"LYNN_RMSNORM_GATED_BACKEND": "triton"}),
        ),
        ("graph_plus_triton_core_no_lm", dict(TRITON_CORE_ENV)),
        (
            "graph_plus_triton_core_no_gqa",
            _merge(TRITON_CORE_ENV, {"LYNN_LINEAR_ATTN_GQA_RECURRENT": "0"}),
        ),
        (
            "graph_plus_triton_core_no_conv",
            _merge(TRITON_CORE_ENV, {"LYNN_LINEAR_ATTN_CONV_BACKEND": "torch"}),
        ),
        (
            "graph_plus_triton_core_no_qkrope",
            _merge(TRITON_CORE_ENV, {"LYNN_QK_NORM_ROPE_BACKEND": "torch"}),
        ),
        (
            "graph_plus_triton_core_no_rmsgated",
            _merge(TRITON_CORE_ENV, {"LYNN_RMSNORM_GATED_BACKEND": "torch"}),
        ),
        (
            "graph_plus_triton_core_native_lm",
            _merge(TRITON_CORE_ENV, {"LYNN_NATIVE_FP4_LM_HEAD": "1"}),
        ),
        ("fast_no_packed_no_lm", fast_no_packed_no_lm),
        ("fast_no_packed_with_lm", fast_no_packed),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B exact fast-path isolation.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = args.prompt or PROMPTS
    max_new_values = [args.max_new]
    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-nvfp4-exact-fast-isolation-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "max_new": args.max_new,
        "max_seq_len": args.max_seq_len,
        "prompts": prompts,
        "strict_reference": "linear_graph_only",
        "candidates": [],
    }

    reference = _run_mode(
        model=args.model,
        label="linear_graph_only_reference",
        env=GRAPH_EXACT_ENV,
        max_new_values=max_new_values,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    report["reference"] = reference
    report["reference_summary"] = _summarize_mode(reference)

    for label, env in _candidate_modes():
        try:
            mode = _run_mode(
                model=args.model,
                label=label,
                env=env,
                max_new_values=max_new_values,
                prompts=prompts,
                max_seq_len=args.max_seq_len,
            )
            comparison = _compare_modes(reference, mode)
            row = {
                "label": label,
                "status": "OK",
                "summary": _summarize_mode(mode),
                "comparison": comparison,
                "env": env,
            }
            print(
                f"[p183] {label}: exact={comparison['exact_count']}/{comparison['total']} "
                f"speedup_mean={comparison['speedup_mean']} "
                f"decode={row['summary'].get('decode_tps_mean')}",
                flush=True,
            )
        except Exception as exc:  # keep sweeping other knobs
            row = {
                "label": label,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "env": env,
            }
            print(f"[p183] {label}: ERROR {type(exc).__name__}: {exc}", flush=True)
        report["candidates"].append(row)

    exact_candidates = [
        c for c in report["candidates"]
        if c.get("status") == "OK" and c.get("comparison", {}).get("all_exact")
    ]
    exact_candidates.sort(
        key=lambda c: c.get("summary", {}).get("decode_tps_mean") or 0.0,
        reverse=True,
    )
    report["best_exact_candidate"] = exact_candidates[0]["label"] if exact_candidates else None
    report["best_exact_decode_tps"] = (
        exact_candidates[0].get("summary", {}).get("decode_tps_mean") if exact_candidates else None
    )
    report["verdict"] = "HAS_EXACT_FAST_CANDIDATE" if exact_candidates else "NO_EXACT_FAST_CANDIDATE"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "verdict": report["verdict"],
        "best_exact_candidate": report["best_exact_candidate"],
        "best_exact_decode_tps": report["best_exact_decode_tps"],
        "reference_summary": report["reference_summary"],
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
