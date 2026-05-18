#!/usr/bin/env python3
"""P148 · Qwen3.5-9B NVFP4 fast-profile replay.

The 9B release matrix proved Lynn-native W4A16 NVFP4 quality is usable, but its
current runtime is much slower than Q4_K_M/llama.cpp.  Before writing dense FFN
kernels, this probe answers a cheaper question:

    Did the published 9B NVFP4 speed run simply miss the proven Lynn fast knobs?

It runs the same model twice in one process:

* baseline: explicit conservative resident settings;
* fast: 35B-safe graph/in-place/packed-decode knobs that should be harmless for
  the 9B dense architecture.

The output compares greedy token exactness and decode TPS.  A fast mode that
drifts stays research-only; a fast mode that is exact but still slow points the
next work at dense FFN/kernel-boundary profiling.
"""
from __future__ import annotations

import argparse
import gc
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
    "用一句话解释 CUDA graph 对推理速度的帮助。",
    "Python 写一个函数判断字符串是否为回文。",
    "Return a compact JSON object with keys city and country for Paris.",
]


BASELINE_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "torch",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "0",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "0",
    "LYNN_LINEAR_ATTN_CONV_BACKEND": "torch",
    "LYNN_QK_NORM_ROPE_BACKEND": "torch",
    "LYNN_RMSNORM_GATED_BACKEND": "torch",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "0",
    "LYNN_LINEAR_STATE_UPDATE": "copy",
    "LYNN_LINEAR_BLOCK_GRAPH": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "0",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "0",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "0",
}


FAST_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_LINEAR_BLOCK_GRAPH": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_REUSE": "1",
    "LYNN_LINEAR_BLOCK_GRAPH_PREWARM": "1",
    "LYNN_PACKED_DECODE": "1",
    "LYNN_PACKED_DECODE_PREPARE_NATIVE": "1",
}


def _set_env(env: dict[str, str]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
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
    env: dict[str, str],
    max_new_values: list[int],
    prompts: list[str],
    max_seq_len: int,
) -> dict[str, Any]:
    from engine.resident_runner import LynnIncrementalRunner

    print(f"[p148] loading mode={label}", flush=True)
    old = _set_env(env)
    try:
        t_load0 = time.time()
        runner = LynnIncrementalRunner(
            model,
            device="cuda",
            dtype=torch.bfloat16,
            max_seq_len=max_seq_len,
            verbose=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_seconds = time.time() - t_load0
        rows: list[dict[str, Any]] = []
        for max_new in max_new_values:
            for prompt_id, prompt in enumerate(prompts):
                print(f"[p148] {label} prompt={prompt_id} max_new={max_new}", flush=True)
                out = runner.generate(prompt, max_new=max_new, use_chat_template=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings = out.get("timings", {})
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "max_new": max_new,
                        "new_ids": out.get("new_ids", []),
                        "completion_text": out.get("completion_text", ""),
                        "decode_tps": timings.get("decode_tps"),
                        "wall_tps": timings.get("wall_tps"),
                        "timings": timings,
                    }
                )
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {
            "label": label,
            "env": env,
            "load_seconds": load_seconds,
            "rows": rows,
        }
    finally:
        _restore_env(old)


def _summarize_mode(mode: dict[str, Any]) -> dict[str, Any]:
    rows = mode["rows"]
    decode = [float(r["decode_tps"]) for r in rows if r.get("decode_tps") is not None]
    wall = [float(r["wall_tps"]) for r in rows if r.get("wall_tps") is not None]
    return {
        "label": mode["label"],
        "load_seconds": mode["load_seconds"],
        "decode_tps_mean": sum(decode) / len(decode) if decode else None,
        "decode_tps_min": min(decode) if decode else None,
        "decode_tps_max": max(decode) if decode else None,
        "wall_tps_mean": sum(wall) / len(wall) if wall else None,
    }


def _compare_modes(baseline: dict[str, Any], fast: dict[str, Any]) -> dict[str, Any]:
    fast_by_key = {(r["prompt_id"], r["max_new"]): r for r in fast["rows"]}
    rows = []
    exact = 0
    total = 0
    speedups = []
    for base_row in baseline["rows"]:
        key = (base_row["prompt_id"], base_row["max_new"])
        fast_row = fast_by_key.get(key)
        if fast_row is None:
            continue
        total += 1
        ids_equal = base_row["new_ids"] == fast_row["new_ids"]
        exact += int(ids_equal)
        base_tps = base_row.get("decode_tps")
        fast_tps = fast_row.get("decode_tps")
        speedup = (float(fast_tps) / float(base_tps)) if base_tps and fast_tps else None
        if speedup is not None:
            speedups.append(speedup)
        first_drift = None
        for idx, (a, b) in enumerate(zip(base_row["new_ids"], fast_row["new_ids"])):
            if a != b:
                first_drift = idx
                break
        rows.append(
            {
                "prompt_id": key[0],
                "max_new": key[1],
                "exact": ids_equal,
                "first_drift_index": first_drift,
                "baseline_decode_tps": base_tps,
                "fast_decode_tps": fast_tps,
                "speedup": speedup,
                "baseline_ids_prefix": base_row["new_ids"][:16],
                "fast_ids_prefix": fast_row["new_ids"][:16],
            }
        )
    return {
        "exact_count": exact,
        "total": total,
        "all_exact": exact == total,
        "speedup_mean": sum(speedups) / len(speedups) if speedups else None,
        "speedup_min": min(speedups) if speedups else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B NVFP4 fast-profile replay.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-new", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = args.prompt or PROMPTS
    report: dict[str, Any] = {
        "schema": "lynn-qwen35-9b-nvfp4-fast-profile-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "max_new_values": args.max_new,
        "max_seq_len": args.max_seq_len,
        "prompts": prompts,
    }

    baseline = _run_mode(
        model=args.model,
        label="baseline_conservative",
        env=BASELINE_ENV,
        max_new_values=args.max_new,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    fast = _run_mode(
        model=args.model,
        label="fast_profile",
        env=FAST_ENV,
        max_new_values=args.max_new,
        prompts=prompts,
        max_seq_len=args.max_seq_len,
    )
    comparison = _compare_modes(baseline, fast)
    report["modes"] = [baseline, fast]
    report["summaries"] = [_summarize_mode(baseline), _summarize_mode(fast)]
    report["comparison"] = comparison
    report["verdict"] = (
        "FAST_PROFILE_EXACT"
        if comparison["all_exact"]
        else "FAST_PROFILE_DRIFT"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "verdict": report["verdict"],
        "summaries": report["summaries"],
        "comparison": {
            "exact_count": comparison["exact_count"],
            "total": comparison["total"],
            "speedup_mean": comparison["speedup_mean"],
            "speedup_min": comparison["speedup_min"],
        },
        "out": str(out_path),
    }, indent=2, ensure_ascii=False))
    return 0 if comparison["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
