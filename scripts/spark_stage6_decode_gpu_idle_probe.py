#!/usr/bin/env python3
"""Stage 6 decode GPU-idle probe.

This is the small gate before investing in a compiled/C++ decode hot loop.
It profiles N and 2N greedy decode runs with the current Spark NVFP4 stack and
uses the delta to cancel prompt/prefill constants. The output estimates how much
of each decode token is actual CUDA kernel work versus host/API gap.

The metric is intentionally conservative:
  * CUDA busy is the summed profiler self CUDA time for GPU events.
  * Host gap = wall delta - CUDA busy delta, clamped at zero.
  * A positive ROI signal is evidence to build a small compiled-loop prototype,
    not a speed promotion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_ENV = {
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_MOE_FAST_FIXED": "1",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
    "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton",
    "LYNN_ROUTER_TOPK_SORTED": "0",
    "LYNN_LINEAR_ATTN_RECURRENT_BACKEND": "triton_fused_prepare",
    "LYNN_LINEAR_ATTN_RECURRENT_INPLACE": "1",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_PACKED_DECODE": "0",
    "LYNN_PACKED_SHARED_EXPERT": "0",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_FULL_ATTN_ROPE_CACHE": "1",
    "LYNN_PACKED_DECODE_BACKEND": "native_fast_2d",
    "LYNN_MOE_DOWN_BLOCK_HIDDEN": "4",
    "LYNN_LINEAR_ATTN_GQA_RECURRENT": "1",
    "LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV": "1",
    "LYNN_RMSNORM_FUSED": "1",
    "LYNN_FULL_ATTN_FUSED": "1",
    "LYNN_SHARED_EXPERT_FUSED": "1",
    "LYNN_LINEAR_ATTN_FUSE_GBETA": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


CUDA_API_KEYS = (
    "cudalaunchkernel",
    "cudagraphlaunch",
    "cudamemcpy",
    "cudamemset",
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "cudaevent",
)


def _is_cuda_event(evt: Any) -> bool:
    device_type = getattr(evt, "device_type", None)
    if device_type is not None and str(device_type).lower().endswith("cuda"):
        return True
    return float(getattr(evt, "self_device_time_total", 0.0) or 0.0) > 0.0 or float(
        getattr(evt, "self_cuda_time_total", 0.0) or 0.0
    ) > 0.0


def _event_cuda_us(evt: Any) -> float:
    return float(getattr(evt, "self_device_time_total", 0.0) or getattr(evt, "self_cuda_time_total", 0.0) or 0.0)


def _event_cpu_us(evt: Any) -> float:
    return float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)


def _event_count(evt: Any) -> int:
    return int(getattr(evt, "count", 1) or 1)


def _is_cuda_api_key(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in CUDA_API_KEYS)


def _top_by(rows: dict[str, dict[str, Any]], field: str, limit: int = 20) -> list[dict[str, Any]]:
    return sorted(rows.values(), key=lambda rec: float(rec.get(field, 0.0)), reverse=True)[:limit]


def _profile_generate(runner: Any, prompt: str, max_new: int, use_chat_template: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        out = runner.generate(prompt, max_new=max_new, use_chat_template=use_chat_template)
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - start

    cuda_by_name: dict[str, dict[str, Any]] = {}
    cpu_api_by_name: dict[str, dict[str, Any]] = {}
    cuda_launch_count = 0
    cuda_self_us_total = 0.0
    cpu_cuda_api_us_total = 0.0
    cpu_cuda_api_count = 0

    for evt in prof.events():
        name = getattr(evt, "name", None) or getattr(evt, "key", None) or "<unknown>"
        if _is_cuda_event(evt):
            count = _event_count(evt)
            cuda_us = _event_cuda_us(evt)
            rec = cuda_by_name.setdefault(name, {"name": name, "count": 0, "self_cuda_time_us": 0.0})
            rec["count"] += count
            rec["self_cuda_time_us"] += cuda_us
            cuda_launch_count += count
            cuda_self_us_total += cuda_us
        elif _is_cuda_api_key(name):
            count = _event_count(evt)
            cpu_us = _event_cpu_us(evt)
            rec = cpu_api_by_name.setdefault(name, {"name": name, "count": 0, "self_cpu_time_us": 0.0})
            rec["count"] += count
            rec["self_cpu_time_us"] += cpu_us
            cpu_cuda_api_count += count
            cpu_cuda_api_us_total += cpu_us

    tokens = len(out.get("new_ids", [])) or max_new
    return {
        "requested_max_new": max_new,
        "tokens": tokens,
        "wall_ms": wall_s * 1000.0,
        "decode_tps_runner": (out.get("timings") or {}).get("decode_tps"),
        "decode_tps_wall": tokens / wall_s if wall_s > 0 else None,
        "cuda_launch_count": cuda_launch_count,
        "cuda_self_time_us": cuda_self_us_total,
        "cpu_cuda_api_count": cpu_cuda_api_count,
        "cpu_cuda_api_time_us": cpu_cuda_api_us_total,
        "top_cuda_events": _top_by(cuda_by_name, "self_cuda_time_us"),
        "top_cpu_cuda_api_events": _top_by(cpu_api_by_name, "self_cpu_time_us"),
        "new_ids_prefix": (out.get("new_ids") or [])[:24],
        "text_prefix": str(out.get("text", ""))[:240],
    }


def _delta(short: dict[str, Any], long: dict[str, Any]) -> dict[str, Any]:
    token_delta = int(long["tokens"]) - int(short["tokens"])
    if token_delta <= 0:
        raise RuntimeError(f"non-positive token delta: short={short['tokens']} long={long['tokens']}")
    wall_ms = float(long["wall_ms"]) - float(short["wall_ms"])
    cuda_busy_ms = (float(long["cuda_self_time_us"]) - float(short["cuda_self_time_us"])) / 1000.0
    cpu_cuda_api_ms = (float(long["cpu_cuda_api_time_us"]) - float(short["cpu_cuda_api_time_us"])) / 1000.0
    launches = int(long["cuda_launch_count"]) - int(short["cuda_launch_count"])
    api_count = int(long["cpu_cuda_api_count"]) - int(short["cpu_cuda_api_count"])

    wall_ms_per_token = wall_ms / token_delta
    cuda_busy_ms_per_token = cuda_busy_ms / token_delta
    host_gap_ms_per_token = max(0.0, wall_ms_per_token - cuda_busy_ms_per_token)
    gpu_busy_ratio = min(cuda_busy_ms_per_token / wall_ms_per_token, 1.0) if wall_ms_per_token > 0 else None
    host_gap_fraction = 1.0 - gpu_busy_ratio if gpu_busy_ratio is not None else None

    if host_gap_fraction is not None and host_gap_fraction >= 0.25 and launches / token_delta >= 500:
        roi_signal = "GO_COMPILED_LOOP_PROTOTYPE"
    elif host_gap_fraction is not None and host_gap_fraction < 0.15:
        roi_signal = "NO_GO_DEEP_RUNTIME_YET"
    else:
        roi_signal = "BORDERLINE_REMEASURE_OR_NSIGHT"

    return {
        "tokens_delta": token_delta,
        "wall_ms_delta": wall_ms,
        "wall_ms_per_token": wall_ms_per_token,
        "cuda_kernel_busy_ms_delta": cuda_busy_ms,
        "cuda_kernel_busy_ms_per_token": cuda_busy_ms_per_token,
        "host_gap_or_idle_ms_per_token_est": host_gap_ms_per_token,
        "gpu_busy_ratio_est": gpu_busy_ratio,
        "host_gap_fraction_est": host_gap_fraction,
        "cuda_launches_delta": launches,
        "cuda_launches_per_token": launches / token_delta,
        "cpu_cuda_api_ms_delta": cpu_cuda_api_ms,
        "cpu_cuda_api_ms_per_token": cpu_cuda_api_ms / token_delta,
        "cpu_cuda_api_calls_delta": api_count,
        "cpu_cuda_api_calls_per_token": api_count / token_delta,
        "compiled_loop_roi_signal": roi_signal,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6 decode GPU-idle probe.")
    ap.add_argument("--model", default=os.environ.get(
        "MODEL", "/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526"
    ))
    ap.add_argument("--prompt", default="If a train travels 60 mph for 2.5 hours, how far does it go? Explain step by step.")
    ap.add_argument("--warmup-new", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--out", default="reports/stage6/decode_gpu_idle_probe_result.json")
    args = ap.parse_args()

    for key, value in BASE_ENV.items():
        os.environ.setdefault(key, value)

    import torch
    from engine.resident_runner import LynnIncrementalRunner

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    runner = LynnIncrementalRunner(args.model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if args.warmup_new > 0:
        runner.generate(args.prompt, max_new=args.warmup_new, use_chat_template=args.use_chat_template)
        torch.cuda.synchronize()

    short = _profile_generate(runner, args.prompt, args.max_new, args.use_chat_template)
    long = _profile_generate(runner, args.prompt, args.max_new * 2, args.use_chat_template)
    delta = _delta(short, long)

    result = {
        "schema": "lynn-stage6-decode-gpu-idle-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "prompt": args.prompt,
        "env_summary": {key: os.environ.get(key) for key in sorted(BASE_ENV)},
        "runs": {
            "short": short,
            "long": long,
        },
        "delta": delta,
        "passes": {
            "token_delta_positive": delta["tokens_delta"] > 0,
            "launches_recorded": delta["cuda_launches_per_token"] > 0,
            "timing_recorded": delta["wall_ms_per_token"] > 0,
            "idle_estimate_recorded": delta["gpu_busy_ratio_est"] is not None,
            "all": True,
        },
        "decision": "PASS_DECODE_GPU_IDLE_PROBE_RECORDED",
        "promotion_boundary": {
            "speed_promotion": False,
            "compiled_loop_default": False,
            "cuda_graph_route": False,
        },
        "caveat": (
            "GPU busy is estimated from PyTorch profiler self CUDA time using N/2N delta; "
            "treat as a go/no-go ROI probe and confirm deep investments with Nsight if borderline."
        ),
    }
    result["passes"]["all"] = all(bool(v) for v in result["passes"].values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "OK",
        "tokens_delta": delta["tokens_delta"],
        "wall_ms_per_token": delta["wall_ms_per_token"],
        "cuda_kernel_busy_ms_per_token": delta["cuda_kernel_busy_ms_per_token"],
        "host_gap_fraction_est": delta["host_gap_fraction_est"],
        "cuda_launches_per_token": delta["cuda_launches_per_token"],
        "compiled_loop_roi_signal": delta["compiled_loop_roi_signal"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
