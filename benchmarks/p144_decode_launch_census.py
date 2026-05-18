#!/usr/bin/env python3
"""P144: decode CUDA launch census for Qwen3.6 W4A16 serving candidates.

This is a read-only profiling scaffold. It measures a short greedy decode with
torch.profiler and reports CUDA kernel launch count, launches/token, top kernels
by self CUDA time, and coarse kernel groups. It does not change runtime state or
promotion decisions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SAFE_DEFAULT_ENV = {
    "LYNN_PREFILL_WARMUP": "1",
    "LYNN_MOE_IMPL": "packed_nvfp4",
    "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton",
    "LYNN_NATIVE_GATEUP_BACKEND": "triton_fast_decode",
    "LYNN_NATIVE_DOWN_BACKEND": "triton",
    "LYNN_SHARED_EXPERT_GATE_BACKEND": "torch",
    "LYNN_LINEAR_ATTN_CONV_BACKEND": "triton_torch_silu",
    "LYNN_QK_NORM_ROPE_BACKEND": "triton_pair",
    "LYNN_RMSNORM_GATED_BACKEND": "triton",
    "LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4": "1",
    "LYNN_NATIVE_FP4_LM_HEAD": "1",
    "LYNN_LINEAR_STATE_UPDATE": "inplace",
    "LYNN_FULL_TOKEN_GRAPH_SLOT": "0",
}


def _read_profile_env(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = value
    out: dict[str, str] = {}
    for raw in re.split(r"[\n, ]+", text):
        raw = raw.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip().strip("\"'")
    return out


def _set_env(updates: dict[str, str]) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _kernel_group(name: str) -> str:
    n = name.lower()
    if any(s in n for s in ("moe", "expert", "nvfp4", "e2m1")):
        return "moe_nvfp4"
    if any(s in n for s in ("attention", "attn", "flash", "rope", "qk_norm", "softmax")):
        return "attention_rope"
    if any(s in n for s in ("mamba", "gdn", "delta", "scan", "conv1d", "linear_attn")):
        return "linear_gdn_ssm"
    if any(s in n for s in ("rms", "norm")):
        return "norm"
    if any(s in n for s in ("gemm", "matmul", "mm", "cublas", "cutlass")):
        return "gemm"
    if any(s in n for s in ("copy", "memcpy", "fill", "zero")):
        return "memory_fill_copy"
    return "other"


def _is_cuda_event(evt: Any) -> bool:
    device_type = getattr(evt, "device_type", None)
    if device_type is not None and str(device_type).lower().endswith("cuda"):
        return True
    return float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0) > 0.0


def _event_self_cuda_us(evt: Any) -> float:
    return float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0)


def _profile(model: str, prompt: str, warmup_new: int, max_new: int, use_chat_template: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; P144 must run on R6000/Spark GPU host")

    from torch.profiler import ProfilerActivity, profile
    from engine.resident_runner import LynnIncrementalRunner

    runner = LynnIncrementalRunner(model, device="cuda", dtype=torch.bfloat16, verbose=False)
    if warmup_new > 0:
        runner.generate(prompt, max_new=warmup_new, use_chat_template=use_chat_template)
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
    wall = time.perf_counter() - start

    events = [evt for evt in prof.events() if _is_cuda_event(evt)]
    by_name: dict[str, dict[str, Any]] = {}
    for evt in events:
        name = getattr(evt, "name", None) or getattr(evt, "key", None) or "<unknown>"
        rec = by_name.setdefault(name, {"name": name, "count": 0, "self_cuda_time_us": 0.0})
        rec["count"] += 1
        rec["self_cuda_time_us"] += _event_self_cuda_us(evt)

    top = sorted(by_name.values(), key=lambda r: r["self_cuda_time_us"], reverse=True)[:30]
    groups: dict[str, dict[str, Any]] = {}
    for name, rec in by_name.items():
        g = _kernel_group(name)
        grec = groups.setdefault(g, {"group": g, "count": 0, "self_cuda_time_us": 0.0})
        grec["count"] += int(rec["count"])
        grec["self_cuda_time_us"] += float(rec["self_cuda_time_us"])

    tokens = len(out.get("new_ids", [])) or max_new
    return {
        "tokens_profiled": tokens,
        "cuda_launch_count_total": len(events),
        "cuda_launches_per_token": len(events) / max(tokens, 1),
        "wall_ms_total": wall * 1000.0,
        "wall_ms_per_token": (wall * 1000.0) / max(tokens, 1),
        "decode_tps_estimate": max(tokens, 1) / wall if wall > 0 else None,
        "runner_decode_tps": (out.get("timings") or {}).get("decode_tps"),
        "new_ids": out.get("new_ids", []),
        "top_kernels_by_self_cuda_time": top,
        "grouped_by_name_prefix": sorted(groups.values(), key=lambda r: r["self_cuda_time_us"], reverse=True),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get(
        "MODEL", "/root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0"))
    ap.add_argument("--prompt", default="用一句话解释 MoE active parameters")
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "12")))
    ap.add_argument("--warmup-new", type=int, default=int(os.environ.get("WARMUP_NEW", "2")))
    ap.add_argument("--profile-env", default=os.environ.get("PROFILE_ENV"))
    ap.add_argument("--out", default=os.environ.get(
        "OUT", "/root/autodl-tmp/reports/qwen36_35b/p144_decode_launch_census.json"))
    ap.add_argument("--use-chat-template", action="store_true")
    args = ap.parse_args()

    env = dict(SAFE_DEFAULT_ENV)
    env.update(_read_profile_env(args.profile_env))
    old = _set_env(env)
    try:
        result = _profile(args.model, args.prompt, args.warmup_new, args.max_new, args.use_chat_template)
    except Exception as exc:  # noqa: BLE001 - report fail-loud reason.
        result = {
            "schema": "lynn-p144-decode-launch-census-v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model,
            "status": "FAILED",
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "env_summary": env,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    finally:
        _restore_env(old)

    report = {
        "schema": "lynn-p144-decode-launch-census-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "status": "OK",
        "env_summary": env,
        **result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tokens_profiled": report["tokens_profiled"],
        "cuda_launches_per_token": report["cuda_launches_per_token"],
        "wall_ms_per_token": report["wall_ms_per_token"],
        "decode_tps_estimate": report["decode_tps_estimate"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
