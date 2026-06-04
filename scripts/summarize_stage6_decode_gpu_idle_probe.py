#!/usr/bin/env python3
"""Summarize Stage 6 decode GPU-idle probe artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-decode-gpu-idle-probe-v1"
PASS_DECISION = "PASS_DECODE_GPU_IDLE_PROBE_RECORDED"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "decision mismatch"
    if passes.get("all") is not True:
        return "FAIL", "probe gates did not all pass"
    delta = data.get("delta") or {}
    if not delta.get("cuda_launches_per_token"):
        return "FAIL", "missing launch/token metric"
    if delta.get("gpu_busy_ratio_est") is None:
        return "FAIL", "missing GPU busy estimate"
    return "PASS", "decode GPU-idle ROI probe recorded"


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    delta = data.get("delta") or {}
    short = (data.get("runs") or {}).get("short") or {}
    long = (data.get("runs") or {}).get("long") or {}
    boundary = data.get("promotion_boundary") or {}
    lines = [
        "# Stage 6 Decode GPU-Idle Probe Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| ROI signal | `{delta.get('compiled_loop_roi_signal')}` |",
        f"| Token delta | `{delta.get('tokens_delta')}` |",
        f"| Wall ms/token | `{_fmt(delta.get('wall_ms_per_token'))}` |",
        f"| CUDA kernel busy ms/token | `{_fmt(delta.get('cuda_kernel_busy_ms_per_token'))}` |",
        f"| Estimated host gap/idle ms/token | `{_fmt(delta.get('host_gap_or_idle_ms_per_token_est'))}` |",
        f"| Estimated GPU busy ratio | `{_fmt(delta.get('gpu_busy_ratio_est'))}` |",
        f"| Estimated host gap fraction | `{_fmt(delta.get('host_gap_fraction_est'))}` |",
        f"| CUDA launches/token | `{_fmt(delta.get('cuda_launches_per_token'))}` |",
        f"| CPU CUDA API ms/token | `{_fmt(delta.get('cpu_cuda_api_ms_per_token'))}` |",
        f"| CPU CUDA API calls/token | `{_fmt(delta.get('cpu_cuda_api_calls_per_token'))}` |",
        f"| Short runner TPS | `{_fmt(short.get('decode_tps_runner'))}` |",
        f"| Long runner TPS | `{_fmt(long.get('decode_tps_runner'))}` |",
        f"| Speed promotion | `{boundary.get('speed_promotion')}` |",
        f"| Compiled-loop default | `{boundary.get('compiled_loop_default')}` |",
        f"| CUDA graph route | `{boundary.get('cuda_graph_route')}` |",
        "",
        "## Boundary",
        "",
        "- This banks only a compiled-loop ROI measurement.",
        "- It does not bank a speed gain, a CUDA graph route, or a default runtime change.",
        "- If the host-gap fraction is high, the next step is a small compiled-loop/MTP-light prototype; if low, do not spend month-scale runtime work here.",
        "",
        "## Caveat",
        "",
        str(data.get("caveat", "")),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    md = summarize(data)
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    verdict, _ = _verdict(data)
    return 0 if (verdict == "PASS" or not args.strict_exit) else 2


if __name__ == "__main__":
    raise SystemExit(main())
