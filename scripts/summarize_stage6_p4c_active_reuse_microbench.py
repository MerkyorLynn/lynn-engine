#!/usr/bin/env python3
"""Summarize Stage 6 P4C active-reuse microbench artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4c-active-reuse-microbench-v1"
PASS_DECISION = "PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    bench = data.get("bench") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "top-level decision mismatch"
    if data.get("banked_p4c_active_reuse_speed_baseline") is not True:
        return "FAIL", "P4C speed baseline was not banked"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    for gate in (
        "numeric_vs_reference",
        "timing_recorded",
        "speed_floor_recorded",
        "zero_bf16_shadow_weight_abi",
        "active_scratch_reuse_abi",
        "packed_byte_budget",
        "promotion_boundary_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if not (bench.get("reference_p4a_two_stage") or {}).get("median_us"):
        return "FAIL", "missing reference timing"
    if not (bench.get("candidate_p4c_active_reuse_contract") or {}).get("median_us"):
        return "FAIL", "missing candidate timing"
    return "PASS", "P4C active-reuse speed baseline recorded; speed/default promotion still closed"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    bench = data.get("bench") or {}
    ref = bench.get("reference_p4a_two_stage") or {}
    cand = bench.get("candidate_p4c_active_reuse_contract") or {}
    numeric = data.get("numeric_vs_reference") or {}
    out_diff = numeric.get("out") or {}
    inter_diff = numeric.get("inter_scratch") or {}
    byte_budget = data.get("byte_budget") or {}
    lines = [
        "# Stage 6 P4C Active-Reuse Microbench Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Torch/CUDA | `{data.get('torch_version')}` / `{data.get('torch_cuda')}` |",
        f"| Banked P4C speed baseline | `{data.get('banked_p4c_active_reuse_speed_baseline')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| P4A two-stage median | `{ref.get('median_us')}` us |",
        f"| P4C active-reuse median | `{cand.get('median_us')}` us |",
        f"| P4C/P4A speedup | `{bench.get('candidate_vs_reference_speedup')}` |",
        f"| P4C minus P4A | `{bench.get('candidate_minus_reference_us')}` us |",
        f"| Speed ratio floor | `{data.get('speed_ratio_floor')}` |",
        f"| Output rel L2 | `{out_diff.get('rel_l2')}` |",
        f"| Output max abs | `{out_diff.get('max_abs')}` |",
        f"| Scratch rel L2 | `{inter_diff.get('rel_l2')}` |",
        f"| Scratch max abs | `{inter_diff.get('max_abs')}` |",
        f"| Active scratch bytes | `{byte_budget.get('active_scratch_bytes')}` |",
        f"| Zero BF16 shadow weight ABI | `{byte_budget.get('zero_bf16_shadow_weight_abi')}` |",
        f"| Packed/BF16 ratio | `{byte_budget.get('packed_vs_bf16_shadow_ratio')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
        "",
        "## Boundary",
        "",
        "- This banks only `banked_p4c_active_reuse_speed_baseline=true`.",
        "- It is not a fused-kernel speed win and not a default-promotion gate.",
        "- The next real implementation step is replacing the P4C symbol body with a faster active-reuse CUDA/CUTLASS-style candidate while preserving this ABI.",
    ]
    error = data.get("load_error_tail")
    if error:
        lines.extend(["", "## Error Tail", "", "```text", str(error)[-1200:], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json")
    ap.add_argument("--markdown-out", default="")
    ap.add_argument("--strict-exit", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.result_json).read_text())
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
