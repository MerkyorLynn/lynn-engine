#!/usr/bin/env python3
"""Summarize Stage 6 P4B single-CTA microbench artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4b-single-cta-microbench-v1"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    bench = data.get("bench") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_single_cta_microbench") is not True:
        return "FAIL", "single-CTA microbench was not banked"
    if data.get("decision") != "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED":
        return "FAIL", "top-level decision mismatch"
    for gate in ("numeric_vs_reference", "timing_recorded", "no_inter_scratch_candidate_abi", "promotion_boundary_closed", "all"):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if not (bench.get("reference_p4a_two_stage") or {}).get("median_us"):
        return "FAIL", "missing reference timing"
    if not (bench.get("candidate_p4b_single_cta") or {}).get("median_us"):
        return "FAIL", "missing candidate timing"
    return "PASS", "measurement recorded; speed/default promotion still closed"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    bench = data.get("bench") or {}
    ref = bench.get("reference_p4a_two_stage") or {}
    cand = bench.get("candidate_p4b_single_cta") or {}
    diff = data.get("numeric_vs_reference") or {}
    byte_budget = data.get("byte_budget") or {}
    lines = [
        "# Stage 6 P4B Single-CTA Microbench Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Torch/CUDA | `{data.get('torch_version')}` / `{data.get('torch_cuda')}` |",
        f"| Banked microbench | `{data.get('banked_single_cta_microbench')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| P4A two-stage median | `{ref.get('median_us')}` us |",
        f"| P4B single-CTA median | `{cand.get('median_us')}` us |",
        f"| P4B/P4A speedup | `{bench.get('candidate_vs_reference_speedup')}` |",
        f"| P4B minus P4A | `{bench.get('candidate_minus_reference_us')}` us |",
        f"| Numeric rel L2 | `{diff.get('rel_l2')}` |",
        f"| Numeric max abs | `{diff.get('max_abs')}` |",
        f"| No inter_scratch candidate ABI | `{byte_budget.get('no_inter_scratch_candidate_abi')}` |",
        f"| Packed/BF16 ratio | `{byte_budget.get('packed_vs_bf16_shadow_ratio')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
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
