#!/usr/bin/env python3
"""Summarize Stage 6 P4B single-CTA numeric preflight artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4b-single-cta-numeric-preflight-v1"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    byte_budget = data.get("byte_budget") or {}
    candidate = data.get("candidate_output") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel speed boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_single_cta_numeric_preflight") is not True:
        return "FAIL", "single-CTA numeric preflight was not banked"
    if data.get("decision") != "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE":
        return "FAIL", "top-level decision is not PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE"
    for gate in (
        "extension_loaded",
        "symbol_present",
        "reference_symbol_present",
        "reference_output_returned",
        "candidate_output_returned",
        "reference_finite",
        "candidate_finite",
        "numeric_vs_reference",
        "zero_shadow_candidate_abi",
        "packed_byte_budget",
        "no_inter_scratch_candidate_abi",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if byte_budget.get("no_inter_scratch_candidate_abi") is not True:
        return "FAIL", "candidate ABI admits inter_scratch"
    if candidate.get("shape") != [1, 2048] or candidate.get("dtype") != "torch.bfloat16":
        return "FAIL", "candidate shape/dtype mismatch"
    return "PASS", "P4B single-CTA output matches P4A two-stage reference; speed still unbanked"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    ref = data.get("reference_output") or {}
    cand = data.get("candidate_output") or {}
    byte_budget = data.get("byte_budget") or {}
    lines = [
        "# Stage 6 P4B Single-CTA Numeric Preflight Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Symbol | `{data.get('symbol')}` |",
        f"| Reference symbol | `{data.get('reference_symbol')}` |",
        f"| Device | `{data.get('device_name', 'unknown')}` |",
        f"| Capability | `{data.get('capability', 'unknown')}` |",
        f"| Torch/CUDA | `{data.get('torch_version')}` / `{data.get('torch_cuda')}` |",
        f"| Build dir | `{data.get('build_dir', 'unknown')}` |",
        f"| Banked single-CTA numeric preflight | `{data.get('banked_single_cta_numeric_preflight')}` |",
        f"| Banked fused kernel speed | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Reference finite | `{passes.get('reference_finite')}` |",
        f"| Candidate finite | `{passes.get('candidate_finite')}` |",
        f"| Numeric vs reference | `{passes.get('numeric_vs_reference')}` |",
        f"| Candidate rel L2 | `{cand.get('rel_l2_vs_reference')}` |",
        f"| Candidate max abs diff | `{cand.get('max_abs_diff_vs_reference')}` |",
        f"| Reference norm | `{ref.get('norm')}` |",
        f"| Candidate norm | `{cand.get('norm')}` |",
        f"| Zero-shadow candidate ABI | `{passes.get('zero_shadow_candidate_abi')}` |",
        f"| Packed byte budget | `{passes.get('packed_byte_budget')}` |",
        f"| No inter_scratch candidate ABI | `{passes.get('no_inter_scratch_candidate_abi')}` |",
        f"| Packed/BF16 ratio | `{byte_budget.get('packed_vs_bf16_shadow_ratio')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
    ]
    error = data.get("candidate_error") or data.get("reference_error") or data.get("load_error_tail")
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
