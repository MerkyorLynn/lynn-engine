#!/usr/bin/env python3
"""Summarize Stage 6 P4 real-runtime bridge preflight artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA = "lynn-stage6-p4-runtime-bridge-preflight-v1"


def _verdict(data: dict[str, Any]) -> tuple[str, str]:
    passes = data.get("passes") or {}
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("banked_fused_kernel") is not False:
        return "FAIL", "fused-kernel promotion boundary violated"
    if data.get("banked_default_promotion") is not False:
        return "FAIL", "default promotion boundary violated"
    if data.get("banked_runtime_bridge_preflight") is not True:
        return "FAIL", "runtime bridge preflight was not banked"
    for gate in (
        "baseline_triton_nonzero",
        "packed_tensors_present",
        "active_shadows_removed",
        "candidate_fail_loud",
        "fused_kernel_unbanked",
        "default_promotion_closed",
        "all",
    ):
        if passes.get(gate) is not True:
            return "FAIL", f"{gate} gate fail"
    if data.get("decision") != "PASS_RUNTIME_BRIDGE_CONTRACT":
        return "FAIL", "top-level decision is not PASS_RUNTIME_BRIDGE_CONTRACT"
    return "PASS", "runtime bridge reaches P4 fail-loud on real runner path"


def summarize(data: dict[str, Any]) -> str:
    verdict, reason = _verdict(data)
    passes = data.get("passes") or {}
    baseline = data.get("baseline") or {}
    removed = data.get("removed_active_shadows") or {}
    packed = data.get("packed_manifest_before_candidate") or {}
    candidate_error = data.get("candidate_error") or data.get("runner_error") or {}
    lines = [
        "# Stage 6 P4 Runtime Bridge Preflight Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{verdict}** ({reason}) |",
        f"| Decision | `{data.get('decision')}` |",
        f"| Model | `{data.get('model', 'unknown')}` |",
        f"| Layer | `{data.get('layer')}` |",
        f"| Expected backend | `{data.get('expected_backend')}` |",
        f"| Banked runtime bridge preflight | `{data.get('banked_runtime_bridge_preflight')}` |",
        f"| Banked fused kernel | `{data.get('banked_fused_kernel')}` |",
        f"| Banked default promotion | `{data.get('banked_default_promotion')}` |",
        f"| Baseline norm | `{baseline.get('norm')}` |",
        f"| Packed tensors present | `{passes.get('packed_tensors_present')}` |",
        f"| Active shadows removed | `{passes.get('active_shadows_removed')}` |",
        f"| Candidate fail-loud | `{passes.get('candidate_fail_loud')}` |",
        f"| Elapsed seconds | `{data.get('elapsed_s')}` |",
        "",
        "## Removed Active Shadows",
        "",
        "| Key | Shape | DType | Bytes |",
        "|---|---|---|---:|",
    ]
    for key, meta in removed.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    lines.extend(["", "## Packed Tensor Inputs", "", "| Key | Shape | DType | Bytes |", "|---|---|---|---:|"])
    for key, meta in packed.items():
        lines.append(f"| `{key}` | `{meta.get('shape')}` | `{meta.get('dtype')}` | `{meta.get('bytes')}` |")
    if candidate_error:
        lines.extend(["", "## Error Tail", "", "```text", str(candidate_error.get("message", candidate_error))[-1200:], "```"])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P4 runtime bridge result.json")
    ap.add_argument("--markdown-out", default="", help="Optional Markdown output path")
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
