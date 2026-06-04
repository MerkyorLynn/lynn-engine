#!/usr/bin/env python3
"""Summarize Stage 6 P4C active-reuse runtime bridge preflight artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_stage6_p4_runtime_bridge_preflight import _verdict as _p4_verdict  # noqa: E402
from scripts.summarize_stage6_p4_runtime_bridge_preflight import summarize as _p4_summarize  # noqa: E402


SCHEMA = "lynn-stage6-p4c-active-reuse-runtime-bridge-preflight-v1"
EXPECTED_BACKEND = "fused_zero_shadow_active_reuse_contract"
PASS_DECISION = "PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE"


def _compat_for_p4_verdict(data: dict) -> dict:
    compat = dict(data)
    compat["schema"] = "lynn-stage6-p4-runtime-bridge-preflight-v1"
    compat["decision"] = "PASS_TWO_STAGE_RUNTIME_BRIDGE" if data.get("decision") == PASS_DECISION else data.get("decision")
    compat["expected_backend"] = "fused_zero_shadow_out_contract"
    compat["banked_runtime_bridge_preflight"] = bool(data.get("banked_p4c_active_reuse_runtime_bridge"))
    return compat


def _verdict(data: dict) -> tuple[str, str]:
    if data.get("schema") != SCHEMA:
        return "FAIL", "schema mismatch"
    if data.get("expected_backend") != EXPECTED_BACKEND:
        return "FAIL", "expected backend mismatch"
    if data.get("decision") != PASS_DECISION:
        return "FAIL", "top-level decision is not PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE"
    if data.get("banked_p4c_active_reuse_runtime_bridge") is not True:
        return "FAIL", "P4C active-reuse runtime bridge was not banked"
    native = data.get("native_backend_call_count") or {}
    if native.get("key") != "_p4c_fused_zero_shadow_active_reuse_contract_call_count":
        return "FAIL", "P4C native call counter key mismatch"
    last_shapes = native.get("last_shapes") or {}
    if "inter_scratch" not in last_shapes:
        return "FAIL", "P4C last_shapes must prove active scratch reuse"
    verdict, reason = _p4_verdict(_compat_for_p4_verdict(data))
    if verdict != "PASS":
        return verdict, reason
    return "PASS", "P4C active-reuse runtime bridge returns caller-owned two-phase output"


def summarize(data: dict) -> str:
    verdict, reason = _verdict(data)
    compat = _compat_for_p4_verdict(data)
    md = _p4_summarize(compat)
    md = md.replace("# Stage 6 P4 Runtime Bridge Preflight Summary", "# Stage 6 P4C Active-Reuse Runtime Bridge Preflight Summary", 1)
    md = md.replace("PASS_TWO_STAGE_RUNTIME_BRIDGE", str(data.get("decision")), 1)
    md = md.replace("fused_zero_shadow_out_contract", EXPECTED_BACKEND)
    md = md.replace("runtime bridge returns two-stage P4 output on real runner path", reason)
    md = md.replace("caller-owned two-stage packed-NVFP4 active-MoE reference", "P4C caller-owned active-reuse two-phase packed-NVFP4 active-MoE contract")
    md = md.replace("Verdict | **PASS**", f"Verdict | **{verdict}**", 1)
    if verdict != "PASS":
        md = md.replace("Verdict | **FAIL**", f"Verdict | **{verdict}**", 1)
        md = md.replace("(runtime bridge preflight was not banked)", f"({reason})", 1)
    lines = md.rstrip().splitlines()
    lines.extend([
        "",
        "## P4C Boundary",
        "",
        "- This banks only `banked_p4c_active_reuse_runtime_bridge=true`.",
        "- It keeps `banked_fused_kernel=false` and `banked_default_promotion=false`.",
        "- It is **P4C**, not P4B: caller-owned `active[top_k,512]` scratch is allowed and must be reported.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="Path to P4C runtime bridge result.json")
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
