#!/usr/bin/env python3
"""Stage 6 P4C active-reuse runtime bridge preflight.

P4C is the two-phase active-reuse candidate boundary. It intentionally allows
caller-owned ``active[top_k,512]`` scratch, so it must be named and banked
separately from P4B's harder out-only single-kernel objective.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.spark_stage6_p4_runtime_bridge_preflight as p4_bridge  # noqa: E402


EXPECTED_BACKEND = "fused_zero_shadow_active_reuse_contract"
NATIVE_CALL_COUNT_KEY = "_p4c_fused_zero_shadow_active_reuse_contract_call_count"
NATIVE_LAST_SHAPES_KEY = "_p4c_fused_zero_shadow_active_reuse_contract_last_shapes"
SCHEMA = "lynn-stage6-p4c-active-reuse-runtime-bridge-preflight-v1"
PASS_DECISION = "PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE"
FAIL_DECISION = "FAIL_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE"


def run_preflight(args: argparse.Namespace) -> dict:
    old_expected_backend = p4_bridge.EXPECTED_BACKEND
    old_call_key = p4_bridge.NATIVE_CALL_COUNT_KEY
    old_shape_key = p4_bridge.NATIVE_LAST_SHAPES_KEY
    p4_bridge.EXPECTED_BACKEND = EXPECTED_BACKEND
    p4_bridge.NATIVE_CALL_COUNT_KEY = NATIVE_CALL_COUNT_KEY
    p4_bridge.NATIVE_LAST_SHAPES_KEY = NATIVE_LAST_SHAPES_KEY
    try:
        result = p4_bridge.run_preflight(args)
    finally:
        p4_bridge.EXPECTED_BACKEND = old_expected_backend
        p4_bridge.NATIVE_CALL_COUNT_KEY = old_call_key
        p4_bridge.NATIVE_LAST_SHAPES_KEY = old_shape_key

    result["schema"] = SCHEMA
    result["expected_backend"] = EXPECTED_BACKEND
    result["expected_reference"] = "P4C caller-owned active-reuse two-phase packed-NVFP4 active-MoE contract"
    result["banked_p4c_active_reuse_runtime_bridge"] = bool(result.get("passes", {}).get("all"))
    result["banked_fused_kernel"] = False
    result["banked_default_promotion"] = False
    result["decision"] = PASS_DECISION if result["banked_p4c_active_reuse_runtime_bridge"] else FAIL_DECISION
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="P4C active-reuse runtime bridge preflight.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="Explain MoE active parameters in one sentence.")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--rel-l2-threshold", type=float, default=0.02)
    ap.add_argument("--max-abs-threshold", type=float, default=1.0)
    ap.add_argument("--strict-exit", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    started = time.time()
    result = run_preflight(args)
    result["elapsed_s"] = round(time.time() - started, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[p4c-runtime-bridge] decision={result.get('decision')}")
    print(f"[p4c-runtime-bridge] out={out}")
    if result.get("passes", {}).get("all"):
        return 0
    return 2 if args.strict_exit else 0


if __name__ == "__main__":
    raise SystemExit(main())
