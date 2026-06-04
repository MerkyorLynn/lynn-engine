#!/usr/bin/env python3
"""Stage 6 R5-C4 trace-derived candidate rejection probe.

This intentionally builds a bad R5-C4 candidate from prior R5-C3 gate/up timing
and host-composition parity artifacts. The R5-C4 validator must reject it
because it is not same-scope, does not use real model weights/router outputs,
and does not time the full active-MoE boundary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


DEFAULT_R5C3A = "reports/stage6/r5c3a_gateup_prefill_timing_smoke_20260604_203052/result.json"
DEFAULT_R5C3C = "reports/stage6/r5c3c_down_weighted_parity_smoke_20260604_130243/result.json"
VALIDATOR = "scripts/r6000_stage6_r5c4_full_active_moe_speed_ab.py"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_gateup_ms(r5c3a: dict[str, Any]) -> float:
    run_parse = r5c3a.get("run_parse") if isinstance(r5c3a.get("run_parse"), dict) else {}
    runtimes = [float(v) for v in run_parse.get("avg_runtime_ms", [])]
    if not runtimes:
        raise AssertionError("R5-C3A artifact has no avg_runtime_ms")
    return min(runtimes)


def _write_bad_candidate(r5c3a: dict[str, Any], r5c3c: dict[str, Any], out: Path) -> dict[str, Any]:
    gateup_ms = _best_gateup_ms(r5c3a)
    shape3a = r5c3a.get("selected_expert_shape") if isinstance(r5c3a.get("selected_expert_shape"), dict) else {}
    shape3c = r5c3c.get("selected_expert_shape") if isinstance(r5c3c.get("selected_expert_shape"), dict) else {}
    # Deliberately make the speed look good. The validator must still reject it
    # due to boundary/scope booleans being false.
    lane = {
        "candidate_ms": gateup_ms,
        "baseline_w4a16_ms": gateup_ms * 2.0,
        "baseline_packed_p3_ms": gateup_ms * 1.5,
        "numeric_max_abs": 0.0,
        "numeric_rel_l2": 0.0,
        "numeric_cosine": 1.0,
        "route_order_preserved": True,
        "repack_cost_reported": False,
        "fault_injections_detected": True,
        "shape_regression": False,
        "shape": shape3a,
    }
    candidate = {
        "schema": "lynn-stage6-r5c4-trace-derived-bad-candidate-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "r5c3a_decision": r5c3a.get("decision"),
            "r5c3c_decision": r5c3c.get("decision"),
            "r5c3a_shape": shape3a,
            "r5c3c_shape": shape3c,
        },
        "passes": {
            "same_scope_ab": False,
            "real_model_weights": False,
            "real_router_outputs": False,
            "candidate_no_active_bf16_shadow": True,
            "candidate_no_reload": True,
            "candidate_no_bf16_weight_materialization": True,
            "candidate_full_active_moe_boundary_timed": False,
            "timing_includes_gateup_swiglu_down_weighted_scatter": False,
        },
        "lanes": {
            "smoke": dict(lane),
            "production_shape": dict(lane),
        },
        "expected_validator_decision": "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
        "why_bad": [
            "Derived from R5-C3A gate/up-only timing, not full active-MoE timing.",
            "R5-C3C is host composition parity, not a timed R6000 full active-MoE kernel.",
            "No real model weights or real router outputs are used.",
            "No down/scatter timing is included.",
        ],
    }
    out.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return candidate


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = Path(args.candidate_out).expanduser().resolve() if args.candidate_out else out.with_suffix(".candidate.json")
    r5c3a_path = root / args.r5c3a
    r5c3c_path = root / args.r5c3c
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = _write_bad_candidate(_load(r5c3a_path), _load(r5c3c_path), candidate_path)
    validator_out = out.with_suffix(".validator.json")
    proc = subprocess.run(
        [
            sys.executable,
            str(root / VALIDATOR),
            "--input-r5c3c",
            str(r5c3c_path),
            "--candidate-json",
            str(candidate_path),
            "--out",
            str(validator_out),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    validator_data = _load(validator_out)
    passes = {
        "candidate_written": candidate_path.exists(),
        "validator_result_written": validator_out.exists(),
        "validator_rejected_trace_candidate": validator_data.get("decision") == candidate["expected_validator_decision"],
        "validator_returned_failure": proc.returncode != 0,
        "same_scope_false": candidate.get("passes", {}).get("same_scope_ab") is False,
        "real_model_weights_false": candidate.get("passes", {}).get("real_model_weights") is False,
        "full_boundary_timed_false": candidate.get("passes", {}).get("candidate_full_active_moe_boundary_timed") is False,
        "decode_tps_not_banked": validator_data.get("passes", {}).get("banked_decode_tps") is False,
        "default_not_banked": validator_data.get("passes", {}).get("banked_default_promotion") is False,
    }
    passes["all"] = all(passes.values())
    result = {
        "schema": "lynn-stage6-r5c4-trace-candidate-rejection-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": "PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED" if passes["all"] else "FAIL_R5C4_TRACE_DERIVED_CANDIDATE_REJECTION",
        "r5c3a": str(r5c3a_path),
        "r5c3c": str(r5c3c_path),
        "candidate_json": str(candidate_path),
        "validator_json": str(validator_out),
        "validator_stdout": proc.stdout[-2000:],
        "validator_stderr": proc.stderr[-2000:],
        "validator_returncode": proc.returncode,
        "validator_decision": validator_data.get("decision"),
        "passes": passes,
        "promotion_boundary": {
            "full_active_moe_prefill_speed": False,
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "decode_tps": False,
            "server_rc": False,
            "default_runtime": False,
            "full_transformer_prefill": False,
        },
        "caveats": [
            "This banks only validator rejection behavior for a trace-derived bad candidate.",
            "It does not measure or bank R5-C4 full active-MoE speed.",
        ],
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--r5c3a", default=DEFAULT_R5C3A)
    ap.add_argument("--r5c3c", default=DEFAULT_R5C3C)
    ap.add_argument("--candidate-out", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = run(args)
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
