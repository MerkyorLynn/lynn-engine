#!/usr/bin/env python3
"""GPU-free checks for the R5-C4 speed A/B validator."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "r6000_stage6_r5c4_full_active_moe_speed_ab.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c4_full_active_moe_speed_ab.sh"
INPUT_R5C3C = ROOT / "reports" / "stage6" / "r5c3c_down_weighted_parity_smoke_20260604_130243" / "result.json"


def _candidate(speed: float = 1.2, *, numeric_bad: bool = False) -> dict:
    candidate_ms = 10.0
    best = candidate_ms * speed
    lane = {
        "candidate_ms": candidate_ms,
        "baseline_w4a16_ms": candidate_ms * 1.2,
        "baseline_packed_p3_ms": best,
        "numeric_max_abs": 0.0 if not numeric_bad else 1.0,
        "numeric_rel_l2": 0.0 if not numeric_bad else 1.0,
        "numeric_cosine": 1.0 if not numeric_bad else 0.1,
        "route_order_preserved": True,
        "repack_cost_reported": True,
        "fault_injections_detected": True,
        "shape_regression": False,
    }
    return {
        "schema": "lynn-stage6-r5c4-candidate-v1",
        "passes": {
            "same_scope_ab": True,
            "real_model_weights": True,
            "real_router_outputs": True,
            "candidate_no_active_bf16_shadow": True,
            "candidate_no_reload": True,
            "candidate_no_bf16_weight_materialization": True,
            "candidate_full_active_moe_boundary_timed": True,
            "timing_includes_gateup_swiglu_down_weighted_scatter": True,
        },
        "lanes": {"smoke": dict(lane), "production_shape": dict(lane)},
    }


def main() -> int:
    failures: list[str] = []
    for path in [VALIDATOR, WRAPPER, INPUT_R5C3C]:
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
    if VALIDATOR.exists():
        text = VALIDATOR.read_text(encoding="utf-8")
        for needle in [
            "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
            "DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED",
            "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
            "candidate_no_bf16_weight_materialization",
            "timing_includes_gateup_swiglu_down_weighted_scatter",
            "banked_decode_tps",
            "active_moe_prefill_only",
        ]:
            if needle not in text:
                failures.append(f"{VALIDATOR.relative_to(ROOT)} missing {needle!r}")
    if WRAPPER.exists():
        text = WRAPPER.read_text(encoding="utf-8")
        for needle in [
            "CANDIDATE_JSON is required",
            "r6000_stage6_r5c4_full_active_moe_speed_ab.py",
            "summarize_stage6_r5c4_full_active_moe_speed_ab.py",
            "nvidia_smi_before.txt",
        ]:
            if needle not in text:
                failures.append(f"{WRAPPER.relative_to(ROOT)} missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        for name, candidate, expected in [
            ("pass", _candidate(1.2), "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"),
            ("diag", _candidate(1.01), "DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED"),
            ("fail", _candidate(1.2, numeric_bad=True), "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB"),
        ]:
            candidate_path = tmp / f"{name}.candidate.json"
            out_path = tmp / f"{name}.result.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--input-r5c3c",
                    str(INPUT_R5C3C),
                    "--candidate-json",
                    str(candidate_path),
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if data.get("decision") != expected:
                failures.append(f"{name} decision {data.get('decision')} != {expected}")
            if expected == "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB" and proc.returncode == 0:
                failures.append("fail fixture returned success")
            if expected != "FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB" and proc.returncode != 0:
                failures.append(f"{name} fixture returned {proc.returncode}: {proc.stderr or proc.stdout}")
            if any(data.get("passes", {}).get(key) is True for key in ["banked_decode_tps", "banked_server_rc", "banked_default_promotion"]):
                failures.append(f"{name} fixture promoted forbidden boundary")

    if failures:
        print("Stage 6 R5-C4 full active-MoE speed A/B validator self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C4 full active-MoE speed A/B validator self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
