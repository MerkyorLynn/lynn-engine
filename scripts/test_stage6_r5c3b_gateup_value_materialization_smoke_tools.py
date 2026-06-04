#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C3B value-materialization tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c3b_gateup_value_materialization_smoke.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c3b_gateup_value_materialization_smoke.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c3b_gateup_value_materialization_smoke.sh"
CONTRACT = ROOT / "reports" / "stage6" / "R5C3B_GATEUP_VALUE_MATERIALIZATION_CONTRACT_20260604.md"


def _fixture(**overrides: bool) -> dict:
    passes = {
        "banked_gateup_value_materialization": True,
        "banked_host_swiglu_checksum_smoke": True,
        "banked_down_projection_numeric_parity": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
        "all": True,
    }
    passes.update(overrides)
    return {
        "schema": "lynn-stage6-r5c3b-gateup-value-materialization-smoke-v1",
        "decision": "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
        "selected_expert_shape": {
            "tokens": 128,
            "top_k": 2,
            "experts": 4,
            "tokens_per_expert": [32, 64, 64, 96],
            "n_gate_up": 128,
            "k_hidden": 256,
        },
        "cutlass_run": {"patch": {"applied": True, "restored": True}},
        "run_parse": {"avg_runtime_ms": [0.03, 0.04]},
        "value_schedules": {
            "cooperative": {
                "records": 256,
                "row_counts": [32, 64, 64, 96],
                "scatter_values_max_abs": 0.0,
                "host_swiglu_checksum": 12.0,
                "passes": {
                    "value_digest_matches_r5c2c_digest": True,
                    "fault_injections_detected": True,
                },
            }
        },
        "passes": passes,
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c3b-gateup-value-materialization-smoke-v1",
            "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
            "LYNN_STAGE6_R5C3B_D_ROW_VALUE_PATCH",
            "lynn_d_row_value",
            "full real CUTLASS D/ref row values",
            "full_d_row_value_bits_captured",
            "banked_gateup_value_materialization",
            "banked_down_projection_numeric_parity",
            "value_digest_matches_r5c2c_digest",
            "host_swiglu_checksum",
        ],
        SUMMARY: [
            "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
            "Gate/up value materialization banked",
            "Down projection numeric parity banked",
            "Grouped-MoE FP4-MMA POC banked",
            "strict-exit",
        ],
        WRAPPER: [
            "r6000_stage6_r5c3b_gateup_value_materialization_smoke.py",
            "summarize_stage6_r5c3b_gateup_value_materialization_smoke.py",
            "--build",
            "nvidia_smi_before.txt",
        ],
        CONTRACT: [
            "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
            "banked_gateup_value_materialization=true",
            "banked_down_projection_numeric_parity=false",
        ],
    }
    for path, needles in checks.items():
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)}: missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        good = tmp / "good.json"
        bad_down = tmp / "bad_down.json"
        bad_speed = tmp / "bad_speed.json"
        md = tmp / "summary.md"
        good.write_text(json.dumps(_fixture()), encoding="utf-8")
        bad_down.write_text(json.dumps(_fixture(banked_down_projection_numeric_parity=True)), encoding="utf-8")
        bad_speed.write_text(json.dumps(_fixture(banked_kernel_speed=True)), encoding="utf-8")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(good), "--markdown-out", str(md), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"summary strict good fixture failed: {ok.stderr or ok.stdout}")
        if "Gate/up value materialization banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing gate/up value materialization row")
        for bad in [bad_down, bad_speed]:
            fail = subprocess.run(
                [sys.executable, str(SUMMARY), str(bad), "--strict-exit"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if fail.returncode == 0:
                failures.append(f"summary strict bad fixture unexpectedly passed: {bad.name}")

    if failures:
        print("Stage 6 R5-C3B gate/up value-materialization tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C3B gate/up value-materialization tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
