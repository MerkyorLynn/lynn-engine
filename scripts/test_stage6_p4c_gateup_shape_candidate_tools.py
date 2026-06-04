#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C gate/up shape-candidate tooling."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, expect: int | None = 0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if expect is not None and proc.returncode != expect:
        raise AssertionError(
            f"command failed with {proc.returncode}, expected {expect}: {cmd}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def pass_fixture() -> dict:
    return {
        "schema": "lynn-stage6-p4c-gateup-shape-candidate-microbench-v1",
        "decision": "PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "current_tile_inter": 8,
        "candidate_tile_inter": 2,
        "candidate_speedup_floor": 1.05,
        "banked_p4c_gateup_shape_candidate": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "bench": {
            "reference_p4a_current_tile": {"median_us": 100.0},
            "reference_p4a_candidate_tile": {"median_us": 70.0},
            "current_p4c_active_reuse_contract": {"median_us": 100.0},
            "candidate_p4c_active_reuse_contract": {"median_us": 70.0},
            "candidate_vs_current_speedup": 1.4285714285714286,
            "candidate_minus_current_us": -30.0,
            "reference_candidate_vs_current_speedup": 1.4285714285714286,
        },
        "numeric_vs_reference": {
            "candidate_vs_p4a_candidate_tile_out": {"rel_l2": 0.0, "max_abs": 0.0},
            "candidate_vs_p4a_candidate_tile_inter_scratch": {"rel_l2": 0.0, "max_abs": 0.0},
            "p4a_candidate_tile_vs_current_tile_out": {"rel_l2": 1e-6, "max_abs": 1e-7},
            "p4a_candidate_tile_vs_current_tile_inter_scratch": {"rel_l2": 1e-6, "max_abs": 1e-7},
        },
        "passes": {
            "all": True,
            "numeric_vs_reference": True,
            "timing_recorded": True,
            "candidate_speed_floor": True,
            "zero_bf16_shadow_weight_abi": True,
            "active_scratch_reuse_abi": True,
            "packed_byte_budget": True,
            "promotion_boundary_closed": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def speed_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_p4c_gateup_shape_candidate"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["candidate_speed_floor"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["candidate_vs_current_speedup"] = 0.99
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4c_gateup_shape_candidate_microbench.sh"
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "promotion_fail": promotion_fail_fixture(),
            "speed_fail": speed_fail_fixture(),
        }
        paths = {}
        for name, data in fixtures.items():
            path = tmp / f"{name}.json"
            write_json(path, data)
            paths[name] = path

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p4c_gateup_shape_candidate_microbench.py",
            "scripts/summarize_stage6_p4c_gateup_shape_candidate_microbench.py",
        ])
        run(["bash", "-n", str(wrapper)])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "p4c_gateup_shape_candidate_" in wrapper_text
        assert "spark_stage6_p4c_gateup_shape_candidate_microbench.py" in wrapper_text
        assert "summarize_stage6_p4c_gateup_shape_candidate_microbench.py" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_gateup_shape_candidate_microbench.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Candidate speedup vs current | `1.4285714285714286`" in pass_summary.stdout
        assert "Banked P4C gate/up shape candidate | `True`" in pass_summary.stdout
        assert "does not bank fused-kernel speed" in pass_summary.stdout

        for name in ("promotion_fail", "speed_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_gateup_shape_candidate_microbench.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4C gate/up shape-candidate tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
