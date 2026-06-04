#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C component-profile tooling."""
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
        "schema": "lynn-stage6-p4c-component-profile-v1",
        "decision": "PASS_P4C_COMPONENT_PROFILE_RECORDED",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "banked_p4c_component_profile": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "component_timing_caveat": "component symbols allocate output tensors",
        "bench": {
            "full_p4c_active_reuse_contract": {"median_us": 100.0},
            "component_gate_up_allocating": {"median_us": 70.0},
            "component_down_allocating": {"median_us": 30.0},
            "component_sum_us": 100.0,
            "gate_share_of_component_sum": 0.7,
            "down_share_of_component_sum": 0.3,
            "component_sum_vs_full_ratio": 1.0,
        },
        "numeric_vs_reference": {
            "gate_inter_scratch": {"rel_l2": 0.0, "max_abs": 0.0},
            "down_on_ref_scratch": {"rel_l2": 0.0, "max_abs": 0.0},
            "gate_plus_down_composed": {"rel_l2": 0.0, "max_abs": 0.0},
        },
        "passes": {
            "all": True,
            "numeric_vs_reference": True,
            "timing_recorded": True,
            "promotion_boundary_closed": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def timing_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_p4c_component_profile"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["timing_recorded"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["component_down_allocating"] = {"median_us": 0.0}
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4c_component_profile.sh"
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "promotion_fail": promotion_fail_fixture(),
            "timing_fail": timing_fail_fixture(),
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
            "scripts/spark_stage6_p4c_component_profile.py",
            "scripts/summarize_stage6_p4c_component_profile.py",
        ])
        run(["bash", "-n", str(wrapper)])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "p4c_component_profile_" in wrapper_text
        assert "spark_stage6_p4c_component_profile.py" in wrapper_text
        assert "summarize_stage6_p4c_component_profile.py" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_component_profile.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked component profile | `True`" in pass_summary.stdout
        assert "Gate share | `0.7`" in pass_summary.stdout
        assert "Down share | `0.3`" in pass_summary.stdout
        assert "diagnostic only" in pass_summary.stdout

        for name in ("promotion_fail", "timing_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_component_profile.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4C component-profile tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
