#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C active-reuse microbench tooling."""
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
        "schema": "lynn-stage6-p4c-active-reuse-microbench-v1",
        "decision": "PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "torch_version": "fixture",
        "torch_cuda": "fixture",
        "banked_p4c_active_reuse_speed_baseline": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "speed_ratio_floor": 0.8,
        "bench": {
            "reference_p4a_two_stage": {"median_us": 10.0},
            "candidate_p4c_active_reuse_contract": {"median_us": 10.1},
            "candidate_vs_reference_speedup": 0.9901,
            "candidate_minus_reference_us": 0.1,
        },
        "numeric_vs_reference": {
            "out": {"rel_l2": 0.0, "max_abs": 0.0},
            "inter_scratch": {"rel_l2": 0.0, "max_abs": 0.0},
        },
        "byte_budget": {
            "active_scratch_bytes": 8192,
            "zero_bf16_shadow_weight_abi": True,
            "active_scratch_reuse_abi": True,
            "packed_byte_budget": True,
            "packed_vs_bf16_shadow_ratio": 0.375,
        },
        "passes": {
            "all": True,
            "numeric_vs_reference": True,
            "timing_recorded": True,
            "speed_floor_recorded": True,
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


def speed_floor_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_p4c_active_reuse_speed_baseline"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["speed_floor_recorded"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["candidate_vs_reference_speedup"] = 0.5
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4c_active_reuse_microbench.sh"
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "promotion_fail": promotion_fail_fixture(),
            "speed_floor_fail": speed_floor_fail_fixture(),
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
            "scripts/spark_stage6_p4c_active_reuse_microbench.py",
            "scripts/summarize_stage6_p4c_active_reuse_microbench.py",
        ])
        run(["bash", "-n", str(wrapper)])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "--speed-ratio-floor N" in wrapper_help.stdout
        assert "p4c_active_reuse_microbench_" in wrapper_text
        assert "spark_stage6_p4c_active_reuse_microbench.py" in wrapper_text
        assert "summarize_stage6_p4c_active_reuse_microbench.py" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_active_reuse_microbench.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked P4C speed baseline | `True`" in pass_summary.stdout
        assert "Banked fused kernel speed | `False`" in pass_summary.stdout
        assert "P4C/P4A speedup | `0.9901`" in pass_summary.stdout
        assert "active-reuse speed baseline" in pass_summary.stdout

        for name in ("promotion_fail", "speed_floor_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_active_reuse_microbench.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4C active-reuse microbench tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
