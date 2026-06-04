#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C gate/up shape-sweep tooling."""
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
        "schema": "lynn-stage6-p4c-gateup-shape-sweep-v1",
        "decision": "PASS_P4C_GATEUP_SHAPE_SWEEP_RECORDED",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "baseline_symbol": "gate_up_silu_tile_inter_scalar",
        "variant_symbol": "gate_up_silu_tile_inter_threads_scalar",
        "baseline_tile_inter": 8,
        "actionable_speedup_floor": 1.05,
        "banked_p4c_gateup_shape_sweep": True,
        "banked_p4c_gateup_candidate": False,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "component_timing_caveat": "gate/up symbols allocate output tensors",
        "bench": {
            "current_baseline_gate_up": {"median_us": 100.0},
            "best_speedup_vs_current": 1.11,
            "best_is_actionable": True,
            "best_variant": {
                "key": "tile_inter_8_threads_256",
                "median_us": 90.0,
                "speedup_vs_current": 1.11,
                "numeric_ok": True,
                "diff": {"rel_l2": 0.0, "max_abs": 0.0},
            },
            "variants": [
                {
                    "key": "tile_inter_1_threads_64",
                    "tile_inter": 1,
                    "threads": 64,
                    "median_us": 120.0,
                    "speedup_vs_current": 0.8333333333333334,
                    "numeric_ok": True,
                    "diff": {"rel_l2": 0.0, "max_abs": 0.0},
                },
                {
                    "key": "tile_inter_8_threads_256",
                    "tile_inter": 8,
                    "threads": 256,
                    "median_us": 90.0,
                    "speedup_vs_current": 1.11,
                    "numeric_ok": True,
                    "diff": {"rel_l2": 0.0, "max_abs": 0.0},
                },
            ],
        },
        "numeric_vs_reference": {
            "current_baseline_gate_up": {"rel_l2": 0.0, "max_abs": 0.0},
            "variants_all_numeric_ok": True,
        },
        "passes": {
            "all": True,
            "baseline_numeric_vs_reference": True,
            "variants_numeric_vs_reference": True,
            "timing_recorded": True,
            "promotion_boundary_closed": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_default_promotion"] = True
    return data


def numeric_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_p4c_gateup_shape_sweep"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["variants_numeric_vs_reference"] = False
    data["passes"]["all"] = False
    data["numeric_vs_reference"] = dict(data["numeric_vs_reference"])
    data["numeric_vs_reference"]["variants_all_numeric_ok"] = False
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4c_gateup_shape_sweep.sh"
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "promotion_fail": promotion_fail_fixture(),
            "numeric_fail": numeric_fail_fixture(),
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
            "scripts/spark_stage6_p4c_gateup_shape_sweep.py",
            "scripts/summarize_stage6_p4c_gateup_shape_sweep.py",
        ])
        run(["bash", "-n", str(wrapper)])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "p4c_gateup_shape_sweep_" in wrapper_text
        assert "spark_stage6_p4c_gateup_shape_sweep.py" in wrapper_text
        assert "summarize_stage6_p4c_gateup_shape_sweep.py" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_gateup_shape_sweep.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked shape sweep | `True`" in pass_summary.stdout
        assert "Best shape | `tile_inter_8_threads_256`" in pass_summary.stdout
        assert "Best actionable >= floor | `True`" in pass_summary.stdout
        assert "does not bank a gate/up speed candidate" in pass_summary.stdout

        for name in ("promotion_fail", "numeric_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_gateup_shape_sweep.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4C gate/up shape-sweep tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
