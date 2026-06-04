#!/usr/bin/env python3
"""Local self-test for Stage 6 P4B single-CTA microbench tooling."""
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
        "schema": "lynn-stage6-p4b-single-cta-microbench-v1",
        "decision": "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "torch_version": "fixture",
        "torch_cuda": "fixture",
        "banked_single_cta_microbench": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "bench": {
            "reference_p4a_two_stage": {"median_us": 10.0},
            "candidate_p4b_single_cta": {"median_us": 50.0},
            "candidate_vs_reference_speedup": 0.2,
            "candidate_minus_reference_us": 40.0,
        },
        "numeric_vs_reference": {"rel_l2": 0.0, "max_abs": 0.0},
        "byte_budget": {
            "no_inter_scratch_candidate_abi": True,
            "packed_vs_bf16_shadow_ratio": 0.375,
        },
        "passes": {
            "all": True,
            "numeric_vs_reference": True,
            "timing_recorded": True,
            "no_inter_scratch_candidate_abi": True,
            "promotion_boundary_closed": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def timing_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_single_cta_microbench"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["timing_recorded"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["candidate_p4b_single_cta"] = {"median_us": 0.0}
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4b_single_cta_microbench.sh"
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
            "scripts/spark_stage6_p4b_single_cta_microbench.py",
            "scripts/summarize_stage6_p4b_single_cta_microbench.py",
        ])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "--candidate-mode MODE" in wrapper_help.stdout
        assert "p4b_single_cta_microbench_" in wrapper_text
        assert "spark_stage6_p4b_single_cta_microbench.py" in wrapper_text
        assert "summarize_stage6_p4b_single_cta_microbench.py" in wrapper_text
        assert "--candidate-mode \"$CANDIDATE_MODE\"" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_cta_microbench.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel speed | `False`" in pass_summary.stdout
        assert "P4B/P4A speedup | `0.2`" in pass_summary.stdout

        for name in ("promotion_fail", "timing_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4b_single_cta_microbench.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4B single-CTA microbench tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
