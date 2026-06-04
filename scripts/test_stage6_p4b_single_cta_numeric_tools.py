#!/usr/bin/env python3
"""Local self-test for Stage 6 P4B single-CTA numeric tooling."""
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
        "schema": "lynn-stage6-p4b-single-cta-numeric-preflight-v1",
        "decision": "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE",
        "symbol": "active_moe_fused_zero_shadow_single_kernel_contract",
        "reference_symbol": "active_moe_fused_zero_shadow_out_contract",
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "torch_version": "fixture",
        "torch_cuda": "fixture",
        "build_dir": "/tmp/fixture",
        "banked_single_cta_numeric_preflight": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "reference_output": {"shape": [1, 2048], "dtype": "torch.bfloat16", "finite": True, "norm": 1.0},
        "candidate_output": {
            "shape": [1, 2048],
            "dtype": "torch.bfloat16",
            "finite": True,
            "norm": 1.0,
            "max_abs_diff_vs_reference": 0.01,
            "mean_abs_diff_vs_reference": 0.001,
            "rel_l2_vs_reference": 0.01,
        },
        "byte_budget": {
            "packed_vs_bf16_shadow_ratio": 0.375,
            "no_inter_scratch_candidate_abi": True,
        },
        "passes": {
            "extension_loaded": True,
            "symbol_present": True,
            "reference_symbol_present": True,
            "reference_output_returned": True,
            "candidate_output_returned": True,
            "reference_finite": True,
            "candidate_finite": True,
            "numeric_vs_reference": True,
            "zero_shadow_candidate_abi": True,
            "packed_byte_budget": True,
            "no_inter_scratch_candidate_abi": True,
            "all": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def numeric_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_single_cta_numeric_preflight"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["numeric_vs_reference"] = False
    data["passes"]["all"] = False
    data["candidate_output"] = dict(data["candidate_output"])
    data["candidate_output"]["rel_l2_vs_reference"] = 0.5
    return data


def inter_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_single_cta_numeric_preflight"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["no_inter_scratch_candidate_abi"] = False
    data["passes"]["all"] = False
    data["byte_budget"] = dict(data["byte_budget"])
    data["byte_budget"]["no_inter_scratch_candidate_abi"] = False
    return data


def main() -> int:
    wrapper = ROOT / "scripts" / "run_spark_stage6_p4b_single_cta_numeric_preflight.sh"
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "promotion_fail": promotion_fail_fixture(),
            "numeric_fail": numeric_fail_fixture(),
            "inter_fail": inter_fail_fixture(),
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
            "scripts/spark_stage6_p4b_single_cta_numeric_preflight.py",
            "scripts/summarize_stage6_p4b_single_cta_numeric_preflight.py",
        ])
        wrapper_help = run([str(wrapper), "--help"])
        wrapper_text = wrapper.read_text(encoding="utf-8")
        assert "p4b_single_cta_numeric_preflight_" in wrapper_text
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in wrapper_help.stdout
        assert "--candidate-mode MODE" in wrapper_help.stdout
        assert "spark_stage6_p4b_single_cta_numeric_preflight.py" in wrapper_text
        assert "--candidate-mode \"$CANDIDATE_MODE\"" in wrapper_text
        assert "p4b_multi_cta_numeric_preflight_" in wrapper_text

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_cta_numeric_preflight.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Candidate mode | `single_cta`" in pass_summary.stdout
        assert "Banked fused kernel speed | `False`" in pass_summary.stdout
        assert "No inter_scratch candidate ABI | `True`" in pass_summary.stdout

        for name in ("promotion_fail", "numeric_fail", "inter_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4b_single_cta_numeric_preflight.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4B single-CTA numeric tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
