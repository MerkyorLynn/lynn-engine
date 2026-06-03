#!/usr/bin/env python3
"""Local self-test for Stage 6 P3-A evidence tooling.

This is intentionally GPU-free. It verifies that the P3-A artifact pipeline
preserves the hard promotion boundary while Spark is unavailable.
"""
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
    path.write_text(json.dumps(data, indent=2) + "\n")


def pass_fixture() -> dict:
    return {
        "schema": "lynn-stage6-p3a-grouped-moe-contract-probe-v1",
        "verdict": "PASS",
        "banked_fused_kernel": False,
        "model": "/models/demo",
        "layer": 0,
        "seed": 20260604,
        "batches": [1, 16],
        "shape": {"hidden": 2048, "intermediate": 512, "num_experts": 256, "top_k": 8},
        "tiles": {
            "block_t": 32,
            "block_inter": 8,
            "block_hidden": 128,
            "num_warps": 4,
            "down_block_hidden": 8,
            "down_block_inter": 512,
            "down_num_warps": 8,
        },
        "bytes": {
            "bf16_layer_active_experts": 805306368,
            "packed_layer_active_experts": 234881024,
            "max_inter_scratch_estimate": 131072,
            "mem_after_deleting_bf16_active_gib": 28.2,
        },
        "numeric": {
            "1": {"cosine": 0.9999, "argmax_match": True},
            "16": {"cosine": 0.9998, "argmax_match": True},
        },
        "bench": {
            "rows": [
                {
                    "batch": 1,
                    "unique_experts": 8,
                    "bf16_active_us": 100.0,
                    "p3a_contract_us": 120.0,
                    "p3a_vs_bf16": 0.833,
                    "cosine": 0.9999,
                    "argmax_match": True,
                },
                {
                    "batch": 16,
                    "unique_experts": 64,
                    "bf16_active_us": 4000.0,
                    "p3a_contract_us": 3000.0,
                    "p3a_vs_bf16": 1.333,
                    "cosine": 0.9998,
                    "argmax_match": True,
                },
            ],
        },
        "memory": {
            "p3a_candidate_peak": {
                "1": {"before_gib": 28.2, "after_gib": 28.3, "peak_gib": 29.0},
                "16": {"before_gib": 28.2, "after_gib": 28.4, "peak_gib": 30.0},
            }
        },
        "passes": {
            "numeric": True,
            "shadow_absent_at_candidate_start": True,
            "all": True,
        },
        "notes": ["synthetic pass fixture; no fused kernel banked"],
    }


def fail_numeric_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["passes"] = dict(data["passes"])
    data["passes"]["numeric"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["rows"] = [dict(row) for row in data["bench"]["rows"]]
    data["bench"]["rows"][0]["cosine"] = 0.95
    data["notes"] = ["synthetic numeric fail fixture"]
    return data


def fail_shadow_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["passes"] = dict(data["passes"])
    data["passes"]["shadow_absent_at_candidate_start"] = False
    data["passes"]["all"] = False
    data["notes"] = ["synthetic shadow fail fixture"]
    return data


def fail_promoted_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    data["notes"] = ["synthetic promotion-boundary fail fixture"]
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        fail_numeric_json = tmp / "fail_numeric.json"
        fail_shadow_json = tmp / "fail_shadow.json"
        fail_promoted_json = tmp / "fail_promoted.json"
        write_json(pass_json, pass_fixture())
        write_json(fail_numeric_json, fail_numeric_fixture())
        write_json(fail_shadow_json, fail_shadow_fixture())
        write_json(fail_promoted_json, fail_promoted_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p3a_contract_probe.sh"])
        help_proc = run(["scripts/run_spark_stage6_p3a_contract_probe.sh", "--help"])
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p3a_grouped_moe_contract_probe.py",
            "scripts/summarize_stage6_p3a_contract_probe.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p3a_contract_probe.py",
            str(pass_json),
            "--markdown-out",
            str(tmp / "summary.md"),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert (tmp / "summary.md").exists()

        fail_numeric = run([
            sys.executable,
            "scripts/summarize_stage6_p3a_contract_probe.py",
            str(fail_numeric_json),
            "--strict-exit",
        ], expect=2)
        assert "numeric gate fail" in fail_numeric.stdout

        fail_shadow = run([
            sys.executable,
            "scripts/summarize_stage6_p3a_contract_probe.py",
            str(fail_shadow_json),
            "--strict-exit",
        ], expect=2)
        assert "BF16 active shadow was not absent" in fail_shadow.stdout

        fail_promoted = run([
            sys.executable,
            "scripts/summarize_stage6_p3a_contract_probe.py",
            str(fail_promoted_json),
            "--strict-exit",
        ], expect=2)
        assert "promotion boundary violated" in fail_promoted.stdout

    print("P3-A evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
