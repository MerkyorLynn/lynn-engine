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


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n")
    (path / "git_head.txt").write_text("remote-head\n")
    (path / "head_check.txt").write_text("remote manifest ok\n")
    expected_manifest = "abc scripts/run_spark_stage6_p3a_contract_probe.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p3a_contract_probe.sh\n"
    (path / "expected_provenance_manifest.txt").write_text(expected_manifest)
    (path / "provenance_manifest.txt").write_text(actual_manifest)
    (path / "git_status.txt").write_text("")
    (path / "docker_exit_code.txt").write_text("0\n")
    (path / "nvidia_smi_before.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "nvidia_smi_after.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "run.log").write_text("fixture log tail\n")


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
            "scripts/write_stage6_p3a_report.py",
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

        pass_art = tmp / "pass_artifact"
        fail_art = tmp / "fail_artifact"
        promoted_art = tmp / "promoted_artifact"
        write_artifact(pass_art, pass_fixture())
        write_artifact(fail_art, fail_numeric_fixture())
        write_artifact(promoted_art, fail_promoted_fixture())
        run([
            sys.executable,
            "scripts/summarize_stage6_p3a_contract_probe.py",
            str(pass_art / "result.json"),
            "--markdown-out",
            str(pass_art / "summary.md"),
            "--strict-exit",
        ])
        pass_report = tmp / "pass_report.md"
        fail_report = tmp / "fail_report.md"
        promoted_report = tmp / "promoted_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_p3a_report.py",
            str(pass_art),
            "--report-out",
            str(pass_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3a_report.py",
            str(fail_art),
            "--report-out",
            str(fail_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3a_report.py",
            str(promoted_art),
            "--report-out",
            str(promoted_report),
            "--date",
            "2026-06-04",
        ])
        pass_text = pass_report.read_text()
        fail_text = fail_report.read_text()
        promoted_text = promoted_report.read_text()
        assert "Verdict: **PASS**" in pass_text
        assert "Bank P3-A as a contract-shaped grouped active-MoE probe only" in pass_text
        assert "Do not promote P3 or claim a fused kernel" in pass_text
        assert "Manifest matches | `True`" in pass_text
        assert "Verdict: **FAIL**" in fail_text
        assert "Do not bank P3-A" in fail_text
        assert "promotion boundary violated" in promoted_text

    print("P3-A evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
