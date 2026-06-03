#!/usr/bin/env python3
"""Local self-test for Stage 6 P3-B evidence tooling.

GPU-free by design: it verifies wrapper syntax, summary verdicts, and report
boundaries for the selected-prefill gate while Spark may be unavailable.
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
        "schema": "lynn-stage6-p3b-selected-prefill-gate-v1",
        "verdict": "PASS",
        "banked_fused_kernel": False,
        "banked_server_path": False,
        "model": "/models/demo",
        "layers": [0, 1, 2, 3],
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "seed": 20260604,
        "seq_lens": [16, 64],
        "env": {
            "candidate_mode": "p3a_grouped",
            "reference_mode": "p2e_hybrid",
            "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA": "1",
        },
        "shape": {"hidden": 2048, "num_experts": 256, "top_k": 8},
        "bytes": {
            "bf16_active_experts": 6442450944,
            "packed_active_experts": 2415919104,
            "mem_after_load_gib": 10.0,
            "mem_after_deleting_bf16_active_gib": 4.0,
            "mem_drop_after_deleting_bf16_active_gib": 6.0,
        },
        "numeric": {
            "p2n_T16_vs_bf16": {"cosine": 0.99995, "argmax_match": True},
            "p3b_T16_vs_bf16": {"cosine": 0.99994, "argmax_match": True},
            "p3b_T16_vs_p2n": {"cosine": 0.99999, "argmax_match": True},
            "p2n_T64_vs_bf16": {"cosine": 0.99991, "argmax_match": True},
            "p3b_T64_vs_bf16": {"cosine": 0.99990, "argmax_match": True},
            "p3b_T64_vs_p2n": {"cosine": 0.99999, "argmax_match": True},
        },
        "bench": {
            "rows": [
                {
                    "seq_len": 16,
                    "bf16_prefill_us": 100000.0,
                    "p2n_reference_us": 75000.0,
                    "p3b_selected_prefill_us": 70000.0,
                    "p3b_vs_bf16": 1.428,
                    "p3b_vs_p2n": 1.071,
                    "p3b_cosine_vs_bf16": 0.99994,
                    "p3b_argmax_vs_bf16": True,
                },
                {
                    "seq_len": 64,
                    "bf16_prefill_us": 180000.0,
                    "p2n_reference_us": 160000.0,
                    "p3b_selected_prefill_us": 150000.0,
                    "p3b_vs_bf16": 1.2,
                    "p3b_vs_p2n": 1.066,
                    "p3b_cosine_vs_bf16": 0.99990,
                    "p3b_argmax_vs_bf16": True,
                },
            ]
        },
        "memory": {
            "p2n_peak": {"16": {"peak_gib": 5.0}, "64": {"peak_gib": 5.5}},
            "p3b_peak": {"16": {"peak_gib": 5.0}, "64": {"peak_gib": 5.5}},
        },
        "shadow_absence_checks": {
            "after_delete": True,
            "after_p2n_T16": True,
            "after_p3b_T16": True,
            "after_p2n_T64": True,
            "after_p3b_T64": True,
        },
        "reload_trap": {
            "installed": True,
            "status": "installed",
        },
        "passes": {
            "predecessors_pass": True,
            "numeric": True,
            "final_stack_cosine_min": 0.99990,
            "final_stack_argmax_match": True,
            "no_active_bf16_shadow": True,
            "reload_trap_installed": True,
            "reload_not_called": True,
            "speed_vs_p2n_reference": True,
            "all": True,
        },
        "reload_calls": [],
        "notes": ["synthetic pass fixture"],
    }


def predecessor_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["passes"] = dict(data["passes"])
    data["passes"]["predecessors_pass"] = False
    data["passes"]["all"] = False
    return data


def speed_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["passes"] = dict(data["passes"])
    data["passes"]["speed_vs_p2n_reference"] = False
    data["passes"]["all"] = False
    data["bench"] = dict(data["bench"])
    data["bench"]["rows"] = [dict(row) for row in data["bench"]["rows"]]
    data["bench"]["rows"][0]["p3b_vs_p2n"] = 0.75
    return data


def promoted_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def malformed_pass_fixture() -> dict:
    data = pass_fixture()
    del data["banked_fused_kernel"]
    del data["banked_server_path"]
    return data


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n")
    (path / "git_head.txt").write_text("remote-head\n")
    (path / "head_check.txt").write_text("remote manifest ok\n")
    expected_manifest = "abc scripts/run_spark_stage6_p3b_selected_prefill_gate.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p3b_selected_prefill_gate.sh\n"
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
        predecessor_fail_json = tmp / "predecessor_fail.json"
        speed_fail_json = tmp / "speed_fail.json"
        promoted_fail_json = tmp / "promoted_fail.json"
        malformed_pass_json = tmp / "malformed_pass.json"
        write_json(pass_json, pass_fixture())
        write_json(predecessor_fail_json, predecessor_fail_fixture())
        write_json(speed_fail_json, speed_fail_fixture())
        write_json(promoted_fail_json, promoted_fail_fixture())
        write_json(malformed_pass_json, malformed_pass_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh"])
        help_proc = run(["scripts/run_spark_stage6_p3b_selected_prefill_gate.sh", "--help"])
        assert "--predecessors-pass" in help_proc.stdout
        assert "--tokens" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p3b_selected_prefill_gate.py",
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            "scripts/write_stage6_p3b_report.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(pass_json),
            "--markdown-out",
            str(tmp / "summary.md"),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert (tmp / "summary.md").exists()

        predecessor_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(predecessor_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "predecessor evidence gate fail" in predecessor_fail.stdout

        speed_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(speed_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "slower than P2-N reference" in speed_fail.stdout

        promoted_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(promoted_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "promotion boundary violated" in promoted_fail.stdout

        malformed_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(malformed_pass_json),
            "--strict-exit",
        ], expect=2)
        assert "promotion boundary violated" in malformed_fail.stdout

        pass_art = tmp / "pass_artifact"
        fail_art = tmp / "fail_artifact"
        write_artifact(pass_art, pass_fixture())
        write_artifact(fail_art, speed_fail_fixture())
        run([
            sys.executable,
            "scripts/summarize_stage6_p3b_selected_prefill_gate.py",
            str(pass_art / "result.json"),
            "--markdown-out",
            str(pass_art / "summary.md"),
            "--strict-exit",
        ])
        pass_report = tmp / "pass_report.md"
        fail_report = tmp / "fail_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_p3b_report.py",
            str(pass_art),
            "--report-out",
            str(pass_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3b_report.py",
            str(fail_art),
            "--report-out",
            str(fail_report),
            "--date",
            "2026-06-04",
        ])
        pass_text = pass_report.read_text()
        fail_text = fail_report.read_text()
        assert "Verdict: **PASS**" in pass_text
        assert "Bank P3-B selected-prefill composition only" in pass_text
        assert "not bank a fused grouped-MoE kernel" in pass_text
        assert "P3-C server integration" in pass_text
        assert "Reload trap installed" in pass_text
        assert "Verdict: **FAIL**" in fail_text
        assert "Do not bank P3-B" in fail_text

    print("P3-B selected-prefill evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
