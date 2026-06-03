#!/usr/bin/env python3
"""GPU-free self-test for the Stage 6 GPU gate-suite wrapper."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, expect: int = 0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != expect:
        raise AssertionError(
            f"command failed with {proc.returncode}, expected {expect}: {cmd}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        run(["bash", "-n", "scripts/run_stage6_gpu_gate_suite.sh"])
        help_proc = run(["scripts/run_stage6_gpu_gate_suite.sh", "--help"])
        assert "--dry-run" in help_proc.stdout
        assert "--skip-p3a" in help_proc.stdout
        assert "--skip-p3b" in help_proc.stdout
        assert "--p2o-max-seq-len" in help_proc.stdout
        assert "--p3b-layers" in help_proc.stdout

        dry = run([
            "scripts/run_stage6_gpu_gate_suite.sh",
            "--dry-run",
            "--host",
            "dry-host",
            "--model",
            "/models/demo",
            "--image",
            "demo:latest",
            "--remote-repo",
            "/remote/repo",
            "--local-root",
            str(tmp),
            "--expect-head",
            "demo-head",
            "--allow-provenance-mismatch",
            "--p2o-max-new",
            "4",
            "--p2o-max-seq-len",
            "1024",
            "--p3a-layer",
            "2",
            "--p3a-batches",
            "1,4",
            "--p3b-layers",
            "0-3",
            "--p3b-tokens",
            "16,64",
        ])
        assert "p2o-basic" in dry.stdout
        assert "p2o-rc-mini" in dry.stdout
        assert "p3a-contract" in dry.stdout
        assert "p3b-selected-prefill" in dry.stdout
        suite_dirs = list(tmp.glob("stage6_gpu_gate_suite_*"))
        assert len(suite_dirs) == 1
        suite = suite_dirs[0]
        status = (suite / "suite_status.tsv").read_text()
        commands = (suite / "commands.sh").read_text()
        summary = (suite / "summary.md").read_text()
        report = (suite / "report.md").read_text()
        assert "p2o-basic\tDRY_RUN\t0" in status
        assert "p2o-rc-mini\tDRY_RUN\t0" in status
        assert "p3a-contract\tDRY_RUN\t0" in status
        assert "p3b-selected-prefill\tDRY_RUN\t0" in status
        assert "scripts/run_spark_stage6_p2o_rc_smoke.sh" in commands
        assert "--preset rc-mini" in commands
        assert "scripts/run_spark_stage6_p3a_contract_probe.sh" in commands
        assert "--batches" in commands
        assert "1,4" in commands or "1\\,4" in commands
        assert "scripts/run_spark_stage6_p3b_selected_prefill_gate.sh" in commands
        assert "--predecessors-pass" in commands
        assert "16,64" in commands or "16\\,64" in commands
        assert "Failures | `0`" in summary
        assert "Verdict: **DRY_RUN**" in report
        assert "dry-run only" in report

        dry_no_strict = run([
            "scripts/run_stage6_gpu_gate_suite.sh",
            "--dry-run",
            "--no-strict",
            "--local-root",
            str(tmp),
        ])
        assert "artifacts:" in dry_no_strict.stdout
        no_strict_dirs = sorted(tmp.glob("stage6_gpu_gate_suite_*"), key=lambda p: p.stat().st_mtime)
        no_strict_commands = (no_strict_dirs[-1] / "commands.sh").read_text()
        assert "--no-strict" not in no_strict_commands

        before_skip = set(tmp.glob("stage6_gpu_gate_suite_*"))
        skip = run([
            "scripts/run_stage6_gpu_gate_suite.sh",
            "--dry-run",
            "--local-root",
            str(tmp),
            "--skip-p2o-basic",
            "--skip-p2o-rc-mini",
            "--skip-p3a",
            "--skip-p3b",
        ])
        assert "artifacts:" in skip.stdout
        after_skip = set(tmp.glob("stage6_gpu_gate_suite_*"))
        new_skip = sorted(after_skip - before_skip)
        assert len(new_skip) == 1
        assert "Verdict: **SKIP**" in (new_skip[0] / "report.md").read_text()

        before_skip_real = set(tmp.glob("stage6_gpu_gate_suite_*"))
        skip_real = run([
            "scripts/run_stage6_gpu_gate_suite.sh",
            "--local-root",
            str(tmp),
            "--skip-p2o-basic",
            "--skip-p2o-rc-mini",
            "--skip-p3a",
            "--skip-p3b",
        ])
        assert "artifacts:" in skip_real.stdout
        after_skip_real = set(tmp.glob("stage6_gpu_gate_suite_*"))
        new_skip_real = sorted(after_skip_real - before_skip_real)
        assert len(new_skip_real) == 1
        assert "Verdict: **SKIP**" in (new_skip_real[0] / "report.md").read_text()

        synthetic = tmp / "synthetic_suite"
        synthetic.mkdir()
        (synthetic / "suite_meta.env").write_text(
            "\n".join([
                "local_head=abc",
                "expected_head=abc",
                "host=dgx-spark",
                "model=/models/demo",
                "image=demo:latest",
                "remote_repo=/remote/repo",
                "strict=1",
                "dry_run=0",
            ]) + "\n"
        )
        (synthetic / "suite_status.tsv").write_text(
            "step\tstatus\texit_code\n"
            "p2o-basic\tPASS\t0\n"
            "p2o-rc-mini\tPASS\t0\n"
            "p3a-contract\tPASS\t0\n"
            "p3b-selected-prefill\tPASS\t0\n"
        )
        (synthetic / "commands.sh").write_text("echo synthetic\n")
        (synthetic / "summary.md").write_text("# Synthetic summary\n\nFailures | `0`\n")
        (synthetic / "local_git_status.txt").write_text("")
        child = synthetic / "p3a_layer0_grouped_moe_contract_probe_20260604_000000"
        child.mkdir()
        (child / "result.json").write_text("{}\n")
        (child / "summary.md").write_text("# Child summary\n\nVerdict | **PASS**\n")
        (child / "head_check.txt").write_text("remote HEAD ok\n")
        (child / "docker_exit_code.txt").write_text("0\n")
        p3b_child = synthetic / "p3b_layers0-3_selected_prefill_gate_20260604_000000"
        p3b_child.mkdir()
        (p3b_child / "result.json").write_text("{}\n")
        (p3b_child / "summary.md").write_text("# P3-B child summary\n\nVerdict | **PASS**\n")
        (p3b_child / "head_check.txt").write_text("remote HEAD ok\n")
        (p3b_child / "docker_exit_code.txt").write_text("0\n")
        report_out = tmp / "synthetic_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_gpu_gate_suite_report.py",
            str(synthetic),
            "--report-out",
            str(report_out),
            "--date",
            "2026-06-04",
        ])
        synthetic_report = report_out.read_text()
        assert "Verdict: **PASS**" in synthetic_report
        assert "p3a_layer0_grouped_moe_contract_probe" in synthetic_report
        assert "p3b_layers0-3_selected_prefill_gate" in synthetic_report
        assert "Child summary" in synthetic_report
        assert "P3-B child summary" in synthetic_report

    print("Stage 6 GPU gate-suite self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
