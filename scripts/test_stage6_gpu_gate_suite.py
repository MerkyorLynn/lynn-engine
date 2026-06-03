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
        assert "--p2o-max-seq-len" in help_proc.stdout

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
        ])
        assert "p2o-basic" in dry.stdout
        assert "p2o-rc-mini" in dry.stdout
        assert "p3a-contract" in dry.stdout
        suite_dirs = list(tmp.glob("stage6_gpu_gate_suite_*"))
        assert len(suite_dirs) == 1
        suite = suite_dirs[0]
        status = (suite / "suite_status.tsv").read_text()
        commands = (suite / "commands.sh").read_text()
        summary = (suite / "summary.md").read_text()
        assert "p2o-basic\tDRY_RUN\t0" in status
        assert "p2o-rc-mini\tDRY_RUN\t0" in status
        assert "p3a-contract\tDRY_RUN\t0" in status
        assert "scripts/run_spark_stage6_p2o_rc_smoke.sh" in commands
        assert "--preset rc-mini" in commands
        assert "scripts/run_spark_stage6_p3a_contract_probe.sh" in commands
        assert "--batches" in commands
        assert "1,4" in commands or "1\\,4" in commands
        assert "Failures | `0`" in summary

        skip = run([
            "scripts/run_stage6_gpu_gate_suite.sh",
            "--dry-run",
            "--local-root",
            str(tmp),
            "--skip-p2o-basic",
            "--skip-p2o-rc-mini",
            "--skip-p3a",
        ])
        assert "artifacts:" in skip.stdout

    print("Stage 6 GPU gate-suite self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
