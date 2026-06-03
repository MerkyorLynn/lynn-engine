#!/usr/bin/env python3
"""Local self-test for Stage 6 P3-C evidence tooling."""
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
        "schema": "lynn-stage6-p3c-resident-prompt-gate-v1",
        "verdict": "PASS",
        "banked_server_path": False,
        "banked_rc_quality": False,
        "model": "/models/demo",
        "preset": "basic",
        "max_new": 8,
        "memory": {
            "loaded_gib": 88.2,
            "after_release_gib": 28.1,
            "drop_gib": 60.1,
            "release": {"released_gib": 60.1, "released_tensors": 120},
        },
        "baseline": [{"prefill_seconds": 1.0, "decode_tps": 44.0, "degenerate": False}],
        "candidate_no_reload": [{"prefill_seconds": 0.8, "decode_tps": 43.0, "degenerate": False}],
        "comparisons": [{
            "index": 0,
            "token_exact": True,
            "token_prefix_match": 8,
            "token_prefix_n": 8,
            "text_prefix_200_match": True,
            "baseline_ids": [1, 2],
            "candidate_ids": [1, 2],
        }],
        "passes": {
            "p3b_pass": True,
            "prompt_count": True,
            "functional_non_degenerate": True,
            "generated_token_exact": True,
            "token_exact": True,
            "text_prefix_200_match": True,
            "release_meaningful": True,
            "memory_drop_meaningful": True,
            "reload_not_called": True,
            "all": True,
        },
        "reload_calls": [],
        "notes": ["synthetic pass fixture"],
    }


def p3b_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["passes"] = dict(data["passes"])
    data["passes"]["p3b_pass"] = False
    data["passes"]["all"] = False
    return data


def release_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["memory"] = {
        "loaded_gib": 88.2,
        "after_release_gib": 88.1,
        "drop_gib": 0.1,
        "release": {"released_gib": 0.0, "released_tensors": 0},
    }
    data["passes"] = dict(data["passes"])
    data["passes"]["release_meaningful"] = False
    data["passes"]["memory_drop_meaningful"] = False
    data["passes"]["all"] = False
    return data


def promoted_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_server_path"] = True
    return data


def malformed_pass_fixture() -> dict:
    data = pass_fixture()
    del data["banked_server_path"]
    return data


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n")
    (path / "git_head.txt").write_text("remote-head\n")
    (path / "head_check.txt").write_text("remote manifest ok\n")
    expected_manifest = "abc scripts/run_spark_stage6_p3c_resident_prompt_gate.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p3c_resident_prompt_gate.sh\n"
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
        p3b_fail_json = tmp / "p3b_fail.json"
        release_fail_json = tmp / "release_fail.json"
        promoted_fail_json = tmp / "promoted_fail.json"
        malformed_pass_json = tmp / "malformed_pass.json"
        write_json(pass_json, pass_fixture())
        write_json(p3b_fail_json, p3b_fail_fixture())
        write_json(release_fail_json, release_fail_fixture())
        write_json(promoted_fail_json, promoted_fail_fixture())
        write_json(malformed_pass_json, malformed_pass_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p3c_resident_prompt_gate.sh"])
        help_proc = run(["scripts/run_spark_stage6_p3c_resident_prompt_gate.sh", "--help"])
        assert "--p3b-pass" in help_proc.stdout
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        bad_preset = run(["scripts/run_spark_stage6_p3c_resident_prompt_gate.sh", "--preset", "nope"], expect=2)
        assert "--preset must be basic or rc-mini" in bad_preset.stderr

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p3c_resident_prompt_gate.py",
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            "scripts/write_stage6_p3c_report.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked server path | `False`" in pass_summary.stdout

        p3b_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(p3b_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "p3b_pass gate fail" in p3b_fail.stdout

        release_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(release_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "release_meaningful gate fail" in release_fail.stdout

        promoted_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(promoted_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "promotion boundary violated" in promoted_fail.stdout

        malformed_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(malformed_pass_json),
            "--strict-exit",
        ], expect=2)
        assert "promotion boundary violated" in malformed_fail.stdout

        pass_art = tmp / "pass_artifact"
        fail_art = tmp / "fail_artifact"
        write_artifact(pass_art, pass_fixture())
        write_artifact(fail_art, release_fail_fixture())
        run([
            sys.executable,
            "scripts/summarize_stage6_p3c_resident_prompt_gate.py",
            str(pass_art / "result.json"),
            "--markdown-out",
            str(pass_art / "summary.md"),
            "--strict-exit",
        ])
        pass_report = tmp / "pass_report.md"
        fail_report = tmp / "fail_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_p3c_report.py",
            str(pass_art),
            "--report-out",
            str(pass_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3c_report.py",
            str(fail_art),
            "--report-out",
            str(fail_report),
            "--date",
            "2026-06-04",
        ])
        pass_text = pass_report.read_text()
        fail_text = fail_report.read_text()
        assert "Verdict: **PASS**" in pass_text
        assert "Bank P3-C as a resident-runner real-prompt smoke only" in pass_text
        assert "not a server/default/RC promotion" in pass_text
        assert "Verdict: **FAIL**" in fail_text
        assert "Do not bank P3-C" in fail_text

        empty_prompts = tmp / "empty_prompts.json"
        empty_prompts.write_text("[]\n")
        empty_proc = run([
            sys.executable,
            "scripts/spark_stage6_p3c_resident_prompt_gate.py",
            "--prompts-json",
            str(empty_prompts),
        ], expect=None)
        if empty_proc.returncode == 0:
            raise AssertionError("empty prompt list unexpectedly succeeded")
        assert "prompt list must be non-empty" in (empty_proc.stderr + empty_proc.stdout)

    print("P3-C resident-prompt evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
