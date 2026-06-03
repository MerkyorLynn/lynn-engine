#!/usr/bin/env python3
"""Local self-test for Stage 6 P3-D server-smoke evidence tooling."""
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


def _row(text: str, *, degenerate: bool = False) -> dict:
    return {
        "index": 0,
        "prompt": "Explain why water boils.",
        "text": text,
        "normalized_text": text.strip(),
        "degenerate": degenerate,
        "finish_reason": "length",
        "usage": {"completion_tokens": 8},
        "metrics": {"tokens_per_second": 44.0},
        "elapsed_seconds": 0.5,
    }


def pass_fixture() -> dict:
    return {
        "schema": "lynn-stage6-p3d-server-rc-gate-v1",
        "verdict": "PASS",
        "banked_server_smoke": True,
        "banked_default_promotion": False,
        "banked_full_rc_quality": False,
        "model": "/models/demo",
        "preset": "basic",
        "max_new": 8,
        "max_seq_len": 2048,
        "baseline": {
            "models_ok": True,
            "completions": [_row("Water boils when vapor pressure matches ambient pressure.")],
            "chat": [_row("Water boils when vapor pressure matches ambient pressure.")],
        },
        "candidate": {
            "models_ok": True,
            "completions": [_row("Water boils when vapor pressure matches ambient pressure.")],
            "chat": [_row("Water boils when vapor pressure matches ambient pressure.")],
        },
        "comparisons": {
            "completions": [{
                "index": 0,
                "text_exact": True,
                "baseline_text": "Water boils when vapor pressure matches ambient pressure.",
                "candidate_text": "Water boils when vapor pressure matches ambient pressure.",
            }],
            "chat": [{
                "index": 0,
                "text_exact": True,
                "baseline_text": "Water boils when vapor pressure matches ambient pressure.",
                "candidate_text": "Water boils when vapor pressure matches ambient pressure.",
            }],
        },
        "candidate_health": {
            "release_reload_count": 1,
            "reload_expected_min": 1,
            "last_release_gib": 60.0,
            "last_reload_seconds": 23.0,
            "release_enabled": True,
            "release_consumed": True,
            "decode_shadows_currently_released": True,
        },
        "passes": {
            "p3c_pass": True,
            "server_surface": True,
            "prompt_count": True,
            "functional_non_degenerate": True,
            "server_text_exact": True,
            "release_enabled": True,
            "release_consumed": True,
            "decode_shadows_currently_released": True,
            "release_meaningful": True,
            "reload_observed": True,
            "all": True,
        },
        "notes": ["synthetic pass fixture"],
    }


def predecessor_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_server_smoke"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["p3c_pass"] = False
    data["passes"]["all"] = False
    return data


def text_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_server_smoke"] = False
    data["comparisons"] = dict(data["comparisons"])
    data["comparisons"]["completions"] = [dict(data["comparisons"]["completions"][0], text_exact=False)]
    data["passes"] = dict(data["passes"])
    data["passes"]["server_text_exact"] = False
    data["passes"]["all"] = False
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_default_promotion"] = True
    return data


def full_rc_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_full_rc_quality"] = True
    return data


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n")
    (path / "git_head.txt").write_text("remote-head\n")
    (path / "head_check.txt").write_text("remote manifest ok\n")
    expected_manifest = "abc scripts/run_spark_stage6_p3d_server_rc_gate.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p3d_server_rc_gate.sh\n"
    (path / "expected_provenance_manifest.txt").write_text(expected_manifest)
    (path / "provenance_manifest.txt").write_text(actual_manifest)
    (path / "git_status.txt").write_text("")
    (path / "docker_exit_code.txt").write_text("0\n")
    (path / "nvidia_smi_before.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "nvidia_smi_after.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "run.log").write_text("fixture run log tail\n")
    (path / "baseline_server.log").write_text("fixture baseline log tail\n")
    (path / "candidate_server.log").write_text("fixture candidate log tail\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        predecessor_fail_json = tmp / "predecessor_fail.json"
        text_fail_json = tmp / "text_fail.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        full_rc_fail_json = tmp / "full_rc_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(predecessor_fail_json, predecessor_fail_fixture())
        write_json(text_fail_json, text_fail_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())
        write_json(full_rc_fail_json, full_rc_fail_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p3d_server_rc_gate.sh"])
        help_proc = run(["scripts/run_spark_stage6_p3d_server_rc_gate.sh", "--help"])
        assert "--p3c-pass" in help_proc.stdout
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        bad_preset = run(["scripts/run_spark_stage6_p3d_server_rc_gate.sh", "--preset", "nope"], expect=2)
        assert "--preset must be basic or rc-mini" in bad_preset.stderr

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p3d_server_rc_gate.py",
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            "scripts/write_stage6_p3d_report.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked default promotion | `False`" in pass_summary.stdout
        assert "Banked full RC quality | `False`" in pass_summary.stdout

        predecessor_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(predecessor_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "p3c_pass gate fail" in predecessor_fail.stdout

        text_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(text_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "server_text_exact gate fail" in text_fail.stdout

        promotion_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(promotion_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "default promotion boundary violated" in promotion_fail.stdout

        full_rc_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(full_rc_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "full RC quality boundary violated" in full_rc_fail.stdout

        pass_art = tmp / "pass_artifact"
        fail_art = tmp / "fail_artifact"
        write_artifact(pass_art, pass_fixture())
        write_artifact(fail_art, text_fail_fixture())
        run([
            sys.executable,
            "scripts/summarize_stage6_p3d_server_rc_gate.py",
            str(pass_art / "result.json"),
            "--markdown-out",
            str(pass_art / "summary.md"),
            "--strict-exit",
        ])
        pass_report = tmp / "pass_report.md"
        fail_report = tmp / "fail_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_p3d_report.py",
            str(pass_art),
            "--report-out",
            str(pass_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3d_report.py",
            str(fail_art),
            "--report-out",
            str(fail_report),
            "--date",
            "2026-06-04",
        ])
        pass_text = pass_report.read_text()
        fail_text = fail_report.read_text()
        assert "Verdict: **PASS**" in pass_text
        assert "opt-in server smoke only" in pass_text
        assert "not default promotion and not full RC quality" in pass_text
        assert "Verdict: **FAIL**" in fail_text
        assert "Do not bank P3-D" in fail_text

        empty_prompts = tmp / "empty_prompts.json"
        empty_prompts.write_text("[]\n")
        empty_proc = run([
            sys.executable,
            "scripts/spark_stage6_p3d_server_rc_gate.py",
            "--prompts-json",
            str(empty_prompts),
        ], expect=None)
        if empty_proc.returncode == 0:
            raise AssertionError("empty prompt list unexpectedly succeeded")
        assert "prompt list must be non-empty" in (empty_proc.stderr + empty_proc.stdout)

    print("P3-D server RC smoke evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
