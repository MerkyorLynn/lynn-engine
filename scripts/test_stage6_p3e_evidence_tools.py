#!/usr/bin/env python3
"""Local self-test for Stage 6 P3-E RC quality-battery evidence tooling."""
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
        "schema": "lynn-stage6-p3e-rc-quality-battery-v1",
        "verdict": "PASS",
        "banked_rc_quality_smoke": True,
        "banked_default_promotion": False,
        "banked_full_leaderboard_quality": False,
        "model": "/models/demo",
        "served_name": "Lynn-P3E-candidate",
        "max_seq_len": 32768,
        "wall_seconds": 123.0,
        "thresholds": {
            "mmlu_floor": 0.70,
            "gpqa_floor": 0.30,
            "max_parse_fail_rate": 0.10,
            "mmlu_sample": 100,
            "gpqa_sample": 50,
            "longctx_target_tokens": 8192,
        },
        "health_before": {"status": "ok", "release_decode_shadows_after_prefill": True},
        "health_after": {
            "status": "ok",
            "release_decode_shadows_after_prefill": True,
            "skip_reload_when_packed_prefill": True,
            "release_reload_count": 0,
            "last_release_gib": 60.0,
        },
        "models": {"data": [{"id": "Lynn-P3E-candidate"}]},
        "structured": {"ok": True, "text": "{\"city\":\"Tokyo\",\"unit\":\"celsius\"}"},
        "tool": {"ok": True, "tool_calls": [{"function": {"name": "get_weather"}}]},
        "prompt_smoke": {
            "ok": True,
            "rows": [
                {"id": "v8_format_smoke", "ok": True, "text": "linear attention helps long context with linear scaling."},
                {"id": "v9_reasoning_smoke", "ok": True, "text": "Experts let routing focus compute on relevant expert capacity."},
            ],
        },
        "longctx": {"ok": True, "needle": "P3E-LONGCTX-NEEDLE-7749", "text": "P3E-LONGCTX-NEEDLE-7749"},
        "mcq": {
            "mmlu": {
                "available": True,
                "summary_data": {"n": 100, "correct": 76, "accuracy": 0.76, "parse_fail": 0, "errors": 0},
            },
            "gpqa": {
                "available": True,
                "summary_data": {"n": 50, "correct": 20, "accuracy": 0.40, "parse_fail": 1, "errors": 0},
            },
        },
        "passes": {
            "p3d_pass": True,
            "server_ready": True,
            "models_surface": True,
            "release_enabled": True,
            "skip_reload_enabled": True,
            "zero_reload_observed": True,
            "structured_json": True,
            "tool_call": True,
            "v8_v9_prompt_smoke": True,
            "longctx_needle": True,
            "mmlu_available": True,
            "gpqa_available": True,
            "mmlu_score": True,
            "gpqa_score": True,
            "no_runner_error": True,
            "all": True,
        },
        "notes": ["synthetic pass fixture"],
    }


def predecessor_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_rc_quality_smoke"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["p3d_pass"] = False
    data["passes"]["all"] = False
    return data


def mmlu_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_rc_quality_smoke"] = False
    data["mcq"] = dict(data["mcq"])
    data["mcq"]["mmlu"] = {
        "available": True,
        "summary_data": {"n": 100, "correct": 50, "accuracy": 0.50, "parse_fail": 0, "errors": 0},
    }
    data["passes"] = dict(data["passes"])
    data["passes"]["mmlu_score"] = False
    data["passes"]["all"] = False
    return data


def missing_gpqa_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_rc_quality_smoke"] = False
    data["mcq"] = dict(data["mcq"])
    data["mcq"]["gpqa"] = {"available": False}
    data["passes"] = dict(data["passes"])
    data["passes"]["gpqa_available"] = False
    data["passes"]["gpqa_score"] = False
    data["passes"]["all"] = False
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_default_promotion"] = True
    return data


def full_leaderboard_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_full_leaderboard_quality"] = True
    return data


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n")
    (path / "git_head.txt").write_text("remote-head\n")
    (path / "head_check.txt").write_text("remote manifest ok\n")
    expected_manifest = "abc scripts/run_spark_stage6_p3e_rc_quality_battery.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p3e_rc_quality_battery.sh\n"
    (path / "expected_provenance_manifest.txt").write_text(expected_manifest)
    (path / "provenance_manifest.txt").write_text(actual_manifest)
    (path / "git_status.txt").write_text("")
    (path / "docker_exit_code.txt").write_text("0\n")
    (path / "nvidia_smi_before.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "nvidia_smi_after.txt").write_text("NVIDIA GPU, 0 %, 1024 MiB, 122880 MiB\n")
    (path / "run.log").write_text("fixture run log tail\n")
    (path / "candidate_server.log").write_text("fixture candidate server log tail\n")
    (path / "mmlu_sample.summary.json").write_text(json.dumps(data["mcq"].get("mmlu", {}).get("summary_data") or {}) + "\n")
    (path / "gpqa_sample.summary.json").write_text(json.dumps(data["mcq"].get("gpqa", {}).get("summary_data") or {}) + "\n")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        predecessor_fail_json = tmp / "predecessor_fail.json"
        mmlu_fail_json = tmp / "mmlu_fail.json"
        missing_gpqa_json = tmp / "missing_gpqa.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        full_leaderboard_fail_json = tmp / "full_leaderboard_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(predecessor_fail_json, predecessor_fail_fixture())
        write_json(mmlu_fail_json, mmlu_fail_fixture())
        write_json(missing_gpqa_json, missing_gpqa_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())
        write_json(full_leaderboard_fail_json, full_leaderboard_fail_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p3e_rc_quality_battery.sh"])
        help_proc = run(["scripts/run_spark_stage6_p3e_rc_quality_battery.sh", "--help"])
        assert "--p3d-pass" in help_proc.stdout
        assert "--mmlu-sample" in help_proc.stdout
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p3e_rc_quality_battery.py",
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            "scripts/write_stage6_p3e_report.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked default promotion | `False`" in pass_summary.stdout
        assert "Banked full leaderboard quality | `False`" in pass_summary.stdout

        predecessor_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(predecessor_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "p3d_pass gate fail" in predecessor_fail.stdout

        mmlu_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(mmlu_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "mmlu_score gate fail" in mmlu_fail.stdout

        missing_gpqa = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(missing_gpqa_json),
            "--strict-exit",
        ], expect=2)
        assert "gpqa_available gate fail" in missing_gpqa.stdout

        promotion_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(promotion_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "default promotion boundary violated" in promotion_fail.stdout

        full_leaderboard_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(full_leaderboard_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "full leaderboard boundary violated" in full_leaderboard_fail.stdout

        pass_art = tmp / "pass_artifact"
        fail_art = tmp / "fail_artifact"
        write_artifact(pass_art, pass_fixture())
        write_artifact(fail_art, mmlu_fail_fixture())
        run([
            sys.executable,
            "scripts/summarize_stage6_p3e_rc_quality_battery.py",
            str(pass_art / "result.json"),
            "--markdown-out",
            str(pass_art / "summary.md"),
            "--strict-exit",
        ])
        pass_report = tmp / "pass_report.md"
        fail_report = tmp / "fail_report.md"
        run([
            sys.executable,
            "scripts/write_stage6_p3e_report.py",
            str(pass_art),
            "--report-out",
            str(pass_report),
            "--date",
            "2026-06-04",
        ])
        run([
            sys.executable,
            "scripts/write_stage6_p3e_report.py",
            str(fail_art),
            "--report-out",
            str(fail_report),
            "--date",
            "2026-06-04",
        ])
        pass_text = pass_report.read_text()
        fail_text = fail_report.read_text()
        assert "Verdict: **PASS**" in pass_text
        assert "RC quality smoke only" in pass_text
        assert "not default" in pass_text
        assert "Verdict: **FAIL**" in fail_text
        assert "Do not bank P3-E" in fail_text

    print("P3-E RC quality-battery evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
