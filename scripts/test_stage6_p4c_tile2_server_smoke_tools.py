#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C tile2 server-smoke evidence tooling."""
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


def _row(text: str, *, degenerate: bool = False) -> dict:
    return {
        "index": 0,
        "prompt": "Answer exactly: 17 + 25 = ?",
        "text": text,
        "normalized_text": text.strip().lower(),
        "degenerate": degenerate,
        "finish_reason": "length",
        "usage": {"completion_tokens": 4},
        "metrics": {"tokens_per_second": 40.0},
        "elapsed_seconds": 0.5,
    }


def _health(*, calls: int = 80, tile: int = 2, release: bool = True) -> dict:
    return {
        "status": "ok",
        "release_decode_shadows_after_prefill": release,
        "skip_reload_when_packed_prefill": False,
        "release_decode_shadows_consumed": release,
        "decode_shadows_currently_released": release,
        "release_reload_count": 1,
        "last_reload_seconds": 23.0,
        "last_release_gib": 60.0 if release else 0.0,
        "runtime": {
            "native_active_moe_backend": "fused_zero_shadow_active_reuse_contract",
            "native_active_moe_layers": "all",
            "native_gateup_tile_inter": str(tile),
            "native_moe_counters": {
                "p4c_active_reuse_contract": {
                    "count_key": "_p4c_fused_zero_shadow_active_reuse_contract_call_count",
                    "last_shapes_key": "_p4c_fused_zero_shadow_active_reuse_contract_last_shapes",
                    "total_calls": calls,
                    "layers_with_calls": 40 if calls else 0,
                    "layers": [{
                        "layer": 0,
                        "count": max(calls // 40, 0),
                        "last_shapes": {
                            "hidden": [1, 2048],
                            "expert_ids": [1, 8],
                            "inter_scratch": [1, 8, 512],
                            "out": [1, 2048],
                            "tile_tokens": 1,
                            "gateup_tile_inter": tile,
                            "down_tile_hidden": 8,
                        },
                    }] if calls else [],
                    "last_shapes": {
                        "hidden": [1, 2048],
                        "expert_ids": [1, 8],
                        "inter_scratch": [1, 8, 512],
                        "out": [1, 2048],
                        "tile_tokens": 1,
                        "gateup_tile_inter": tile,
                        "down_tile_hidden": 8,
                    } if calls else None,
                },
            },
        },
    }


def pass_fixture() -> dict:
    base_row = _row("42")
    cand_row = _row("42")
    return {
        "schema": "lynn-stage6-p4c-tile2-server-smoke-v1",
        "decision": "PASS_P4C_TILE2_SERVER_SMOKE",
        "verdict": "PASS",
        "banked_p4c_tile2_server_smoke": True,
        "banked_default_promotion": False,
        "banked_full_rc_quality": False,
        "model": "/models/demo",
        "preset": "basic",
        "prompt_limit": 2,
        "chat_prompts": 0,
        "max_new": 4,
        "max_seq_len": 2048,
        "gateup_tile_inter": 2,
        "env": {
            "baseline": {"LYNN_NATIVE_ACTIVE_MOE_BACKEND": "triton"},
            "candidate": {
                "LYNN_NATIVE_ACTIVE_MOE_BACKEND": "fused_zero_shadow_active_reuse_contract",
                "LYNN_NATIVE_GATEUP_TILE_INTER": "2",
                "LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL": "1",
            },
        },
        "baseline": {
            "models_ok": True,
            "health_before": _health(calls=0, release=False),
            "health_after": _health(calls=0, release=False),
            "completions": [base_row],
            "chat": [],
        },
        "candidate": {
            "models_ok": True,
            "health_before": _health(calls=0),
            "health_after": _health(calls=80),
            "completions": [cand_row],
            "chat": [],
        },
        "comparisons": {
            "completions": [{
                "index": 0,
                "text_exact": True,
                "baseline_text": "42",
                "candidate_text": "42",
            }],
            "chat": [],
        },
        "candidate_native_counter": {
            "counter_name": "p4c_active_reuse_contract",
            "before_total_calls": 0,
            "after_total_calls": 80,
            "delta_total_calls": 80,
            "after": (_health(calls=80)["runtime"]["native_moe_counters"]["p4c_active_reuse_contract"]),
            "last_shapes": (_health(calls=80)["runtime"]["native_moe_counters"]["p4c_active_reuse_contract"]["last_shapes"]),
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
            "p4c_runtime_predecessor_pass": True,
            "server_surface": True,
            "prompt_count": True,
            "functional_non_degenerate": True,
            "server_text_exact": True,
            "candidate_runtime_env": True,
            "p4c_native_backend_called": True,
            "p4c_tile_recorded": True,
            "p4c_active_reuse_shapes_recorded": True,
            "release_enabled": True,
            "release_consumed": True,
            "decode_shadows_currently_released": True,
            "release_meaningful": True,
            "reload_observed": True,
            "default_promotion_closed": True,
            "full_rc_quality_unbanked": True,
            "all": True,
        },
        "notes": ["synthetic pass fixture"],
    }


def call_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_p4c_tile2_server_smoke"] = False
    data["candidate_native_counter"] = dict(data["candidate_native_counter"])
    data["candidate_native_counter"]["delta_total_calls"] = 0
    data["passes"] = dict(data["passes"])
    data["passes"]["p4c_native_backend_called"] = False
    data["passes"]["all"] = False
    return data


def tile_fail_fixture() -> dict:
    data = pass_fixture()
    data["verdict"] = "FAIL"
    data["banked_p4c_tile2_server_smoke"] = False
    data["candidate_native_counter"] = dict(data["candidate_native_counter"])
    data["candidate_native_counter"]["last_shapes"] = dict(data["candidate_native_counter"]["last_shapes"])
    data["candidate_native_counter"]["last_shapes"]["gateup_tile_inter"] = 8
    data["passes"] = dict(data["passes"])
    data["passes"]["p4c_tile_recorded"] = False
    data["passes"]["all"] = False
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_default_promotion"] = True
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "call_fail": call_fail_fixture(),
            "tile_fail": tile_fail_fixture(),
            "promotion_fail": promotion_fail_fixture(),
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
            "scripts/spark_stage6_p4c_tile2_server_smoke.py",
            "scripts/summarize_stage6_p4c_tile2_server_smoke.py",
        ])
        run(["bash", "-n", "scripts/run_spark_stage6_p4c_tile2_server_smoke.sh"])
        help_proc = run(["scripts/run_spark_stage6_p4c_tile2_server_smoke.sh", "--help"])
        assert "--p4c-runtime-pass" in help_proc.stdout
        assert "--gateup-tile-inter" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout

        bad_preset = run(["scripts/run_spark_stage6_p4c_tile2_server_smoke.sh", "--preset", "nope"], expect=2)
        assert "--preset must be basic or rc-mini" in bad_preset.stderr

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_tile2_server_smoke.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "PASS_P4C_TILE2_SERVER_SMOKE" in pass_summary.stdout
        assert "fused_zero_shadow_active_reuse_contract" in pass_summary.stdout
        assert "P4C native call delta | `80`" in pass_summary.stdout
        assert "Recorded tile_inter | `2`" in pass_summary.stdout
        assert "Banked default promotion | `False`" in pass_summary.stdout

        for name, expected_reason in (
            ("call_fail", "p4c_native_backend_called gate fail"),
            ("tile_fail", "p4c_tile_recorded gate fail"),
            ("promotion_fail", "default promotion boundary violated"),
        ):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_tile2_server_smoke.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert expected_reason in fail.stdout

        empty_prompts = tmp / "empty_prompts.json"
        empty_prompts.write_text("[]\n", encoding="utf-8")
        empty_proc = run([
            sys.executable,
            "scripts/spark_stage6_p4c_tile2_server_smoke.py",
            "--prompts-json",
            str(empty_prompts),
        ], expect=None)
        if empty_proc.returncode == 0:
            raise AssertionError("empty prompt list unexpectedly succeeded")
        assert "prompt list must be non-empty" in (empty_proc.stderr + empty_proc.stdout)

    print("P4C tile2 server smoke evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
