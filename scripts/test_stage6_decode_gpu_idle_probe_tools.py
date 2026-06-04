#!/usr/bin/env python3
"""GPU-free self-test for Stage 6 decode GPU-idle probe tooling."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"command failed with {proc.returncode}: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def fixture() -> dict:
    return {
        "schema": "lynn-stage6-decode-gpu-idle-probe-v1",
        "created": "2026-06-04T00:00:00",
        "model": "/fixture/model",
        "prompt": "fixture",
        "env_summary": {},
        "runs": {
            "short": {"tokens": 48, "decode_tps_runner": 44.0},
            "long": {"tokens": 96, "decode_tps_runner": 44.5},
        },
        "delta": {
            "tokens_delta": 48,
            "wall_ms_per_token": 22.5,
            "cuda_kernel_busy_ms_per_token": 13.5,
            "host_gap_or_idle_ms_per_token_est": 9.0,
            "gpu_busy_ratio_est": 0.6,
            "host_gap_fraction_est": 0.4,
            "cuda_launches_per_token": 1527.0,
            "cpu_cuda_api_ms_per_token": 2.0,
            "cpu_cuda_api_calls_per_token": 1527.0,
            "compiled_loop_roi_signal": "GO_COMPILED_LOOP_PROTOTYPE",
        },
        "passes": {
            "token_delta_positive": True,
            "launches_recorded": True,
            "timing_recorded": True,
            "idle_estimate_recorded": True,
            "all": True,
        },
        "decision": "PASS_DECODE_GPU_IDLE_PROBE_RECORDED",
        "promotion_boundary": {
            "speed_promotion": False,
            "compiled_loop_default": False,
            "cuda_graph_route": False,
        },
        "caveat": "fixture caveat",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        result = tmp / "result.json"
        summary = tmp / "summary.md"
        result.write_text(json.dumps(fixture()), encoding="utf-8")
        run([
            sys.executable,
            "scripts/summarize_stage6_decode_gpu_idle_probe.py",
            str(result),
            "--markdown-out",
            str(summary),
            "--strict-exit",
        ])
        text = summary.read_text(encoding="utf-8")
        assert "PASS" in text
        assert "GO_COMPILED_LOOP_PROTOTYPE" in text
        assert "Speed promotion | `False`" in text

    print("Stage 6 decode GPU-idle probe tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
