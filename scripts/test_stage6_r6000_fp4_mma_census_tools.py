#!/usr/bin/env python3
"""GPU-free self-test for the R6000 FP4-MMA census summarizer."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.r6000_stage6_fp4_mma_census import _public_kernel_census  # noqa: E402


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
        "schema": "lynn-stage6-r6000-fp4-mma-census-v1",
        "created": "2026-06-04T00:00:00",
        "workspace": "/fixture",
        "min_disk_gib": 500,
        "system": {"disk_workspace": {"stdout_tail": "/dev/root 943G 1G 942G 1% /"}},
        "torch": {
            "json": {
                "torch": "2.10.0+cu128",
                "torch_cuda": "12.8",
                "cuda_available": True,
                "device_name": "NVIDIA RTX PRO 6000 Blackwell",
                "capability": [12, 0],
                "total_memory_gib": 95.5,
            }
        },
        "public_kernel_census": {
            "json": {
                "packages": {"vllm": {"importable": True}, "cutlass": {"importable": False}},
                "explicit_imports": {
                    "vllm.model_executor.kernels.linear.nvfp4.marlin": {"importable": True}
                },
            }
        },
        "contracts": {"p76_cutlass_cute_toolchain": {}},
        "contract_passes": {
            "p76_cutlass_cute_toolchain": True,
            "p79_nvcc_fp4_mma_target_matrix": True,
            "p85_blockscaled_fp4_mma_contract": True,
            "p87_layout_tile_contract": True,
            "p103_fp8_activation_fp4_weight_mma": True,
        },
        "passes": {
            "torch_cuda_recorded": True,
            "blackwell_capability": True,
            "r6000_class_memory": True,
            "disk_headroom": True,
            "public_kernel_census_recorded": True,
            "vllm_nvfp4_or_marlin_seen": True,
            "contract_suite_recorded": True,
            "contract_suite_all_pass": True,
            "all": True,
        },
        "decision": "PASS_R6000_FP4_MMA_BRINGUP",
        "promotion_boundary": {
            "kernel_promoted": False,
            "default_runtime_changed": False,
            "speed_claim": False,
        },
    }


def main() -> int:
    public = _public_kernel_census()
    assert public["process"]["ok"], public
    public_json = public["json"]
    assert "packages" in public_json
    assert "vllm" in public_json["packages"]
    assert "explicit_imports" in public_json
    assert "vllm.model_executor.kernels.linear.nvfp4.marlin" in public_json["explicit_imports"]

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        result = tmp / "result.json"
        summary = tmp / "summary.md"
        result.write_text(json.dumps(fixture()), encoding="utf-8")
        run([
            sys.executable,
            "scripts/summarize_stage6_r6000_fp4_mma_census.py",
            str(result),
            "--markdown-out",
            str(summary),
            "--strict-exit",
        ])
        text = summary.read_text(encoding="utf-8")
        assert "PASS" in text
        assert "RTX PRO 6000" in text
        assert "marlin" in text
        assert "kernel_promoted" in text

    print("Stage 6 R6000 FP4-MMA census tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
