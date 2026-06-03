#!/usr/bin/env python3
"""Local self-test for Stage 6 P4 native ABI evidence tooling."""
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


def pass_fixture() -> dict:
    return {
        "schema": "lynn-stage6-p4-native-fused-moe-abi-preflight-v1",
        "decision": "PASS_TWO_STAGE_REFERENCE_CONTRACT",
        "symbol": "active_moe_fused_zero_shadow_out_contract",
        "expected_reference": "caller-owned two-stage packed-NVFP4 active-MoE reference",
        "banked_native_abi_preflight": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "torch_version": "fixture",
        "torch_cuda": "fixture",
        "build_dir": "/tmp/fixture",
        "elapsed_s": 1.23,
        "tensor_manifest": {
            "hidden": {"shape": [2, 2048], "dtype": "torch.bfloat16", "bytes": 8192, "contiguous": True},
            "expert_ids": {"shape": [2, 8], "dtype": "torch.int32", "bytes": 64, "contiguous": True},
            "routing_weights": {"shape": [2, 8], "dtype": "torch.float32", "bytes": 64, "contiguous": True},
            "gate_up_packed": {"shape": [8, 1024, 1024], "dtype": "torch.uint8", "bytes": 8388608, "contiguous": True},
            "gate_up_scale": {"shape": [8, 1024, 128], "dtype": "torch.float32", "bytes": 4194304, "contiguous": True},
            "gate_up_global_scale": {"shape": [1], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
            "down_packed": {"shape": [8, 2048, 256], "dtype": "torch.uint8", "bytes": 4194304, "contiguous": True},
            "down_scale": {"shape": [8, 2048, 32], "dtype": "torch.float32", "bytes": 2097152, "contiguous": True},
            "down_global_scale": {"shape": [1], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
            "inter_scratch": {"shape": [2, 8, 512], "dtype": "torch.bfloat16", "bytes": 16384, "contiguous": True},
            "out": {"shape": [2, 2048], "dtype": "torch.bfloat16", "bytes": 8192, "contiguous": True},
        },
        "byte_budget": {
            "packed_weight_names": [
                "gate_up_packed",
                "gate_up_scale",
                "gate_up_global_scale",
                "down_packed",
                "down_scale",
                "down_global_scale",
            ],
            "activation_io_names": ["hidden", "expert_ids", "routing_weights", "inter_scratch", "out"],
            "packed_weight_bytes": 18874372,
            "activation_io_bytes": 16640,
            "bf16_shadow_equivalent_bytes": 50331648,
            "packed_vs_bf16_shadow_ratio": 0.3750000794728597,
            "forbidden_shadow_tensor_names": [],
            "zero_shadow_abi": True,
            "packed_byte_budget": True,
        },
        "output": {"shape": [2, 2048], "dtype": "torch.bfloat16", "finite": True, "norm": 0.0},
        "passes": {
            "extension_loaded": True,
            "symbol_present": True,
            "reference_output_returned": True,
            "output_finite": True,
            "zero_shadow_abi": True,
            "packed_byte_budget": True,
            "all": True,
        },
    }


def symbol_fail_fixture() -> dict:
    data = pass_fixture()
    data["decision"] = "BLOCKED_SYMBOL_MISSING"
    data["passes"] = dict(data["passes"])
    data["passes"]["symbol_present"] = False
    data["passes"]["reference_output_returned"] = False
    data["passes"]["output_finite"] = False
    data["passes"]["all"] = False
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def shadow_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_native_abi_preflight"] = False
    data["byte_budget"] = dict(data["byte_budget"])
    data["byte_budget"]["forbidden_shadow_tensor_names"] = ["mlp.experts.gate_up_proj.weight"]
    data["byte_budget"]["zero_shadow_abi"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["zero_shadow_abi"] = False
    data["passes"]["all"] = False
    return data


def write_artifact(path: Path, data: dict, *, manifest_match: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "result.json", data)
    (path / "expected_git_head.txt").write_text("expected-head\n", encoding="utf-8")
    (path / "git_head.txt").write_text("remote-head\n", encoding="utf-8")
    (path / "head_check.txt").write_text("remote manifest ok\n", encoding="utf-8")
    expected_manifest = "abc scripts/run_spark_stage6_p4_native_abi_preflight.sh\n"
    actual_manifest = expected_manifest if manifest_match else "def scripts/run_spark_stage6_p4_native_abi_preflight.sh\n"
    (path / "expected_provenance_manifest.txt").write_text(expected_manifest, encoding="utf-8")
    (path / "provenance_manifest.txt").write_text(actual_manifest, encoding="utf-8")
    (path / "git_status.txt").write_text("", encoding="utf-8")
    (path / "docker_exit_code.txt").write_text("0\n", encoding="utf-8")
    (path / "nvidia_smi_before.txt").write_text("fixture-gpu, 0 %, 1024 MiB, 122880 MiB\n", encoding="utf-8")
    (path / "nvidia_smi_after.txt").write_text("fixture-gpu, 0 %, 1024 MiB, 122880 MiB\n", encoding="utf-8")
    (path / "run.log").write_text("fixture run log tail\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        symbol_fail_json = tmp / "symbol_fail.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        shadow_fail_json = tmp / "shadow_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(symbol_fail_json, symbol_fail_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())
        write_json(shadow_fail_json, shadow_fail_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p4_native_abi_preflight.sh"])
        help_proc = run(["scripts/run_spark_stage6_p4_native_abi_preflight.sh", "--help"])
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout
        assert "--no-strict" in help_proc.stdout

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p4_native_abi_preflight.py",
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            "scripts/write_stage6_p4_native_abi_report.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert "Zero-shadow ABI | `True`" in pass_summary.stdout
        assert "Packed byte budget | `True`" in pass_summary.stdout

        fail_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            str(symbol_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "symbol_present gate fail" in fail_summary.stdout

        promotion_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            str(promotion_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "fused-kernel promotion boundary violated" in promotion_fail.stdout

        shadow_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            str(shadow_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "ABI preflight was not banked" in shadow_fail.stdout

        artifact = tmp / "artifact"
        write_artifact(artifact, pass_fixture())
        run([
            sys.executable,
            "scripts/write_stage6_p4_native_abi_report.py",
            str(artifact),
            "--report-out",
            str(tmp / "report.md"),
            "--date",
            "2026-06-04",
        ])
        report = (tmp / "report.md").read_text(encoding="utf-8")
        assert "Verdict: **PASS**" in report
        assert "Bank P4 native ABI preflight only" in report
        assert "Banked fused kernel | `False`" in report
        assert "Zero-shadow ABI | `true`" in report
        assert "Packed byte budget | `true`" in report

    print("P4 native ABI evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
