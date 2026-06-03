#!/usr/bin/env python3
"""Local self-test for Stage 6 P4B single-kernel evidence tooling."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAIL_LOUD_TEXT = (
    "P4B single-kernel fused zero-shadow contract is not implemented yet; "
    "do not bank fused-kernel speed or promote this backend"
)


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
        "schema": "lynn-stage6-p4b-single-kernel-preflight-v1",
        "decision": "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT",
        "symbol": "active_moe_fused_zero_shadow_single_kernel_contract",
        "expected_behavior": "fail-loud single-kernel contract; no fused implementation is banked",
        "banked_single_kernel_contract_preflight": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "device_name": "fixture-gpu",
        "capability": [12, 1],
        "torch_version": "fixture",
        "torch_cuda": "fixture",
        "build_dir": "/tmp/fixture",
        "elapsed_s": 1.23,
        "call_returned": False,
        "call_error_tail": FAIL_LOUD_TEXT,
        "fail_loud_needles": [
            "P4B single-kernel fused zero-shadow contract is not implemented yet",
            "do not bank fused-kernel speed or promote this backend",
        ],
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
            "activation_io_names": ["hidden", "expert_ids", "routing_weights", "out"],
            "packed_weight_bytes": 18874372,
            "activation_io_bytes": 16512,
            "bf16_shadow_equivalent_bytes": 50331648,
            "packed_vs_bf16_shadow_ratio": 0.3750000794728597,
            "forbidden_shadow_tensor_names": [],
            "zero_shadow_abi": True,
            "packed_byte_budget": True,
            "no_inter_scratch_abi": True,
        },
        "passes": {
            "extension_loaded": True,
            "symbol_present": True,
            "call_returned_false": True,
            "fail_loud_not_implemented": True,
            "zero_shadow_abi": True,
            "packed_byte_budget": True,
            "no_inter_scratch_abi": True,
            "all": True,
        },
    }


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def call_returned_fail_fixture() -> dict:
    data = pass_fixture()
    data["decision"] = "FAIL_SINGLE_KERNEL_CONTRACT"
    data["banked_single_kernel_contract_preflight"] = False
    data["call_returned"] = True
    data.pop("call_error_tail", None)
    data["passes"] = dict(data["passes"])
    data["passes"]["call_returned_false"] = False
    data["passes"]["fail_loud_not_implemented"] = False
    data["passes"]["all"] = False
    return data


def inter_scratch_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_single_kernel_contract_preflight"] = False
    data["tensor_manifest"] = dict(data["tensor_manifest"])
    data["tensor_manifest"]["inter_scratch"] = {
        "shape": [2, 8, 512],
        "dtype": "torch.bfloat16",
        "bytes": 16384,
        "contiguous": True,
    }
    data["byte_budget"] = dict(data["byte_budget"])
    data["byte_budget"]["no_inter_scratch_abi"] = False
    data["byte_budget"]["zero_shadow_abi"] = False
    data["byte_budget"]["forbidden_shadow_tensor_names"] = ["inter_scratch"]
    data["passes"] = dict(data["passes"])
    data["passes"]["no_inter_scratch_abi"] = False
    data["passes"]["zero_shadow_abi"] = False
    data["passes"]["all"] = False
    return data


def packed_budget_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_single_kernel_contract_preflight"] = False
    data["byte_budget"] = dict(data["byte_budget"])
    data["byte_budget"]["packed_byte_budget"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["packed_byte_budget"] = False
    data["passes"]["all"] = False
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        call_returned_fail_json = tmp / "call_returned_fail.json"
        inter_scratch_fail_json = tmp / "inter_scratch_fail.json"
        packed_budget_fail_json = tmp / "packed_budget_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())
        write_json(call_returned_fail_json, call_returned_fail_fixture())
        write_json(inter_scratch_fail_json, inter_scratch_fail_fixture())
        write_json(packed_budget_fail_json, packed_budget_fail_fixture())

        run(["bash", "-n", "scripts/run_spark_stage6_p4b_single_kernel_preflight.sh"])
        help_proc = run(["scripts/run_spark_stage6_p4b_single_kernel_preflight.sh", "--help"])
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "LYNN_STAGE6_EXPECT_MANIFEST" in help_proc.stdout
        assert "--no-strict" in help_proc.stdout

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p4b_single_kernel_preflight.py",
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert "Call returned false | `True`" in pass_summary.stdout
        assert "No inter_scratch ABI | `True`" in pass_summary.stdout

        promotion_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            str(promotion_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "fused-kernel promotion boundary violated" in promotion_fail.stdout

        call_returned_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            str(call_returned_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "single-kernel contract preflight was not banked" in call_returned_fail.stdout

        inter_scratch_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            str(inter_scratch_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "single-kernel contract preflight was not banked" in inter_scratch_fail.stdout

        packed_budget_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4b_single_kernel_preflight.py",
            str(packed_budget_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "single-kernel contract preflight was not banked" in packed_budget_fail.stdout

    print("P4B single-kernel evidence tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
