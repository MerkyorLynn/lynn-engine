#!/usr/bin/env python3
"""Local self-test for Stage 6 P4C active-reuse runtime bridge tooling."""
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
        "schema": "lynn-stage6-p4c-active-reuse-runtime-bridge-preflight-v1",
        "decision": "PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE",
        "model": "/models/demo",
        "layer": 0,
        "prompt": "fixture",
        "expected_backend": "fused_zero_shadow_active_reuse_contract",
        "expected_reference": "P4C caller-owned active-reuse two-phase packed-NVFP4 active-MoE contract",
        "native_layer_selected_for_candidate": True,
        "native_backend_call_count": {
            "key": "_p4c_fused_zero_shadow_active_reuse_contract_call_count",
            "before": 0,
            "after": 1,
            "delta": 1,
            "last_shapes": {
                "hidden": [1, 2048],
                "expert_ids": [1, 8],
                "inter_scratch": [1, 8, 512],
                "out": [1, 2048],
            },
        },
        "banked_runtime_bridge_preflight": True,
        "banked_p4c_active_reuse_runtime_bridge": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "baseline": {"backend": "triton", "output_shape": [1, 1, 2048], "output_dtype": "torch.bfloat16", "norm": 12.5},
        "removed_active_shadows": {
            "mlp.experts.gate_up_proj": {"shape": [256, 1024, 2048], "dtype": "torch.bfloat16", "bytes": 1073741824},
            "mlp.experts.down_proj": {"shape": [256, 2048, 512], "dtype": "torch.bfloat16", "bytes": 536870912},
        },
        "active_shadow_keys_present_after_delete": [],
        "bf16_active_shadow_aliases_after_delete": {},
        "active_scratch_manifest": {
            "mlp.experts._active_inter_scratch": {"shape": [8, 512], "dtype": "torch.bfloat16", "bytes": 8192, "contiguous": True},
            "mlp.experts._active_out_scratch": {"shape": [2048], "dtype": "torch.bfloat16", "bytes": 4096, "contiguous": True},
        },
        "packed_manifest_before_candidate": {
            "mlp.experts._gate_up_packed": {"shape": [256, 1024, 1024], "dtype": "torch.uint8", "bytes": 268435456, "contiguous": True},
            "mlp.experts._gate_up_scale": {"shape": [256, 1024, 128], "dtype": "torch.float32", "bytes": 134217728, "contiguous": True},
            "mlp.experts._gate_up_global_scale": {"shape": [], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
            "mlp.experts._down_packed": {"shape": [256, 2048, 256], "dtype": "torch.uint8", "bytes": 134217728, "contiguous": True},
            "mlp.experts._down_scale": {"shape": [256, 2048, 32], "dtype": "torch.float32", "bytes": 67108864, "contiguous": True},
            "mlp.experts._down_global_scale": {"shape": [], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
        },
        "candidate_error": None,
        "candidate": {
            "backend": "fused_zero_shadow_active_reuse_contract",
            "output_shape": [1, 1, 2048],
            "output_dtype": "torch.bfloat16",
            "norm": 12.49,
            "finite": True,
            "max_abs_diff_vs_baseline": 0.25,
            "mean_abs_diff_vs_baseline": 0.01,
            "rel_l2_vs_baseline": 0.01,
        },
        "passes": {
            "cuda_available": True,
            "native_layer_selected": True,
            "native_backend_called": True,
            "baseline_triton_nonzero": True,
            "baseline_shape_dtype": True,
            "packed_tensors_present": True,
            "active_scratch_present": True,
            "active_shadows_removed": True,
            "candidate_output_returned": True,
            "candidate_shape_dtype": True,
            "candidate_numeric_vs_triton": True,
            "fused_kernel_unbanked": True,
            "default_promotion_closed": True,
            "all": True,
        },
    }


def fail_fixture_without_active_reuse() -> dict:
    data = pass_fixture()
    data["banked_p4c_active_reuse_runtime_bridge"] = False
    data["native_backend_call_count"] = dict(data["native_backend_call_count"])
    data["native_backend_call_count"]["last_shapes"] = {
        "hidden": [1, 2048],
        "expert_ids": [1, 8],
        "out": [1, 2048],
    }
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def backend_fail_fixture() -> dict:
    data = pass_fixture()
    data["expected_backend"] = "fused_zero_shadow_out_contract"
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        fixtures = {
            "pass": pass_fixture(),
            "no_active_reuse": fail_fixture_without_active_reuse(),
            "promotion_fail": promotion_fail_fixture(),
            "backend_fail": backend_fail_fixture(),
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
            "scripts/spark_stage6_p4c_runtime_bridge_preflight.py",
            "scripts/summarize_stage6_p4c_runtime_bridge_preflight.py",
        ])
        run(["bash", "-n", "scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh"])
        help_proc = run(["scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh", "--help"])
        assert "LYNN_STAGE6_P4_MODEL" in help_proc.stdout
        assert "--allow-provenance-mismatch" in help_proc.stdout
        assert "spark_stage6_p4c_runtime_bridge_preflight.py" not in help_proc.stderr

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4c_runtime_bridge_preflight.py",
            str(paths["pass"]),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE" in pass_summary.stdout
        assert "fused_zero_shadow_active_reuse_contract" in pass_summary.stdout
        assert "P4C Boundary" in pass_summary.stdout
        assert "banked_p4c_active_reuse_runtime_bridge=true" in pass_summary.stdout

        for name in ("no_active_reuse", "promotion_fail", "backend_fail"):
            fail = run([
                sys.executable,
                "scripts/summarize_stage6_p4c_runtime_bridge_preflight.py",
                str(paths[name]),
                "--strict-exit",
            ], expect=2)
            assert "Verdict | **FAIL**" in fail.stdout

    print("P4C active-reuse runtime bridge tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
