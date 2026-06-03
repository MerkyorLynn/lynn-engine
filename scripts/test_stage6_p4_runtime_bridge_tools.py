#!/usr/bin/env python3
"""Local self-test for Stage 6 P4 runtime bridge preflight tooling."""
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
        "schema": "lynn-stage6-p4-runtime-bridge-preflight-v1",
        "decision": "PASS_TWO_STAGE_RUNTIME_BRIDGE",
        "model": "/models/demo",
        "layer": 0,
        "prompt": "fixture",
        "expected_backend": "fused_zero_shadow_out_contract",
        "expected_reference": "caller-owned two-stage packed-NVFP4 active-MoE reference",
        "native_layer_selected_for_candidate": True,
        "native_backend_call_count": {
            "key": "_p4_fused_zero_shadow_out_contract_call_count",
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
            "mlp.experts._gate_up_global_scale": {"shape": [1], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
            "mlp.experts._down_packed": {"shape": [256, 2048, 256], "dtype": "torch.uint8", "bytes": 134217728, "contiguous": True},
            "mlp.experts._down_scale": {"shape": [256, 2048, 32], "dtype": "torch.float32", "bytes": 67108864, "contiguous": True},
            "mlp.experts._down_global_scale": {"shape": [1], "dtype": "torch.float32", "bytes": 4, "contiguous": True},
        },
        "candidate_error": None,
        "candidate": {
            "backend": "fused_zero_shadow_out_contract",
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


def shadow_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_runtime_bridge_preflight"] = False
    data["active_shadow_keys_present_after_delete"] = ["mlp.experts.gate_up_proj"]
    data["passes"] = dict(data["passes"])
    data["passes"]["active_shadows_removed"] = False
    data["passes"]["all"] = False
    return data


def promotion_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_fused_kernel"] = True
    return data


def packed_manifest_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_runtime_bridge_preflight"] = False
    data["packed_manifest_before_candidate"] = dict(data["packed_manifest_before_candidate"])
    data["packed_manifest_before_candidate"].pop("mlp.experts._down_scale")
    data["passes"] = dict(data["passes"])
    data["passes"]["packed_tensors_present"] = False
    data["passes"]["all"] = False
    return data


def scratch_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_runtime_bridge_preflight"] = False
    data["active_scratch_manifest"] = dict(data["active_scratch_manifest"])
    data["active_scratch_manifest"]["mlp.experts._active_inter_scratch"] = {
        "shape": [4, 512],
        "dtype": "torch.bfloat16",
        "bytes": 4096,
        "contiguous": True,
    }
    data["passes"] = dict(data["passes"])
    data["passes"]["active_scratch_present"] = False
    data["passes"]["all"] = False
    return data


def native_selection_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_runtime_bridge_preflight"] = False
    data["native_layer_selected_for_candidate"] = False
    data["passes"] = dict(data["passes"])
    data["passes"]["native_layer_selected"] = False
    data["passes"]["all"] = False
    return data


def native_call_fail_fixture() -> dict:
    data = pass_fixture()
    data["banked_runtime_bridge_preflight"] = False
    data["native_backend_call_count"] = dict(data["native_backend_call_count"])
    data["native_backend_call_count"]["after"] = 0
    data["native_backend_call_count"]["delta"] = 0
    data["passes"] = dict(data["passes"])
    data["passes"]["native_backend_called"] = False
    data["passes"]["all"] = False
    return data


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        shadow_fail_json = tmp / "shadow_fail.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        packed_fail_json = tmp / "packed_fail.json"
        scratch_fail_json = tmp / "scratch_fail.json"
        native_selection_fail_json = tmp / "native_selection_fail.json"
        native_call_fail_json = tmp / "native_call_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(shadow_fail_json, shadow_fail_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())
        write_json(packed_fail_json, packed_manifest_fail_fixture())
        write_json(scratch_fail_json, scratch_fail_fixture())
        write_json(native_selection_fail_json, native_selection_fail_fixture())
        write_json(native_call_fail_json, native_call_fail_fixture())

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p4_runtime_bridge_preflight.py",
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
        ])
        run(["bash", "-n", "scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh"])
        help_proc = run(["scripts/run_spark_stage6_p4_runtime_bridge_preflight.sh", "--help"])
        assert "--rel-l2-threshold" in help_proc.stdout
        assert "LYNN_STAGE6_P4_MODEL" in help_proc.stdout
        assert "summarize_stage6_p4_runtime_bridge_preflight.py" not in help_proc.stderr

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert "Active scratch present | `True`" in pass_summary.stdout
        assert "Native layer selected | `True`" in pass_summary.stdout
        assert "Native backend call delta | `1`" in pass_summary.stdout
        assert "Active shadows removed | `True`" in pass_summary.stdout
        assert "Candidate output returned | `True`" in pass_summary.stdout

        shadow_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(shadow_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "runtime bridge preflight was not banked" in shadow_fail.stdout

        promotion_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(promotion_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "fused-kernel promotion boundary violated" in promotion_fail.stdout

        packed_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(packed_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "runtime bridge preflight was not banked" in packed_fail.stdout

        scratch_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(scratch_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "runtime bridge preflight was not banked" in scratch_fail.stdout

        native_selection_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(native_selection_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "runtime bridge preflight was not banked" in native_selection_fail.stdout

        native_call_fail = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(native_call_fail_json),
            "--strict-exit",
        ], expect=2)
        assert "runtime bridge preflight was not banked" in native_call_fail.stdout

    print("P4 runtime bridge tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
