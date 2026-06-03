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
        "decision": "PASS_RUNTIME_BRIDGE_CONTRACT",
        "model": "/models/demo",
        "layer": 0,
        "prompt": "fixture",
        "expected_backend": "fused_zero_shadow_out_contract",
        "expected_error": "P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet",
        "banked_runtime_bridge_preflight": True,
        "banked_fused_kernel": False,
        "banked_default_promotion": False,
        "baseline": {"backend": "triton", "output_shape": [1, 1, 2048], "output_dtype": "torch.bfloat16", "norm": 12.5},
        "removed_active_shadows": {
            "mlp.experts.gate_up_proj": {"shape": [256, 1024, 2048], "dtype": "torch.bfloat16", "bytes": 1073741824},
            "mlp.experts.down_proj": {"shape": [256, 2048, 512], "dtype": "torch.bfloat16", "bytes": 536870912},
        },
        "active_shadow_keys_present_after_delete": [],
        "packed_manifest_before_candidate": {
            "mlp.experts._gate_up_packed": {"shape": [256, 1024, 1024], "dtype": "torch.uint8", "bytes": 268435456},
            "mlp.experts._down_packed": {"shape": [256, 2048, 256], "dtype": "torch.uint8", "bytes": 134217728},
        },
        "candidate_error": {
            "type": "RuntimeError",
            "message": "P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet",
        },
        "passes": {
            "cuda_available": True,
            "baseline_triton_nonzero": True,
            "packed_tensors_present": True,
            "active_shadows_removed": True,
            "candidate_fail_loud": True,
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


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        pass_json = tmp / "pass.json"
        shadow_fail_json = tmp / "shadow_fail.json"
        promotion_fail_json = tmp / "promotion_fail.json"
        write_json(pass_json, pass_fixture())
        write_json(shadow_fail_json, shadow_fail_fixture())
        write_json(promotion_fail_json, promotion_fail_fixture())

        run([
            sys.executable,
            "-m",
            "py_compile",
            "scripts/spark_stage6_p4_runtime_bridge_preflight.py",
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
        ])

        pass_summary = run([
            sys.executable,
            "scripts/summarize_stage6_p4_runtime_bridge_preflight.py",
            str(pass_json),
            "--strict-exit",
        ])
        assert "Verdict | **PASS**" in pass_summary.stdout
        assert "Banked fused kernel | `False`" in pass_summary.stdout
        assert "Active shadows removed | `True`" in pass_summary.stdout

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

    print("P4 runtime bridge tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
