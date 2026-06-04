#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C1 CUTLASS numeric-smoke tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c1_cutlass_numeric_smoke.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c1_cutlass_numeric_smoke.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c1_cutlass_numeric_smoke.sh"
DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(speed_promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5c1-cutlass-numeric-smoke-v1",
        "decision": "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE",
        "cutlass_dir": "/fixture/cutlass",
        "example": "/fixture/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu",
        "git": {
            "head": {"ok": True, "stdout_tail": "abc123"},
            "branch": {"ok": True, "stdout_tail": "main"},
        },
        "shape": {"m": 256, "n": 128, "k": 256, "groups": 2, "iterations": 1},
        "build_result": {"atomic_scope_patch": {"applied": True, "restored": True}},
        "run_parse": {
            "avg_runtime_ms": [0.02, 0.03],
            "tflops": [1.0, 2.0],
        },
        "passes": {
            "build_invoked": True,
            "build_succeeded": True,
            "banked_numeric_smoke": True,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": speed_promoted,
            "banked_default_promotion": False,
            "cooperative_passed": True,
            "pingpong_passed": True,
            "host_reference_seen": True,
            "dispositions_passed_count_ge_2": True,
            "no_noop_device_gate": True,
        },
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c1-cutlass-numeric-smoke-v1",
            "79d_blackwell_geforce_nvfp4_grouped_gemm",
            "Host-side verification is now running",
            "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE",
            "banked_numeric_smoke",
            "banked_grouped_moe_fp4_mma_poc",
            "banked_kernel_speed",
            "ATOMIC_NEW",
        ],
        SUMMARY: [
            "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE",
            "Numeric smoke banked",
            "Grouped-MoE FP4-MMA POC banked",
            "strict-exit",
        ],
        WRAPPER: [
            "r6000_stage6_r5c1_cutlass_numeric_smoke.py",
            "summarize_stage6_r5c1_cutlass_numeric_smoke.py",
            "nvidia_smi_before.txt",
            "CUTLASS_DIR",
        ],
        DOC: [
            "R5-C1",
            "minimal numeric GEMM smoke",
            "PASS_R5C1_CUTLASS_NVF4_UE4M3_NUMERIC_SMOKE",
            "banked_numeric_smoke=true",
            "banked_kernel_speed=false",
        ],
    }
    for path, needles in checks.items():
        try:
            text = _read(path)
        except AssertionError as exc:
            failures.append(str(exc))
            continue
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)}: missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        good = tmp / "good.json"
        bad = tmp / "bad.json"
        md = tmp / "summary.md"
        good.write_text(json.dumps(_fixture(False)), encoding="utf-8")
        bad.write_text(json.dumps(_fixture(True)), encoding="utf-8")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(good), "--markdown-out", str(md), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"summary strict good fixture failed: {ok.stderr or ok.stdout}")
        if "Numeric smoke banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing numeric smoke banked row")
        fail = subprocess.run(
            [sys.executable, str(SUMMARY), str(bad), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if fail.returncode == 0:
            failures.append("summary strict bad fixture unexpectedly passed")

    if failures:
        print("Stage 6 R5-C1 CUTLASS numeric-smoke tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C1 CUTLASS numeric-smoke tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
