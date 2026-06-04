#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C CUTLASS UE4M3 census tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c_cutlass_ue4m3_census.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c_cutlass_ue4m3_census.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c_cutlass_ue4m3_census.sh"
DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5c-cutlass-ue4m3-census-v1",
        "decision": "PASS_R5C_NVF4_UE4M3_CUTLASS_ABI",
        "cutlass_dir": "/fixture/cutlass",
        "git": {
            "head": {"ok": True, "stdout": "abc123"},
            "branch": {"ok": True, "stdout": "main"},
        },
        "passes": {
            "banked_cutlass_abi": True,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": promoted,
            "banked_default_promotion": False,
            "sm120_ue4m3_macro_seen": True,
            "scale_format_ue4m3_seen": True,
            "scale_type_ue4m3_seen": True,
            "mxf4_e2m1_format_seen": True,
            "sm120_e2m1_ue4m3_specialization_seen": True,
            "sm120_mxf4nvf4_ue4m3_asm_seen": True,
            "expected_examples_seen": True,
            "sm120_tests_seen": True,
            "all": True,
        },
        "token_hits": {
            "sm120_mma": [{
                "line": 1,
                "text": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3",
            }],
        },
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c-cutlass-ue4m3-census-v1",
            "CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED",
            "SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t",
            "mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
            "banked_cutlass_abi",
            "banked_grouped_moe_fp4_mma_poc",
            "banked_kernel_speed",
        ],
        SUMMARY: [
            "PASS_R5C_NVF4_UE4M3_CUTLASS_ABI",
            "banked_cutlass_abi",
            "strict-exit",
            "Grouped-MoE FP4-MMA POC banked",
        ],
        WRAPPER: [
            "r6000_stage6_r5c_cutlass_ue4m3_census.py",
            "summarize_stage6_r5c_cutlass_ue4m3_census.py",
            "CUTLASS_DIR",
            "nvidia_smi_before.txt",
        ],
        DOC: [
            "R5-C",
            "NVF4 + UE4M3",
            "PASS_R5C_NVF4_UE4M3_CUTLASS_ABI",
            "banked_cutlass_abi=true",
            "banked_grouped_moe_fp4_mma_poc=false",
            "banked_kernel_speed=false",
            "R5-C1",
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

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.json"
        bad = Path(tmp) / "bad.json"
        md = Path(tmp) / "summary.md"
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
        if "CUTLASS ABI banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing ABI banked row")
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
        print("Stage 6 R5-C CUTLASS UE4M3 tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C CUTLASS UE4M3 tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
