#!/usr/bin/env python3
"""GPU-free checks for the Stage 6 R5-A layout bridge tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "r5a_stage6_per16_layout_bridge.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5a_layout_bridge.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5a_layout_bridge.sh"
CONTRACT = ROOT / "reports" / "stage6" / "R6000_GROUPED_MOE_FP4_MMA_POC_CONTRACT_20260604.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5a-per16-layout-bridge-v1",
        "decision": "PASS_R5A_LAYOUT_BRIDGE_E8M0_REPACK_REQUIRED",
        "passes": {
            "power2_padded_per16_layout_ok": True,
            "fold_pair_group32_supported": False,
            "current_lynn_e4m3_scales_zero_copy_supported": False,
            "banked_layout_bridge": True,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": promoted,
            "banked_default_promotion": False,
            "all": True,
        },
        "cases": [
            {
                "scale_case": "power2",
                "shape": {"M": 1, "N": 1024, "K": 2048},
                "candidates": [
                    {
                        "candidate": "padded_per16_group32",
                        "metrics": {"rel_l2": 0.0, "cosine": 1.0},
                        "timing_ms": {"median_ms": 0.1},
                        "bytes": {"packed_ratio_vs_original": 2.0, "scale_ratio_vs_original": 0.25},
                    }
                ],
            }
        ],
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        BENCH: [
            "lynn-stage6-r5a-per16-layout-bridge-v1",
            "fold_pair_group32",
            "padded_per16_group32",
            "current_lynn_e4m3_scales_zero_copy_supported",
            "banked_kernel_speed",
            "PASS_R5A_LAYOUT_BRIDGE_E8M0_REPACK_REQUIRED",
        ],
        SUMMARY: [
            "banked_grouped_moe_fp4_mma_poc",
            "banked_kernel_speed",
            "strict-exit",
            "PASS_R5A_LAYOUT_BRIDGE",
        ],
        WRAPPER: [
            "benchmarks/r5a_stage6_per16_layout_bridge.py",
            "summarize_stage6_r5a_layout_bridge.py",
            "TORCH_CUDA_ARCH_LIST",
            "nvidia_smi_before.txt",
        ],
        CONTRACT: [
            "R5-A layout bridge",
            "The first implementation target is **R5-A**",
            "banked_layout_bridge",
        ],
    }
    for path, needles in checks.items():
        text = _read(path)
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
        if "Layout bridge banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing banked layout bridge row")
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
        print("Stage 6 R5-A layout bridge tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-A layout bridge tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
