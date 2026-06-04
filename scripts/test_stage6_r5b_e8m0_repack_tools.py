#!/usr/bin/env python3
"""GPU-free checks for the Stage 6 R5-B e8m0 repack tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "r5b_stage6_e8m0_repack_bridge.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5b_e8m0_repack.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5b_e8m0_repack.sh"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5b-e8m0-repack-bridge-v1",
        "decision": "PASS_R5B_E8M0_REPACK_NUMERIC",
        "passes": {
            "e4m3_like_e8m0_repack_numeric_ok": True,
            "banked_repack_numeric": True,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": promoted,
            "banked_default_promotion": False,
            "all": True,
        },
        "cases": [{
            "shape": {"M": 16, "N": 1024, "K": 2048},
            "candidates": [{
                "candidate": "e8m0_repack_nearest",
                "metrics": {"rel_l2": 0.02, "cosine": 0.999},
                "timing_ms": {"median_ms": 0.1},
                "act_repack": {"value_rel_l2": 0.01},
                "weight_repack": {"value_rel_l2": 0.01},
            }],
        }],
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        BENCH: [
            "lynn-stage6-r5b-e8m0-repack-bridge-v1",
            "_requantize_per16_to_e8m0",
            "banked_repack_numeric",
            "banked_grouped_moe_fp4_mma_poc",
            "PASS_R5B_E8M0_REPACK_NUMERIC",
        ],
        SUMMARY: [
            "banked_repack_numeric",
            "banked_grouped_moe_fp4_mma_poc",
            "strict-exit",
            "PASS_R5B_E8M0_REPACK",
        ],
        WRAPPER: [
            "benchmarks/r5b_stage6_e8m0_repack_bridge.py",
            "summarize_stage6_r5b_e8m0_repack.py",
            "TORCH_CUDA_ARCH_LIST",
            "nvidia_smi_before.txt",
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
        if "Repack numeric banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing repack numeric row")
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
        print("Stage 6 R5-B e8m0 repack tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-B e8m0 repack tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
