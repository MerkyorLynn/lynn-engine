#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C2 MoE-shape census tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c2_moe_shape_census.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c2_moe_shape_census.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c2_moe_shape_census.sh"
DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5c2-moe-shape-census-v1",
        "decision": "PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED",
        "cutlass_dir": "/fixture/cutlass",
        "git": {
            "head": {"stdout": "abc123"},
            "branch": {"stdout": "main"},
        },
        "passes": {
            "banked_moe_shape_census": True,
            "requires_new_minimal_harness": True,
            "banked_selected_expert_gate_up_smoke": promoted,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": False,
            "banked_default_promotion": False,
        },
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c2-moe-shape-census-v1",
            "MoEProblemShape",
            "tokens_per_expert",
            "GroupProblemShape",
            "PASS_R5C2_MOE_SHAPE_CENSUS_NEW_HARNESS_REQUIRED",
            "banked_selected_expert_gate_up_smoke",
            "banked_kernel_speed",
        ],
        SUMMARY: [
            "MoE shape census banked",
            "Requires new minimal harness",
            "Selected expert gate/up smoke banked",
            "strict-exit",
        ],
        WRAPPER: [
            "r6000_stage6_r5c2_moe_shape_census.py",
            "summarize_stage6_r5c2_moe_shape_census.py",
            "CUTLASS_DIR",
            "probe_rc",
        ],
        DOC: [
            "R5-C2",
            "selected expert gate/up",
            "banked_moe_shape_census=true",
            "banked_selected_expert_gate_up_smoke=false",
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
        if "MoE shape census banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing MoE shape census row")
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
        print("Stage 6 R5-C2 MoE-shape census tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C2 MoE-shape census tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
