#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C3C down/weighted parity tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "stage6_r5c3c_down_weighted_parity_smoke.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c3c_down_weighted_parity_smoke.py"
CONTRACT = ROOT / "reports" / "stage6" / "R5C3C_DOWN_WEIGHTED_PARITY_CONTRACT_20260604.md"


def _fixture(**overrides: bool) -> dict:
    passes = {
        "banked_down_projection_numeric_parity": True,
        "banked_weighted_topk_numeric_parity": True,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
        "all": True,
    }
    passes.update(overrides)
    return {
        "schema": "lynn-stage6-r5c3c-down-weighted-parity-smoke-v1",
        "decision": "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
        "input_decision": "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
        "selected_expert_shape": {
            "tokens": 128,
            "top_k": 2,
            "experts": 4,
            "swiglu_hidden": 64,
            "down_out_dim": 48,
        },
        "schedule_parity": {
            "cooperative": {
                "records": 256,
                "swiglu_d_ref_max_abs": 0.0,
                "down_d_ref_max_abs": 0.0,
                "weighted_topk_d_ref_max_abs": 0.0,
                "passes": {
                    "weighted_topk_hash_match": True,
                    "fault_injections_detected": True,
                },
            }
        },
        "passes": passes,
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c3c-down-weighted-parity-smoke-v1",
            "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
            "banked_down_projection_numeric_parity",
            "banked_weighted_topk_numeric_parity",
            "banked_grouped_moe_fp4_mma_poc",
            "banked_kernel_speed",
            "DEFAULT_INPUT",
            "selected_value_perturbation_detected",
            "route_weight_swap_detected",
        ],
        SUMMARY: [
            "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
            "Down projection numeric parity banked",
            "Weighted top-k numeric parity banked",
            "Grouped-MoE FP4-MMA POC banked",
            "strict-exit",
        ],
        CONTRACT: [
            "PASS_R5C3C_DOWN_WEIGHTED_PARITY_SMOKE",
            "banked_down_projection_numeric_parity=true",
            "banked_weighted_topk_numeric_parity=true",
            "banked_grouped_moe_fp4_mma_poc=false",
        ],
    }
    for path, needles in checks.items():
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)}: missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        good = tmp / "good.json"
        bad_speed = tmp / "bad_speed.json"
        bad_full = tmp / "bad_full.json"
        md = tmp / "summary.md"
        good.write_text(json.dumps(_fixture()), encoding="utf-8")
        bad_speed.write_text(json.dumps(_fixture(banked_kernel_speed=True)), encoding="utf-8")
        bad_full.write_text(json.dumps(_fixture(banked_grouped_moe_fp4_mma_poc=True)), encoding="utf-8")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(good), "--markdown-out", str(md), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"summary strict good fixture failed: {ok.stderr or ok.stdout}")
        if "Down projection numeric parity banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing down parity row")
        for bad in [bad_speed, bad_full]:
            fail = subprocess.run(
                [sys.executable, str(SUMMARY), str(bad), "--strict-exit"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if fail.returncode == 0:
                failures.append(f"summary strict bad fixture unexpectedly passed: {bad.name}")

    if failures:
        print("Stage 6 R5-C3C down/weighted parity tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C3C down/weighted parity tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
