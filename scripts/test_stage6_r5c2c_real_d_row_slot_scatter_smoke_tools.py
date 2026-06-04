#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C2C real D-row slot scatter tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c2c_real_d_row_slot_scatter_smoke.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c2c_real_d_row_slot_scatter_smoke.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c2c_real_d_row_slot_scatter_smoke.sh"
DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"
LEDGER = ROOT / "scripts" / "write_stage6_evidence_ledger.py"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(**overrides: bool) -> dict:
    passes = {
        "build_invoked": True,
        "build_succeeded": True,
        "d_row_digest_patch_applied": True,
        "d_row_digest_patch_restored": True,
        "run_succeeded": True,
        "cooperative_passed": True,
        "pingpong_passed": True,
        "host_reference_seen": True,
        "dispositions_passed_count_ge_2": True,
        "groups_seen_match_experts": True,
        "tokens_per_expert_match": True,
        "grouped_order_complete": True,
        "digest_file_exists": True,
        "schedules_captured": True,
        "schedule_scatters_passed": True,
        "banked_real_d_row_slot_scatter": True,
        "banked_selected_output_kernel_epilogue": False,
        "banked_swiglu_or_down_projection": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }
    passes.update(overrides)
    return {
        "schema": "lynn-stage6-r5c2c-real-d-row-slot-scatter-smoke-v1",
        "decision": "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE",
        "cutlass_dir": "/fixture/cutlass",
        "example": "/fixture/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu",
        "benchmark_file": "/fixture/bench.txt",
        "d_row_digest_file": "/fixture/digest.jsonl",
        "selected_expert_shape": {
            "tokens": 128,
            "top_k": 2,
            "experts": 4,
            "tokens_per_expert": [32, 64, 64, 96],
            "k_hidden": 256,
            "n_gate_up": 128,
        },
        "cutlass_run": {"patch": {"applied": True, "restored": True}},
        "run_parse": {"avg_runtime_ms": [0.02, 0.018]},
        "scatter_schedules": {
            "cooperative": {
                "records": 256,
                "row_counts": [32, 64, 64, 96],
                "passes": {
                    "d_ref_row_digests_match": True,
                    "scatter_d_ref_match": True,
                    "fault_injections_detected": True,
                    "scatter_error_absent": True,
                },
            },
            "pingpong": {
                "records": 256,
                "row_counts": [32, 64, 64, 96],
                "passes": {
                    "d_ref_row_digests_match": True,
                    "scatter_d_ref_match": True,
                    "fault_injections_detected": True,
                    "scatter_error_absent": True,
                },
            },
        },
        "passes": passes,
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c2c-real-d-row-slot-scatter-smoke-v1",
            "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE",
            "LYNN_STAGE6_R5C2C_D_ROW_DIGEST_PATCH",
            "lynn_d_row_digest",
            "_temporary_d_row_digest_patch",
            "_scatter_schedule",
            "duplicate D-row record",
            "scatter_error",
            "banked_real_d_row_slot_scatter",
            "banked_selected_output_kernel_epilogue",
            "banked_swiglu_or_down_projection",
        ],
        SUMMARY: [
            "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE",
            "Real D-row slot scatter banked",
            "Selected-output epilogue kernel banked",
            "SwiGLU/down projection banked",
            "strict-exit",
        ],
        WRAPPER: [
            "r6000_stage6_r5c2c_real_d_row_slot_scatter_smoke.py",
            "summarize_stage6_r5c2c_real_d_row_slot_scatter_smoke.py",
            "--build",
            "nvidia_smi_before.txt",
            "probe_rc",
        ],
        DOC: [
            "R5-C2C Real D-Row Slot Scatter Smoke Gate",
            "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE",
            "banked_real_d_row_slot_scatter=true",
            "banked_selected_output_kernel_epilogue=false",
            "banked_swiglu_or_down_projection=false",
        ],
        LEDGER: [
            "_r5c2c_real_d_row_slot_scatter_smoke_gate",
            "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE",
            "R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE_BANKED",
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
        bad_speed = tmp / "bad_speed.json"
        bad_kernel = tmp / "bad_kernel.json"
        bad_swiglu = tmp / "bad_swiglu.json"
        md = tmp / "summary.md"
        good.write_text(json.dumps(_fixture()), encoding="utf-8")
        bad_speed.write_text(json.dumps(_fixture(banked_kernel_speed=True)), encoding="utf-8")
        bad_kernel.write_text(json.dumps(_fixture(banked_selected_output_kernel_epilogue=True)), encoding="utf-8")
        bad_swiglu.write_text(json.dumps(_fixture(banked_swiglu_or_down_projection=True)), encoding="utf-8")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(good), "--markdown-out", str(md), "--strict-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"summary strict good fixture failed: {ok.stderr or ok.stdout}")
        if "Real D-row slot scatter banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing real D-row banked row")
        for bad in [bad_speed, bad_kernel, bad_swiglu]:
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
        print("Stage 6 R5-C2C real D-row slot scatter tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C2C real D-row slot scatter tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
