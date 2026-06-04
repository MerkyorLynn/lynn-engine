#!/usr/bin/env python3
"""GPU-free checks for R5-C4 full active-MoE speed A/B summary tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c4_full_active_moe_speed_ab.py"


def _passes(**overrides: bool) -> dict:
    passes = {
        "input_r5c3c_passed": True,
        "same_scope_ab": True,
        "real_model_weights": True,
        "real_router_outputs": True,
        "candidate_no_active_bf16_shadow": True,
        "candidate_no_reload": True,
        "candidate_no_bf16_weight_materialization": True,
        "candidate_full_active_moe_boundary_timed": True,
        "timing_includes_gateup_swiglu_down_weighted_scatter": True,
        "numeric_vs_w4a16_or_p3_reference": True,
        "candidate_median_speedup_vs_best_reference_ge_1p05": True,
        "banked_full_active_moe_prefill_speed": True,
        "banked_grouped_moe_fp4_mma_poc": True,
        "banked_kernel_speed": True,
        "banked_decode_tps": False,
        "banked_server_rc": False,
        "banked_default_promotion": False,
        "banked_full_transformer_prefill": False,
    }
    passes.update(overrides)
    return passes


def _fixture(decision: str = "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB", **pass_overrides: bool) -> dict:
    return {
        "schema": "lynn-stage6-r5c4-full-active-moe-prefill-speed-ab-v1",
        "decision": decision,
        "kernel_speed_scope": "active_moe_prefill_only",
        "passes": _passes(**pass_overrides),
        "lanes": {
            "smoke": {
                "candidate_ms": 0.4,
                "best_reference_ms": 0.5,
                "median_speedup_vs_best_reference": 1.25,
                "numeric_max_abs": 0.0,
                "numeric_rel_l2": 0.0,
                "numeric_cosine": 1.0,
            },
            "production_shape": {
                "candidate_ms": 4.0,
                "best_reference_ms": 4.8,
                "median_speedup_vs_best_reference": 1.2,
                "numeric_max_abs": 0.0,
                "numeric_rel_l2": 0.0,
                "numeric_cosine": 1.0,
            },
        },
    }


def main() -> int:
    failures: list[str] = []
    if not SUMMARY.exists():
        failures.append(f"missing {SUMMARY.relative_to(ROOT)}")
    else:
        text = SUMMARY.read_text(encoding="utf-8")
        for needle in [
            "PASS_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB",
            "DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED",
            "banked_decode_tps",
            "banked_server_rc",
            "banked_default_promotion",
            "banked_full_transformer_prefill",
            "strict-pass-exit",
            "strict-diagnostic-exit",
        ]:
            if needle not in text:
                failures.append(f"{SUMMARY.relative_to(ROOT)} missing {needle!r}")

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        good = tmp / "good.json"
        bad_decode = tmp / "bad_decode.json"
        diagnostic = tmp / "diagnostic.json"
        md = tmp / "summary.md"
        good.write_text(json.dumps(_fixture()), encoding="utf-8")
        bad_decode.write_text(json.dumps(_fixture(banked_decode_tps=True)), encoding="utf-8")
        diagnostic_data = _fixture(
            decision="DIAGNOSTIC_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_CLOSED",
            candidate_median_speedup_vs_best_reference_ge_1p05=False,
            banked_full_active_moe_prefill_speed=False,
            banked_grouped_moe_fp4_mma_poc=False,
            banked_kernel_speed=False,
        )
        diagnostic.write_text(json.dumps(diagnostic_data), encoding="utf-8")
        ok = subprocess.run(
            [sys.executable, str(SUMMARY), str(good), "--markdown-out", str(md), "--strict-pass-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if ok.returncode != 0:
            failures.append(f"strict pass good fixture failed: {ok.stderr or ok.stdout}")
        if "Decode TPS banked | `False`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing decode TPS boundary")
        bad = subprocess.run(
            [sys.executable, str(SUMMARY), str(bad_decode), "--strict-pass-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if bad.returncode == 0:
            failures.append("strict pass accepted decode TPS promotion")
        diag = subprocess.run(
            [sys.executable, str(SUMMARY), str(diagnostic), "--strict-diagnostic-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if diag.returncode != 0:
            failures.append(f"strict diagnostic fixture failed: {diag.stderr or diag.stdout}")
        diag_pass = subprocess.run(
            [sys.executable, str(SUMMARY), str(diagnostic), "--strict-pass-exit"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if diag_pass.returncode == 0:
            failures.append("strict pass accepted diagnostic speed-closed artifact")

    if failures:
        print("Stage 6 R5-C4 full active-MoE speed A/B tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C4 full active-MoE speed A/B tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
