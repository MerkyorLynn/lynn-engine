#!/usr/bin/env python3
"""GPU-free checks for Stage 6 R5-C2 selected-expert gate/up smoke tooling."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "r6000_stage6_r5c2_selected_expert_gateup_smoke.py"
SUMMARY = ROOT / "scripts" / "summarize_stage6_r5c2_selected_expert_gateup_smoke.py"
WRAPPER = ROOT / "scripts" / "r6000_stage6_r5c2_selected_expert_gateup_smoke.sh"
DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"
LEDGER = ROOT / "scripts" / "write_stage6_evidence_ledger.py"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _fixture(speed_promoted: bool = False) -> dict:
    return {
        "schema": "lynn-stage6-r5c2-selected-expert-gateup-smoke-v1",
        "decision": "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE",
        "cutlass_dir": "/fixture/cutlass",
        "example": "/fixture/cutlass/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu",
        "benchmark_file": "/fixture/bench.txt",
        "git": {
            "head": {"ok": True, "stdout_tail": "abc123"},
            "branch": {"ok": True, "stdout_tail": "main"},
        },
        "selected_expert_shape": {
            "tokens": 128,
            "top_k": 2,
            "experts": 4,
            "tokens_per_expert": [32, 64, 64, 96],
            "n_gate_up": 128,
            "k_hidden": 256,
            "alignment": 32,
        },
        "groups_seen": 4,
        "build_result": {"atomic_scope_patch": {"applied": False, "restored": False}},
        "run_parse": {
            "avg_runtime_ms": [0.04, 0.05],
            "tflops": [10.0, 12.0],
        },
        "passes": {
            "banked_selected_expert_gate_up_smoke": True,
            "banked_grouped_moe_fp4_mma_poc": False,
            "banked_kernel_speed": speed_promoted,
            "banked_default_promotion": False,
            "route_tokens_match": True,
            "route_topk_unique": True,
            "tokens_per_expert_match": True,
            "benchmark_shapes_aligned_32": True,
            "benchmark_groups_match_experts": True,
            "groups_seen_match_experts": True,
            "cooperative_passed": True,
            "pingpong_passed": True,
            "host_reference_seen": True,
            "dispositions_passed_count_ge_2": True,
        },
    }


def main() -> int:
    failures: list[str] = []
    checks = {
        PROBE: [
            "lynn-stage6-r5c2-selected-expert-gateup-smoke-v1",
            "DEFAULT_COUNTS",
            "tokens_per_expert",
            "benchmark_shapes_aligned_32",
            "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE",
            "banked_selected_expert_gate_up_smoke",
            "banked_grouped_moe_fp4_mma_poc",
            "banked_kernel_speed",
        ],
        SUMMARY: [
            "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE",
            "Selected-expert gate/up smoke banked",
            "Grouped-MoE FP4-MMA POC banked",
            "strict-exit",
        ],
        WRAPPER: [
            "r6000_stage6_r5c2_selected_expert_gateup_smoke.py",
            "summarize_stage6_r5c2_selected_expert_gateup_smoke.py",
            "nvidia_smi_before.txt",
            "probe_rc",
        ],
        DOC: [
            "R5-C2",
            "Selected expert gate/up native GEMM smoke",
            "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE",
            "banked_selected_expert_gate_up_smoke=true",
            "banked_kernel_speed=false",
        ],
        LEDGER: [
            "_r5c2_selected_expert_gateup_smoke_gate",
            "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE",
            "R5C2_SELECTED_EXPERT_GATEUP_SMOKE_BANKED",
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
        if "Selected-expert gate/up smoke banked | `True`" not in md.read_text(encoding="utf-8"):
            failures.append("summary markdown missing selected-expert banked row")
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
        print("Stage 6 R5-C2 selected-expert gate/up smoke tooling self-test FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C2 selected-expert gate/up smoke tooling self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
