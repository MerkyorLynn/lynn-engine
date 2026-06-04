#!/usr/bin/env python3
"""GPU-free static check for the R6000 grouped-MoE FP4-MMA POC contract."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "reports" / "stage6" / "R6000_GROUPED_MOE_FP4_MMA_POC_CONTRACT_20260604.md"
CENSUS = ROOT / "reports" / "stage6" / "r6000_fp4_mma_census_20260604_164457" / "result.json"
RUNBOOK = ROOT / "reports" / "stage6" / "R6000_FP4_MMA_BRINGUP_RUNBOOK_20260604.md"
P3_CONTRACT = ROOT / "reports" / "stage6" / "P3_GROUPED_MOE_ZERO_SHADOW_CONTRACT_20260604.md"


CHECKS = {
    DOC: [
        "contract only; no Lynn kernel, runtime default, or speed claim",
        "PASS_R6000_FP4_MMA_BRINGUP",
        "RTX PRO 6000 96GB",
        "omit rental host IDs",
        "Spark owns 35B serving/memory/MTP/compiled-loop ROI",
        "R6000 owns native FP4-MMA/CUTLASS/CuTe grouped-kernel evidence",
        "Lynn resident NVFP4",
        "Blackwell/CUTLASS block-scaled FP4",
        "zero-copy reinterpretation",
        "explicit repack",
        "Silent reinterpretation is forbidden",
        "H = 2048",
        "I = 512",
        "E = 256",
        "top_k = 8",
        "R5-A layout bridge",
        "R5-B gate/up FP4-MMA",
        "R5-C down FP4-MMA",
        "R5-D grouped active MoE",
        "The first implementation target is **R5-A**",
        "banked_layout_bridge",
        "banked_grouped_moe_fp4_mma_poc",
        "banked_kernel_speed",
        "banked_default_promotion",
        "per-16-to-block-scaled layout bridge",
        "python3 scripts/test_stage6_r6000_grouped_moe_poc_contract_static.py",
    ],
    CENSUS: [
        "PASS_R6000_FP4_MMA_BRINGUP",
        "contract_suite_all_pass",
        "vllm_nvfp4_or_marlin_seen",
        "p85_blockscaled_fp4_mma_contract",
        "p87_layout_tile_contract",
        "\"kernel_promoted\": false",
        "\"default_runtime_changed\": false",
        "\"speed_claim\": false",
    ],
    RUNBOOK: [
        "Banked Artifact",
        "Redacted rental R6000 host",
        "P76/P79/P85/P87/P103 all pass",
        "does not promote a",
    ],
    P3_CONTRACT: [
        "active_moe_grouped_prefill",
        "mlp.experts._gate_up_packed",
        "mlp.experts._down_packed",
        "banked_fused_kernel=false",
    ],
}


def main() -> int:
    failures: list[str] = []
    for path, needles in CHECKS.items():
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)}: missing {needle!r}")
    if failures:
        print("R6000 grouped-MoE FP4-MMA POC contract static check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("R6000 grouped-MoE FP4-MMA POC contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
