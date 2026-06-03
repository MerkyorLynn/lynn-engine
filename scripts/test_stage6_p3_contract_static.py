#!/usr/bin/env python3
"""GPU-free static check for the Stage 6 P3 grouped-MoE contract."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    (
        "reports/stage6/P3_GROUPED_MOE_ZERO_SHADOW_CONTRACT_20260604.md",
        [
            "active_moe_grouped_prefill",
            "Forbidden for a P3 bank",
            "P3-A PASS requires",
            "P3-A Runnable Probe",
            "scripts/run_spark_stage6_p3a_contract_probe.sh",
            "banked_fused_kernel=false",
            "P2-O remains the next resident-runner gate",
        ],
    ),
    (
        "reports/stage6/P2N_WIDER_LAYER_BLOCK_LINEAR_SMOKE_20260604.md",
        [
            "Verdict: **PASS on wider selected-layer coverage",
            "no active BF16 expert shadow",
            "Next gate: RC/server smoke",
        ],
    ),
    (
        "reports/stage6/P2O_PACKED_PREFILL_RC_GATE_RUNBOOK_20260604.md",
        [
            "P2-O is therefore an active-MoE no-reload",
            "`PASS` requires",
            "scripts/run_spark_stage6_p2o_rc_smoke.sh --preset basic",
        ],
    ),
    (
        "engine/moe_packed_nvfp4.py",
        [
            "def active_moe_grouped_prefill_p3a",
            "not a banked fused P3 kernel",
            "nvfp4_prefill_gate_up_silu_one_expert",
            "def moe_forward_verify_smallm_nvfp4",
            "mlp.experts._gate_up_packed",
            "mlp.experts._down_packed",
            "def moe_forward_decode_packed_nvfp4",
        ],
    ),
    (
        "triton_kernels/nvfp4_moe.py",
        [
            "def nvfp4_prefill_gate_up_silu_one_expert",
            "def nvfp4_grouped_down_weighted_sum",
            "expected grouped 3D tensors",
        ],
    ),
    (
        "scripts/spark_stage6_p2m_selected_layer_block_linear_smoke.py",
        [
            "LYNN_PACKED_PREFILL_SLOW_MODE",
            "p2e_hybrid",
            "LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA",
        ],
    ),
    (
        "scripts/spark_stage6_p3a_grouped_moe_contract_probe.py",
        [
            "active_moe_grouped_prefill_p3a",
            "banked_fused_kernel",
            "shadow_absent_at_candidate_start",
            "Active MoE only",
        ],
    ),
    (
        "scripts/run_spark_stage6_p3a_contract_probe.sh",
        [
            "PROVENANCE_FILES",
            "scripts/spark_stage6_p3a_grouped_moe_contract_probe.py",
            "nvidia_smi_before.txt",
            "passes.all",
        ],
    ),
]


def main() -> int:
    failures: list[str] = []
    for rel, needles in CHECKS:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"missing {rel}")
            continue
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                failures.append(f"{rel}: missing {needle!r}")
    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 2
    print("P3 grouped-MoE contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
