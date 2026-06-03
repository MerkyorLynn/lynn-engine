#!/usr/bin/env python3
"""GPU-free static check for the Stage 6 P4 native fused-MoE ABI."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    (
        "csrc/lynn_native/moe_fused_zero_shadow_contract.cu",
        [
            "lynn_native_active_moe_fused_zero_shadow_out_contract",
            "check_cuda_tensor",
            "hidden must be [T, 2048]",
            "gate_up_packed must be [E, 1024, 1024]",
            "down_packed must be [E, 2048, 256]",
            "active_moe_fused_zero_shadow_out_contract passed all packed-NVFP4 shape/layout checks",
            "do not add BF16 expert shadows",
        ],
    ),
    (
        "engine/native_cuda.py",
        [
            "moe_fused_zero_shadow_contract.cu",
        ],
    ),
    (
        "csrc/lynn_native/bindings.cpp",
        [
            "void lynn_native_active_moe_fused_zero_shadow_out_contract",
            "active_moe_fused_zero_shadow_out_contract",
            "P4 fail-loud caller-owned-output ABI",
        ],
    ),
    (
        "engine/moe_packed_nvfp4.py",
        [
            "def _active_moe_native_fused_zero_shadow_out_contract",
            "active_moe_fused_zero_shadow_out_contract",
            "LYNN_NATIVE_FUSED_ZERO_SHADOW_TILE_TOKENS",
            "fused_zero_shadow_out_contract",
        ],
    ),
    (
        "scripts/spark_stage6_p4_native_abi_preflight.py",
        [
            "lynn-stage6-p4-native-fused-moe-abi-preflight-v1",
            "banked_native_abi_preflight",
            "banked_fused_kernel",
            "byte_budget",
            "bf16_shadow_equivalent_bytes",
            "packed_vs_bf16_shadow_ratio",
            "forbidden_shadow_tensor_names",
            "PASS_ABI_CONTRACT",
            "BLOCKED_SYMBOL_MISSING",
            "UNEXPECTED_IMPLEMENTED",
            "P4 fused 4-bit zero-shadow CUDA kernel is not implemented yet",
        ],
    ),
    (
        "scripts/run_spark_stage6_p4_native_abi_preflight.sh",
        [
            "PROVENANCE_FILES",
            "scripts/spark_stage6_p4_native_abi_preflight.py",
            "scripts/summarize_stage6_p4_native_abi_preflight.py",
            "nvidia_smi_before.txt",
            "summary.md",
        ],
    ),
    (
        "scripts/summarize_stage6_p4_native_abi_preflight.py",
        [
            "PASS_ABI_CONTRACT",
            "fused-kernel promotion boundary violated",
            "Banked fused kernel",
            "Zero-shadow ABI",
            "Packed byte budget",
        ],
    ),
    (
        "scripts/write_stage6_p4_native_abi_report.py",
        [
            "Bank P4 native ABI preflight only",
            "Banked fused kernel",
            "Packed byte budget",
            "Provenance manifest",
        ],
    ),
    (
        "scripts/test_stage6_p4_zero_shadow_firewall.py",
        [
            "P4 zero-shadow firewall PASS",
            "mlp.experts.gate_up_proj",
            "reload_decode_bf16_shadows",
            "torch::empty",
            "active_moe_fused_zero_shadow_out_contract",
        ],
    ),
    (
        "reports/stage6/P4_NATIVE_FUSED_MOE_ABI_CONTRACT_20260604.md",
        [
            "ABI/PREFLIGHT ONLY; no fused P4 kernel is banked yet",
            "active_moe_fused_zero_shadow_out_contract",
            "caller-owned `out`",
            "no BF16 expert weight tensors in the ABI",
            "PASS_ABI_CONTRACT",
            "python3 scripts/test_stage6_p4_zero_shadow_firewall.py",
            "python3 scripts/test_stage6_p4_native_abi_static.py",
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
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{rel}: missing {needle!r}")
    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 2
    print("P4 native fused-MoE ABI static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
