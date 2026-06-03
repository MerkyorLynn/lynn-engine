#!/usr/bin/env python3
"""GPU-free static gate for Stage 6 P4B true fused single-kernel ABI."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CU = ROOT / "csrc" / "lynn_native" / "moe_fused_zero_shadow_contract.cu"
BINDINGS = ROOT / "csrc" / "lynn_native" / "bindings.cpp"
PY_BACKEND = ROOT / "engine" / "moe_packed_nvfp4.py"
REPORT = ROOT / "reports" / "stage6" / "P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md"
PREFLIGHT = ROOT / "scripts" / "spark_stage6_p4b_single_kernel_preflight.py"
SUMMARIZER = ROOT / "scripts" / "summarize_stage6_p4b_single_kernel_preflight.py"
WRAPPER = ROOT / "scripts" / "run_spark_stage6_p4b_single_kernel_preflight.sh"


def _extract_cu_function(text: str, name: str) -> str:
    marker = f"void {name}("
    start = text.find(marker)
    if start < 0:
        raise AssertionError(f"missing C++ function {name}")
    next_fn = text.find("\nvoid ", start + len(marker))
    return text[start:] if next_fn < 0 else text[start:next_fn]


def _extract_py_function(text: str, name: str) -> str:
    match = re.search(rf"^def {re.escape(name)}\(", text, re.MULTILINE)
    if not match:
        raise AssertionError(f"missing Python function {name}")
    next_match = re.search(r"^def ", text[match.end():], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.start():end]


def _extract_active_moe_backend_fallback_set(text: str) -> str:
    match = re.search(r"elif backend in \{(?P<body>.*?)\}:", text, re.DOTALL)
    if not match:
        raise AssertionError("missing active-MoE generic backend fallback set")
    return match.group("body")


def _check_contains(label: str, text: str, needles: list[str], failures: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            failures.append(f"{label}: missing {needle!r}")


def _check_absent(label: str, text: str, needles: list[str], failures: list[str]) -> None:
    for needle in needles:
        if needle in text:
            failures.append(f"{label}: forbidden {needle!r}")


def main() -> int:
    failures: list[str] = []
    cu_text = CU.read_text(encoding="utf-8")
    bindings_text = BINDINGS.read_text(encoding="utf-8")
    py_text = PY_BACKEND.read_text(encoding="utf-8")
    report_text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else ""
    preflight_text = PREFLIGHT.read_text(encoding="utf-8") if PREFLIGHT.exists() else ""
    summarizer_text = SUMMARIZER.read_text(encoding="utf-8") if SUMMARIZER.exists() else ""
    wrapper_text = WRAPPER.read_text(encoding="utf-8") if WRAPPER.exists() else ""
    cu_fn = _extract_cu_function(cu_text, "lynn_native_active_moe_fused_zero_shadow_single_kernel_contract")
    py_fn = _extract_py_function(py_text, "_active_moe_native_fused_zero_shadow_single_kernel_contract")
    fallback_set = _extract_active_moe_backend_fallback_set(py_text)

    _check_contains(
        "P4B C++ single-kernel ABI",
        cu_fn,
        [
            "P4B hidden must be [T, 2048]",
            "P4B gate_up_packed must be [E, 1024, 1024]",
            "P4B down_packed must be [E, 2048, 256]",
            "P4B single-kernel fused zero-shadow contract is not implemented yet",
            "do not bank fused-kernel speed or promote this backend",
        ],
        failures,
    )
    _check_absent(
        "P4B C++ single-kernel ABI",
        cu_fn,
        [
            "inter_scratch",
            "lynn_native_active_moe_grouped_per16_nonatomic_out_reference",
        ],
        failures,
    )
    _check_contains(
        "P4B pybind",
        bindings_text,
        [
            "void lynn_native_active_moe_fused_zero_shadow_single_kernel_contract",
            "active_moe_fused_zero_shadow_single_kernel_contract",
            "P4B fail-loud single-kernel ABI",
        ],
        failures,
    )
    _check_contains(
        "P4B Python backend",
        py_fn,
        [
            "LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_single_kernel_contract requires",
            "active_moe_fused_zero_shadow_single_kernel_contract",
            "mlp.experts._active_out_scratch",
            "LYNN_NATIVE_FUSED_ZERO_SHADOW_TILE_EXPERTS",
        ],
        failures,
    )
    _check_absent("active-MoE generic fallback set", fallback_set, ["fused_zero_shadow_single_kernel_contract"], failures)
    _check_contains(
        "P4B dispatch",
        py_text,
        [
            "elif backend == \"fused_zero_shadow_single_kernel_contract\" and _layer_selected_for_native_cuda(cfg):",
            "fall back to the generic Triton two-stage path",
        ],
        failures,
    )
    _check_contains(
        "P4B report",
        report_text,
        [
            "Stage 6 P4B - native fused single-kernel contract",
            "not implemented yet",
            "banked_fused_kernel=false",
            "fused_zero_shadow_single_kernel_contract",
            "run_spark_stage6_p4b_single_kernel_preflight.sh",
            "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT",
        ],
        failures,
    )
    _check_contains(
        "P4B fail-loud preflight",
        preflight_text,
        [
            "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT",
            "FAIL_SINGLE_KERNEL_CONTRACT",
            "banked_single_kernel_contract_preflight",
            "banked_fused_kernel",
            "no_inter_scratch_abi",
            "do not bank fused-kernel speed or promote this backend",
        ],
        failures,
    )
    _check_contains(
        "P4B preflight summarizer",
        summarizer_text,
        [
            "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT",
            "fused-kernel promotion boundary violated",
            "tensor manifest contains inter_scratch",
            "single-kernel fail-loud contract preflight passed",
        ],
        failures,
    )
    _check_contains(
        "P4B Spark wrapper",
        wrapper_text,
        [
            "LYNN_STAGE6_EXPECT_MANIFEST",
            "p4b_single_kernel_preflight_",
            "spark_stage6_p4b_single_kernel_preflight.py",
            "summarize_stage6_p4b_single_kernel_preflight.py",
        ],
        failures,
    )

    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 2
    print("P4B single-kernel static gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
