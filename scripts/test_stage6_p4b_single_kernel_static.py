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
RUNTIME_PREFLIGHT = ROOT / "scripts" / "spark_stage6_p4b_runtime_bridge_preflight.py"
RUNTIME_SUMMARIZER = ROOT / "scripts" / "summarize_stage6_p4b_runtime_bridge_preflight.py"
RUNTIME_WRAPPER = ROOT / "scripts" / "run_spark_stage6_p4b_runtime_bridge_preflight.sh"
SINGLE_CTA_PREFLIGHT = ROOT / "scripts" / "spark_stage6_p4b_single_cta_numeric_preflight.py"
SINGLE_CTA_SUMMARIZER = ROOT / "scripts" / "summarize_stage6_p4b_single_cta_numeric_preflight.py"
SINGLE_CTA_WRAPPER = ROOT / "scripts" / "run_spark_stage6_p4b_single_cta_numeric_preflight.sh"
SINGLE_CTA_MICROBENCH = ROOT / "scripts" / "spark_stage6_p4b_single_cta_microbench.py"
SINGLE_CTA_MICROBENCH_SUMMARIZER = ROOT / "scripts" / "summarize_stage6_p4b_single_cta_microbench.py"
SINGLE_CTA_MICROBENCH_WRAPPER = ROOT / "scripts" / "run_spark_stage6_p4b_single_cta_microbench.sh"


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
    runtime_preflight_text = RUNTIME_PREFLIGHT.read_text(encoding="utf-8") if RUNTIME_PREFLIGHT.exists() else ""
    runtime_summarizer_text = RUNTIME_SUMMARIZER.read_text(encoding="utf-8") if RUNTIME_SUMMARIZER.exists() else ""
    runtime_wrapper_text = RUNTIME_WRAPPER.read_text(encoding="utf-8") if RUNTIME_WRAPPER.exists() else ""
    single_cta_preflight_text = SINGLE_CTA_PREFLIGHT.read_text(encoding="utf-8") if SINGLE_CTA_PREFLIGHT.exists() else ""
    single_cta_summarizer_text = SINGLE_CTA_SUMMARIZER.read_text(encoding="utf-8") if SINGLE_CTA_SUMMARIZER.exists() else ""
    single_cta_wrapper_text = SINGLE_CTA_WRAPPER.read_text(encoding="utf-8") if SINGLE_CTA_WRAPPER.exists() else ""
    single_cta_microbench_text = SINGLE_CTA_MICROBENCH.read_text(encoding="utf-8") if SINGLE_CTA_MICROBENCH.exists() else ""
    single_cta_microbench_summarizer_text = SINGLE_CTA_MICROBENCH_SUMMARIZER.read_text(encoding="utf-8") if SINGLE_CTA_MICROBENCH_SUMMARIZER.exists() else ""
    single_cta_microbench_wrapper_text = SINGLE_CTA_MICROBENCH_WRAPPER.read_text(encoding="utf-8") if SINGLE_CTA_MICROBENCH_WRAPPER.exists() else ""
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
            "LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE",
            "LYNN_NATIVE_P4B_MULTI_CTA_REFERENCE",
            "p4b_single_cta_reference_kernel",
            "p4b_multi_cta_reference_kernel",
            "do not bank fused-kernel speed or promote this backend",
        ],
        failures,
    )
    _check_absent(
        "P4B C++ single-kernel ABI",
        cu_fn,
        [
            "inter_scratch",
            "inter_out",
            "torch::empty",
            "torch::zeros",
            "TensorOptions",
            "active_moe_fused_atomic_scalar_kernel",
            "atomicAdd",
            "float* __restrict__ out",
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
            "_p4b_fused_zero_shadow_single_kernel_contract_call_count",
            "_p4b_fused_zero_shadow_single_kernel_contract_last_shapes",
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
            "run_spark_stage6_p4b_runtime_bridge_preflight.sh",
            "run_spark_stage6_p4b_single_cta_numeric_preflight.sh",
            "run_spark_stage6_p4b_single_cta_microbench.sh",
            "PASS_SINGLE_KERNEL_FAILLOUD_CONTRACT",
            "PASS_P4B_RUNTIME_BRIDGE_FAILLOUD",
            "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE",
            "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED",
            "LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE=1",
            "39539.166us",
            "0.007145x",
            "must not reuse the historical `active_moe_fused_atomic_scalar_kernel`",
            "No output scratch",
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
    _check_contains(
        "P4B runtime bridge preflight",
        runtime_preflight_text,
        [
            "PASS_P4B_RUNTIME_BRIDGE_FAILLOUD",
            "banked_p4b_runtime_bridge_preflight",
            "_p4b_fused_zero_shadow_single_kernel_contract_call_count",
            "p4b_last_shapes_out_only",
            "do not bank fused-kernel speed or promote this backend",
        ],
        failures,
    )
    _check_contains(
        "P4B runtime bridge summarizer",
        runtime_summarizer_text,
        [
            "PASS_P4B_RUNTIME_BRIDGE_FAILLOUD",
            "P4B native backend call count did not advance exactly once",
            "P4B last_shapes did not prove out-only ABI",
            "real runtime bridge reaches P4B fail-loud symbol",
        ],
        failures,
    )
    _check_contains(
        "P4B runtime Spark wrapper",
        runtime_wrapper_text,
        [
            "LYNN_STAGE6_EXPECT_MANIFEST",
            "p4b_runtime_bridge_preflight_",
            "spark_stage6_p4b_runtime_bridge_preflight.py",
            "summarize_stage6_p4b_runtime_bridge_preflight.py",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA numeric preflight",
        single_cta_preflight_text,
        [
            "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE",
            "LYNN_NATIVE_P4B_SINGLE_CTA_REFERENCE",
            "banked_single_cta_numeric_preflight",
            "banked_fused_kernel",
            "no_inter_scratch_candidate_abi",
            "rel_l2_vs_reference",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA numeric summarizer",
        single_cta_summarizer_text,
        [
            "PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE",
            "fused-kernel speed boundary violated",
            "P4B candidate output matches P4A two-stage reference",
            "No inter_scratch candidate ABI",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA Spark wrapper",
        single_cta_wrapper_text,
        [
            "LYNN_STAGE6_EXPECT_MANIFEST",
            "p4b_single_cta_numeric_preflight_",
            "spark_stage6_p4b_single_cta_numeric_preflight.py",
            "summarize_stage6_p4b_single_cta_numeric_preflight.py",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA microbench",
        single_cta_microbench_text,
        [
            "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED",
            "banked_single_cta_microbench",
            "banked_fused_kernel",
            "candidate_vs_reference_speedup",
            "measurement_only_reference_path",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA microbench summarizer",
        single_cta_microbench_summarizer_text,
        [
            "PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED",
            "measurement recorded; speed/default promotion still closed",
            "fused-kernel speed boundary violated",
            "P4B candidate median",
        ],
        failures,
    )
    _check_contains(
        "P4B single-CTA microbench wrapper",
        single_cta_microbench_wrapper_text,
        [
            "LYNN_STAGE6_EXPECT_MANIFEST",
            "p4b_single_cta_microbench_",
            "spark_stage6_p4b_single_cta_microbench.py",
            "summarize_stage6_p4b_single_cta_microbench.py",
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
