#!/usr/bin/env python3
"""GPU-free static gate for the P4C active-reuse kernel decision."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "reports" / "stage6" / "P4C_ACTIVE_REUSE_KERNEL_DECISION_20260604.md"
P4B_DOC = ROOT / "reports" / "stage6" / "P4B_NATIVE_FUSED_SINGLE_KERNEL_CONTRACT_20260604.md"
CU = ROOT / "csrc" / "lynn_native" / "moe_fused_zero_shadow_contract.cu"
BINDINGS = ROOT / "csrc" / "lynn_native" / "bindings.cpp"
PY_BACKEND = ROOT / "engine" / "moe_packed_nvfp4.py"
SINGLE_SUMMARY = ROOT / "reports" / "stage6" / "p4b_single_cta_microbench_20260604_085842" / "summary.md"
MULTI_SUMMARY = ROOT / "reports" / "stage6" / "p4b_multi_cta_microbench_20260604_091150" / "summary.md"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"missing required file: {path.relative_to(ROOT)}")
    return path.read_text(encoding="utf-8")


def _check_contains(label: str, text: str, needles: list[str], failures: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            failures.append(f"{label}: missing {needle!r}")


def main() -> int:
    failures: list[str] = []
    doc = _read(DOC)
    p4b_doc = _read(P4B_DOC)
    cu = _read(CU)
    bindings = _read(BINDINGS)
    py_backend = _read(PY_BACKEND)
    single_summary = _read(SINGLE_SUMMARY)
    multi_summary = _read(MULTI_SUMMARY)

    _check_contains(
        "P4C decision doc",
        doc,
        [
            "decision bank only; no new fused-kernel speed is banked here",
            "39.54 ms vs P4A 0.283 ms = 0.007x",
            "48.34 ms vs P4A 0.279 ms =",
            "0.0058x",
            "The next kernel must preserve",
            "active reuse.",
            "ordinary CUDA shared memory is CTA-local",
            "Recomputing active per output tile is forbidden",
            "P4C, not P4B",
            "LYNN_NATIVE_ACTIVE_MOE_BACKEND=fused_zero_shadow_active_reuse_contract",
            "computes `active[top_k,512]` once per decode token, not once per output tile",
            "reports speed against P4A synthetic reference and current `~44-45 TPS` RC",
            "banked_default_promotion=false",
            "Forbidden False Positives",
            "scripts/run_spark_stage6_p4c_runtime_bridge_preflight.sh",
            "PASS_P4C_ACTIVE_REUSE_RUNTIME_BRIDGE",
            "banked_p4c_active_reuse_runtime_bridge=true",
        ],
        failures,
    )
    _check_contains(
        "P4B contract keeps active-reuse caveat",
        p4b_doc,
        [
            "single-CTA numeric reference banked",
            "microbench proves it is not the speed path",
            "multi-CTA/CUTLASS-style",
            "banked_fused_kernel=false",
        ],
        failures,
    )
    _check_contains(
        "P4B/P4C C++ structure",
        cu,
        [
            "__shared__ __nv_bfloat16 active[kP4BTopK * kIntermediate]",
            "p4b_single_cta_reference_kernel",
            "p4b_multi_cta_reference_kernel",
            "It recomputes active[slot, inter] per tile",
            "LYNN_NATIVE_P4B_MULTI_CTA_REFERENCE",
            "lynn_native_active_moe_fused_zero_shadow_active_reuse_contract",
        ],
        failures,
    )
    _check_contains(
        "P4C pybind/backend",
        bindings + "\n" + py_backend,
        [
            "active_moe_fused_zero_shadow_active_reuse_contract",
            "P4C active-reuse two-phase packed-NVFP4 zero-shadow active MoE contract",
            "_active_moe_native_fused_zero_shadow_active_reuse_contract",
            "fused_zero_shadow_active_reuse_contract",
            "_p4c_fused_zero_shadow_active_reuse_contract_call_count",
            "refusing to ",
            "fall back to the generic Triton two-stage path",
        ],
        failures,
    )
    _check_contains(
        "single-CTA measured artifact",
        single_summary,
        [
            "Candidate mode | `single_cta`",
            "P4B candidate median | `39539.166259765625` us",
            "P4B/P4A speedup | `0.007145198514702697`",
            "Numeric rel L2 | `0.0`",
        ],
        failures,
    )
    _check_contains(
        "multi-CTA measured artifact",
        multi_summary,
        [
            "Candidate mode | `multi_cta`",
            "P4B candidate median | `48339.202880859375` us",
            "P4B/P4A speedup | `0.0057775712698842595`",
            "Numeric rel L2 | `0.0`",
        ],
        failures,
    )

    if failures:
        print("P4C active-reuse decision static gate FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("P4C active-reuse decision static gate PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
