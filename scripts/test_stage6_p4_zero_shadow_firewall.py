#!/usr/bin/env python3
"""GPU-free firewall for the Stage 6 P4 zero-shadow native boundary."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PY_BACKEND = ROOT / "engine" / "moe_packed_nvfp4.py"
CU_CONTRACT = ROOT / "csrc" / "lynn_native" / "moe_fused_zero_shadow_contract.cu"

PY_FORBIDDEN = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "reload_decode_bf16_shadows",
    "nvfp4_grouped_gate_up",
    "nvfp4_grouped_down",
    "F.linear",
    "_dequant_nvfp4_slot",
    "torch.empty",
    "empty_like",
]

CU_FORBIDDEN = [
    "gate_up_proj",
    "down_proj",
    "torch::empty",
    "at::empty",
    "cudaMalloc",
    "reload_decode_bf16_shadows",
    "nvfp4_grouped",
]

PY_REQUIRED = [
    "mlp.experts._active_inter_scratch",
    "mlp.experts._active_out_scratch",
    "mlp.experts._gate_up_packed",
    "mlp.experts._gate_up_scale",
    "mlp.experts._gate_up_global_scale",
    "mlp.experts._down_packed",
    "mlp.experts._down_scale",
    "mlp.experts._down_global_scale",
    "active_moe_fused_zero_shadow_out_contract",
]

PY_DISPATCH_REQUIRED = [
    "refusing to ",
    "fall back to the generic Triton two-stage path",
]

CU_REQUIRED = [
    "lynn_native_active_moe_grouped_per16_nonatomic_out_reference",
    "torch::Tensor inter_scratch",
    "torch::Tensor out",
    "check_cuda_tensor(inter_scratch, \"inter_scratch\", torch::kBFloat16)",
    "check_cuda_tensor(out, \"out\", torch::kBFloat16)",
    "inter_scratch must be [T, top_k, 512]",
    "P4 two-stage reference currently supports T=1 decode only",
    "gate_up_packed must be [E, 1024, 1024]",
    "down_packed must be [E, 2048, 256]",
]


def _extract_function(text: str, name: str) -> str:
    pattern = re.compile(rf"^def {re.escape(name)}\(", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise AssertionError(f"missing function {name}")
    start = match.start()
    next_match = re.search(r"^def ", text[match.end():], re.MULTILINE)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[start:end]


def _extract_active_moe_backend_fallback_set(text: str) -> str:
    match = re.search(r"elif backend in \{(?P<body>.*?)\}:", text, re.DOTALL)
    if not match:
        raise AssertionError("missing active-MoE generic backend fallback set")
    return match.group("body")


def _check_contains(label: str, text: str, needles: list[str], failures: list[str]) -> None:
    for needle in needles:
        if needle not in text:
            failures.append(f"{label}: missing required {needle!r}")


def _check_absent(label: str, text: str, needles: list[str], failures: list[str]) -> None:
    for needle in needles:
        if needle in text:
            failures.append(f"{label}: forbidden {needle!r}")


def main() -> int:
    failures: list[str] = []
    py_text = PY_BACKEND.read_text(encoding="utf-8")
    cu_text = CU_CONTRACT.read_text(encoding="utf-8")
    py_fn = _extract_function(py_text, "_active_moe_native_fused_zero_shadow_out_contract")
    fallback_set = _extract_active_moe_backend_fallback_set(py_text)

    _check_contains("P4 Python backend", py_fn, PY_REQUIRED, failures)
    _check_absent("P4 Python backend", py_fn, PY_FORBIDDEN, failures)
    _check_contains("active-MoE dispatch", py_text, PY_DISPATCH_REQUIRED, failures)
    _check_absent("active-MoE generic fallback set", fallback_set, ["fused_zero_shadow_out_contract"], failures)
    _check_contains("P4 CUDA contract", cu_text, CU_REQUIRED, failures)
    _check_absent("P4 CUDA contract", cu_text, CU_FORBIDDEN, failures)

    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 2
    print("P4 zero-shadow firewall PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
