"""Phase A: per-kernel implementation toggle for native CUDA island work.

Each Phase A target (MTP verify ABI, active-MoE fused boundary, transposed
NVFP4 decode layout, native full-attn boundary, MTP policy surface) lands as
an *opt-in* native kernel that can be A/B switched against the current
Triton/PyTorch baseline via environment variables.

This module is the single source of truth for those toggles. It deliberately
avoids defaulting any kernel to "native" until that kernel has passed:

- per-kernel cosine parity (>=0.9999) and max rel_l2 (<=1e-3) in
  ``benchmarks/kernel_parity_harness.py``;
- a 6-prompt smoke against a serving probe;
- a 24h 16k-context smoke for state-carrying kernels.

The toggle defaults match production Config D today: every Phase A kernel
returns the ``baseline`` backend until explicitly enabled. Failure mode is
always fall-back to baseline, not crash.

See ``docs/PHASE_A_FOUNDATION_INFRA_20260517.md`` for the foundation contract
and ``docs/LYNN_ENGINE_CPP_RUST_REWRITE_ROI_20260517.md`` for the Phase A
target list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


KernelBackend = Literal["baseline", "native_cuda", "native_cpp"]

#: Phase A targets keyed to the ROI doc's priority order. Update this dict
#: when a new Phase A kernel island lands. The string keys are also the
#: environment variable suffixes: ``LYNN_NATIVE_KERNEL_<KEY>``.
_PHASE_A_KERNELS: tuple[str, ...] = (
    # 1. P112/P113/P115 — MTP K=2/K=3 verify ABI + decoder layer kernel.
    "mtp_verify",
    # 2. P50/P69/P97 — variable-expert active-MoE fused boundary.
    "active_moe",
    # 3. Atlas clean-room — transposed NVFP4/W4A8 decode weight layout.
    "transposed_decode",
    # 4. P9H/P9I/P9U/P9V/P9W — native/static full-attn layer boundary.
    "full_attn_boundary",
    # 5. P117 — runtime MTP policy surface (disabled/shadow/allowlist).
    "mtp_policy",
)

_VALID_BACKENDS: frozenset[str] = frozenset({"baseline", "native_cuda", "native_cpp"})


def _env_key(kernel: str) -> str:
    return f"LYNN_NATIVE_KERNEL_{kernel.upper()}"


@dataclass(slots=True)
class KernelToggle:
    """Per-kernel backend selection for Phase A native island A/B tests.

    Default is ``baseline`` for every Phase A kernel. Native backends are
    opt-in via env vars and must have a working fallback to baseline if the
    native implementation is missing or fails its parity gate.
    """

    selections: dict[str, KernelBackend] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "KernelToggle":
        src = os.environ if env is None else env
        selections: dict[str, KernelBackend] = {}
        for kernel in _PHASE_A_KERNELS:
            raw = src.get(_env_key(kernel), "baseline").strip().lower()
            if raw not in _VALID_BACKENDS:
                raise ValueError(
                    f"{_env_key(kernel)}={raw!r} is not one of {sorted(_VALID_BACKENDS)}; "
                    f"set to 'baseline' to opt out."
                )
            selections[kernel] = raw  # type: ignore[assignment]
        return cls(selections=selections)

    @classmethod
    def all_baseline(cls) -> "KernelToggle":
        return cls(selections={k: "baseline" for k in _PHASE_A_KERNELS})

    def backend(self, kernel: str) -> KernelBackend:
        if kernel not in _PHASE_A_KERNELS:
            raise KeyError(
                f"unknown Phase A kernel {kernel!r}; valid: {list(_PHASE_A_KERNELS)}"
            )
        return self.selections.get(kernel, "baseline")

    def is_native(self, kernel: str) -> bool:
        return self.backend(kernel) != "baseline"

    def any_native(self) -> bool:
        return any(self.is_native(k) for k in _PHASE_A_KERNELS)

    def summary(self) -> dict[str, str]:
        """Stable dict suitable for serving banner / metrics."""
        return {k: self.selections.get(k, "baseline") for k in _PHASE_A_KERNELS}


def known_kernels() -> tuple[str, ...]:
    """Return the canonical Phase A kernel name tuple in ship order."""
    return _PHASE_A_KERNELS


def env_keys() -> dict[str, str]:
    """Map each kernel name to its environment variable name."""
    return {k: _env_key(k) for k in _PHASE_A_KERNELS}
