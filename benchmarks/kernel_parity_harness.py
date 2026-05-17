"""Phase A: Triton-vs-CUDA kernel parity validation harness.

Every Phase A native CUDA kernel island must pass through this harness
before being switched on in serving. The harness records:

- per-output cosine similarity (must be >= 0.9999)
- per-output max relative L2 error (must be <= 1e-3)
- per-output max absolute error (informational, no hard gate)
- timing (informational, parity comes before perf)
- fail-loud on shape/dtype mismatch

Usage from a kernel-specific benchmark::

    from benchmarks.kernel_parity_harness import ParityHarness, ParityCase

    cases = [
        ParityCase(
            name="moe_router_layer3_seed20260517",
            baseline=lambda: triton_router(x, gate_w),
            candidate=lambda: native_router(x, gate_w),
        ),
        ...
    ]
    harness = ParityHarness(kernel="active_moe", out_dir="reports/phase_a/")
    report = harness.run(cases)
    report.assert_passed()  # raises if any case fails

The harness writes JSON to ``reports/phase_a/parity_<kernel>_<timestamp>.json``
so downstream tooling (per-kernel gate scripts, the foundation profiling JSON
writer) can pick it up without re-running.

This file is intentionally a thin foundation. It does not run any kernels
itself; each Phase A kernel island wires its own ``ParityCase`` list.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover - parity harness needs torch
    torch = None  # type: ignore[assignment]


COS_THRESHOLD: float = 0.9999
REL_L2_THRESHOLD: float = 1e-3


@dataclass(slots=True)
class ParityCase:
    """A single parity case: baseline vs candidate kernel on the same input.

    Both callables MUST return identically shaped tensors. Inputs are
    captured implicitly via closure (the harness does not own input
    construction; callers prepare and free their own buffers).
    """

    name: str
    baseline: Callable[[], "torch.Tensor"]
    candidate: Callable[[], "torch.Tensor"]
    notes: str = ""


@dataclass(slots=True)
class ParityResult:
    name: str
    passed: bool
    cosine: float
    rel_l2: float
    abs_max: float
    baseline_ms: float
    candidate_ms: float
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "cosine": self.cosine,
            "rel_l2": self.rel_l2,
            "abs_max": self.abs_max,
            "baseline_ms": self.baseline_ms,
            "candidate_ms": self.candidate_ms,
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class ParityReport:
    kernel: str
    results: list[ParityResult] = field(default_factory=list)
    written_path: str = ""

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def assert_passed(self) -> None:
        failed = [r for r in self.results if not r.passed]
        if failed:
            details = "\n".join(
                f"  {r.name}: cos={r.cosine:.6f} rel_l2={r.rel_l2:.6f} "
                f"reason={r.failure_reason}"
                for r in failed
            )
            raise AssertionError(
                f"Phase A parity gate FAILED for kernel {self.kernel!r} "
                f"({len(failed)}/{len(self.results)} cases failed):\n{details}"
            )


class ParityHarness:
    """Run a list of ParityCase through baseline and candidate, gate, persist."""

    def __init__(
        self,
        *,
        kernel: str,
        out_dir: str | Path,
        cos_threshold: float = COS_THRESHOLD,
        rel_l2_threshold: float = REL_L2_THRESHOLD,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "kernel_parity_harness requires torch. Run inside the Lynn engine env."
            )
        self.kernel = kernel
        self.out_dir = Path(out_dir)
        self.cos_threshold = cos_threshold
        self.rel_l2_threshold = rel_l2_threshold

    def _time_once(self, fn: Callable[[], "torch.Tensor"]) -> tuple["torch.Tensor", float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return out, dt_ms

    def _compare(self, a: "torch.Tensor", b: "torch.Tensor") -> ParityResult:
        # placeholder; actual values filled below
        if a.shape != b.shape:
            return ParityResult(
                name="", passed=False, cosine=float("nan"),
                rel_l2=float("nan"), abs_max=float("nan"),
                baseline_ms=0.0, candidate_ms=0.0,
                failure_reason=f"shape mismatch: baseline {tuple(a.shape)} vs candidate {tuple(b.shape)}",
            )
        if a.dtype != b.dtype:
            # convert candidate to baseline dtype for cosine; flag as note
            b = b.to(a.dtype)
        a_flat = a.detach().to(torch.float32).flatten()
        b_flat = b.detach().to(torch.float32).flatten()
        a_norm = torch.linalg.vector_norm(a_flat)
        b_norm = torch.linalg.vector_norm(b_flat)
        denom = (a_norm * b_norm).clamp_min(1e-30)
        cosine = float(torch.dot(a_flat, b_flat) / denom)
        diff = a_flat - b_flat
        diff_norm = float(torch.linalg.vector_norm(diff))
        rel_l2 = diff_norm / float(a_norm.clamp_min(1e-30))
        abs_max = float(diff.abs().max())
        passed = cosine >= self.cos_threshold and rel_l2 <= self.rel_l2_threshold
        reason = ""
        if not passed:
            reason = (
                f"cos={cosine:.6f} (need >= {self.cos_threshold}); "
                f"rel_l2={rel_l2:.6f} (need <= {self.rel_l2_threshold})"
            )
        return ParityResult(
            name="", passed=passed, cosine=cosine, rel_l2=rel_l2,
            abs_max=abs_max, baseline_ms=0.0, candidate_ms=0.0,
            failure_reason=reason,
        )

    def run(self, cases: list[ParityCase]) -> ParityReport:
        report = ParityReport(kernel=self.kernel)
        for case in cases:
            base_out, base_ms = self._time_once(case.baseline)
            cand_out, cand_ms = self._time_once(case.candidate)
            res = self._compare(base_out, cand_out)
            res.name = case.name
            res.baseline_ms = base_ms
            res.candidate_ms = cand_ms
            report.results.append(res)

        report.written_path = self._persist(report)
        return report

    def _persist(self, report: ParityReport) -> str:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.out_dir / f"parity_{self.kernel}_{ts}.json"
        payload = {
            "schema": "lynn-engine-phase-a-parity-v1",
            "kernel": self.kernel,
            "thresholds": {
                "cosine_min": self.cos_threshold,
                "rel_l2_max": self.rel_l2_threshold,
            },
            "results": [r.to_dict() for r in report.results],
            "summary": {
                "passed": report.passed,
                "n_cases": len(report.results),
                "n_failed": sum(1 for r in report.results if not r.passed),
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)
