#!/usr/bin/env python3
"""GPU-free static check for the Stage 6 P3-B selected-prefill runbook."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = [
    (
        "reports/stage6/P3B_SELECTED_PREFILL_GATE_RUNBOOK_20260604.md",
        [
            "RUNBOOK ONLY; no P3-B result is banked yet",
            "Required Predecessors",
            "P2-O `basic` report",
            "P2-O `rc-mini` report",
            "P3-A report",
            "selected layers: start with `0-3`",
            "active BF16 expert shadow reload/rebuild",
            "final selected-stack cosine >= 0.999",
            "no reload call is observed",
            "scripts/run_stage6_gpu_gate_suite.sh",
        ],
    ),
    (
        "scripts/write_stage6_p2o_report.py",
        [
            "Bank P2-O for this preset",
            "Do not bank P2-O",
            "Manifest matches",
        ],
    ),
    (
        "scripts/write_stage6_p3a_report.py",
        [
            "Bank P3-A as a contract-shaped grouped active-MoE probe only",
            "Do not promote P3 or claim a fused kernel",
            "Those remain P3-B/P3-C/P3-D gates",
        ],
    ),
    (
        "scripts/write_stage6_gpu_gate_suite_report.py",
        [
            "orchestration-level evidence only",
            "child reports remain authoritative",
            "P2-O and P3-A must still",
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
    print("P3-B selected-prefill contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
