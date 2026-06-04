#!/usr/bin/env python3
"""GPU-free checks for the R5-C3B gate/up value-materialization contract."""
from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "reports" / "stage6" / "R5C3B_GATEUP_VALUE_MATERIALIZATION_CONTRACT_20260604.md"
R5C_DOC = ROOT / "reports" / "stage6" / "R5C_NVF4_UE4M3_CUTLASS_CONTRACT_20260604.md"
TRACE = ROOT / "reports" / "stage6" / "R5C3A_GATEUP_PREFILL_TIMING_TRACE_20260604.md"


def _scatter(values: list[list[float]], inverse: dict[int, tuple[int, int]], tokens: int, top_k: int) -> list[list[list[float]]]:
    selected: list[list[list[float] | None]] = [[None for _ in range(top_k)] for _ in range(tokens)]
    for row_idx, row in enumerate(values):
        if row_idx not in inverse:
            raise AssertionError(f"missing inverse entry {row_idx}")
        token_idx, slot = inverse[row_idx]
        if selected[token_idx][slot] is not None:
            raise AssertionError(f"duplicate selected slot {(token_idx, slot)}")
        selected[token_idx][slot] = row
    if any(slot is None for row in selected for slot in row):
        raise AssertionError("missing selected slot")
    return [[list(slot) for slot in row] for row in selected if all(slot is not None for slot in row)]


def _swiglu_checksum(selected: list[list[list[float]]]) -> float:
    acc = 0.0
    for token in selected:
        for row in token:
            if len(row) % 2 != 0:
                raise AssertionError("N_gateup must be even")
            half = len(row) // 2
            for gate, up in zip(row[:half], row[half:]):
                acc += (gate / (1.0 + math.exp(-gate))) * up
    if not math.isfinite(acc):
        raise AssertionError("non-finite SwiGLU checksum")
    return acc


def _fixture() -> tuple[list[list[float]], dict[int, tuple[int, int]]]:
    values = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6],
    ]
    inverse = {
        0: (0, 0),
        1: (1, 1),
        2: (0, 1),
        3: (1, 0),
    }
    return values, inverse


def main() -> int:
    failures: list[str] = []
    checks = {
        DOC: [
            "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE",
            "full D-row values",
            "value_digest_matches_r5c2c_digest",
            "scatter_values_d_ref_match",
            "host_swiglu_checksum_recorded",
            "banked_gateup_value_materialization=true",
            "banked_host_swiglu_checksum_smoke=true",
            "banked_down_projection_numeric_parity=false",
            "banked_grouped_moe_fp4_mma_poc=false",
            "banked_kernel_speed=false",
            "banked_default_promotion=false",
            "R5-C3C",
            "digest-only",
            "timing-only",
        ],
        R5C_DOC: [
            "R5-C3A",
            "R5-C3B",
            "R5-C3C",
            "Gate/up value materialization",
        ],
        TRACE: [
            "trace-only speed evidence",
            "2.11x",
            "Does not bank full MoE speed",
        ],
    }
    for path, needles in checks.items():
        if not path.exists():
            failures.append(f"missing {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{path.relative_to(ROOT)}: missing {needle!r}")

    values, inverse = _fixture()
    expected = [[[0.1, 0.2, 0.3, 0.4], [0.9, 1.0, 1.1, 1.2]], [[1.3, 1.4, 1.5, 1.6], [0.5, 0.6, 0.7, 0.8]]]
    selected = _scatter(values, inverse, tokens=2, top_k=2)
    if selected != expected:
        failures.append("slot scatter fixture did not preserve inverse order")
    checksum = _swiglu_checksum(selected)
    if not math.isclose(checksum, 5.497892317577369, rel_tol=0.0, abs_tol=1e-12):
        failures.append(f"unexpected SwiGLU checksum {checksum}")

    duplicate_inverse = dict(inverse)
    duplicate_inverse[1] = duplicate_inverse[0]
    try:
        _scatter(values, duplicate_inverse, tokens=2, top_k=2)
        failures.append("duplicate selected slot was not rejected")
    except AssertionError:
        pass

    odd_values = [[0.1, 0.2, 0.3]]
    try:
        _swiglu_checksum([[odd_values[0]]])
        failures.append("odd N_gateup was not rejected")
    except AssertionError:
        pass

    if failures:
        print("Stage 6 R5-C3B gate/up value-materialization contract static check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C3B gate/up value-materialization contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
