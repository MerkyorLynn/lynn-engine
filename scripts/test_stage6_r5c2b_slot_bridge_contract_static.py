#!/usr/bin/env python3
"""GPU-free checks for the R5-C2B slot-preserving selected-output contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "reports" / "stage6" / "R5C2B_SLOT_PRESERVING_SELECTED_OUTPUT_CONTRACT_20260604.md"


@dataclass(frozen=True)
class Pair:
    token_idx: int
    top_k_slot: int
    expert_id: int


def _pair_payload(pair: Pair) -> int:
    return pair.expert_id * 100_000 + pair.token_idx * 10 + pair.top_k_slot


def _build_routes(tokens: int, top_k: int, counts: list[int]) -> list[list[int]]:
    if sum(counts) != tokens * top_k:
        raise AssertionError("counts must sum to tokens * top_k")
    remaining = counts[:]
    routes: list[list[int]] = []
    for _token in range(tokens):
        row: list[int] = []
        for _slot in range(top_k):
            candidates = sorted(range(len(counts)), key=lambda expert: (-remaining[expert], expert))
            expert = next(
                (
                    candidate
                    for candidate in candidates
                    if remaining[candidate] > 0 and candidate not in row
                ),
                None,
            )
            if expert is None:
                raise AssertionError("could not construct unique top-k route")
            row.append(expert)
            remaining[expert] -= 1
        routes.append(row)
    if any(remaining):
        raise AssertionError(f"unconsumed expert counts: {remaining}")
    return routes


def _expected_selected_slots(pair_order: list[Pair], tokens: int, top_k: int) -> list[list[int]]:
    selected: list[list[int | None]] = [[None for _slot in range(top_k)] for _token in range(tokens)]
    for pair in pair_order:
        if selected[pair.token_idx][pair.top_k_slot] is not None:
            raise AssertionError(f"duplicate selected slot {(pair.token_idx, pair.top_k_slot)}")
        selected[pair.token_idx][pair.top_k_slot] = _pair_payload(pair)
    if any(value is None for row in selected for value in row):
        raise AssertionError("missing selected slot after reference fill")
    return [[int(value) for value in row] for row in selected]


def _reference_grouped_bridge(pair_order: list[Pair], num_experts: int) -> dict:
    grouped_order: list[Pair] = []
    observed_counts: list[int] = []
    for expert in range(num_experts):
        group = [pair for pair in pair_order if pair.expert_id == expert]
        grouped_order.extend(group)
        observed_counts.append(len(group))

    expert_offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for expert, count in enumerate(observed_counts):
        expert_offsets[expert] = (cursor, cursor + count)
        cursor += count

    inverse_order = {
        grouped_idx: (pair.token_idx, pair.top_k_slot)
        for grouped_idx, pair in enumerate(grouped_order)
    }
    return {
        "grouped_order": grouped_order,
        "observed_counts": observed_counts,
        "expert_offsets": expert_offsets,
        "inverse_order": inverse_order,
    }


def _scatter_grouped_rows(
    grouped_values: list[int],
    inverse_order: dict[int, tuple[int, int]],
    tokens: int,
    top_k: int,
) -> list[list[int]]:
    selected: list[list[int | None]] = [[None for _slot in range(top_k)] for _token in range(tokens)]
    for grouped_idx, value in enumerate(grouped_values):
        if grouped_idx not in inverse_order:
            raise AssertionError(f"missing inverse entry for grouped row {grouped_idx}")
        token_idx, top_k_slot = inverse_order[grouped_idx]
        if not (0 <= token_idx < tokens and 0 <= top_k_slot < top_k):
            raise AssertionError(f"inverse entry out of bounds: {(token_idx, top_k_slot)}")
        if selected[token_idx][top_k_slot] is not None:
            raise AssertionError(f"duplicate scatter slot {(token_idx, top_k_slot)}")
        selected[token_idx][top_k_slot] = value
    if any(value is None for row in selected for value in row):
        raise AssertionError("missing selected slot after inverse scatter")
    return [[int(value) for value in row] for row in selected]


def _require_mismatch(name: str, candidate: list[list[int]], expected: list[list[int]], failures: list[str]) -> None:
    if candidate == expected:
        failures.append(f"fault injection did not perturb selected slots: {name}")


def _require_assertion(name: str, failures: list[str], fn) -> None:
    try:
        fn()
    except AssertionError:
        return
    failures.append(f"fault injection was not rejected: {name}")


def _slot_bridge_fixture(tokens: int = 128, top_k: int = 2, counts: list[int] | None = None) -> dict:
    counts = counts or [32, 64, 64, 96]
    routes = _build_routes(tokens, top_k, counts)
    pair_order = [
        Pair(token_idx=token_idx, top_k_slot=slot, expert_id=expert)
        for token_idx, row in enumerate(routes)
        for slot, expert in enumerate(row)
    ]
    bridge = _reference_grouped_bridge(pair_order, len(counts))
    grouped_values = [_pair_payload(pair) for pair in bridge["grouped_order"]]
    expected = _expected_selected_slots(pair_order, tokens, top_k)
    scatter = _scatter_grouped_rows(grouped_values, bridge["inverse_order"], tokens, top_k)

    return {
        "routes": routes,
        "pair_order": pair_order,
        "grouped_order": bridge["grouped_order"],
        "observed_counts": bridge["observed_counts"],
        "expert_offsets": bridge["expert_offsets"],
        "inverse_order": bridge["inverse_order"],
        "grouped_values": grouped_values,
        "scatter": scatter,
        "expected": expected,
    }


def _run_fault_injection_checks(fixture: dict, failures: list[str], tokens: int = 128, top_k: int = 2) -> None:
    inverse_order = dict(fixture["inverse_order"])
    grouped_values = list(fixture["grouped_values"])
    expected = fixture["expected"]

    swapped_values = grouped_values[:]
    swapped_values[0], swapped_values[1] = swapped_values[1], swapped_values[0]
    swapped_scatter = _scatter_grouped_rows(swapped_values, inverse_order, tokens, top_k)
    _require_mismatch("swapped grouped rows", swapped_scatter, expected, failures)

    swapped_inverse = dict(inverse_order)
    swapped_inverse[0], swapped_inverse[1] = swapped_inverse[1], swapped_inverse[0]
    swapped_inverse_scatter = _scatter_grouped_rows(grouped_values, swapped_inverse, tokens, top_k)
    _require_mismatch("swapped inverse scatter rows", swapped_inverse_scatter, expected, failures)

    duplicate_inverse = dict(inverse_order)
    duplicate_inverse[1] = duplicate_inverse[0]
    _require_assertion(
        "duplicate inverse scatter slot",
        failures,
        lambda: _scatter_grouped_rows(grouped_values, duplicate_inverse, tokens, top_k),
    )

    missing_inverse = dict(inverse_order)
    del missing_inverse[0]
    _require_assertion(
        "missing inverse entry",
        failures,
        lambda: _scatter_grouped_rows(grouped_values, missing_inverse, tokens, top_k),
    )


def main() -> int:
    failures: list[str] = []
    if not DOC.exists():
        failures.append(f"missing {DOC.relative_to(ROOT)}")
    else:
        text = DOC.read_text(encoding="utf-8")
        for needle in [
            "PASS_R5C2B_SLOT_PRESERVING_SELECTED_OUTPUT_CONTRACT",
            "banked_slot_bridge_contract=true",
            "banked_selected_output_kernel=false",
            "banked_grouped_moe_fp4_mma_poc=false",
            "banked_kernel_speed=false",
            "banked_default_promotion=false",
            "token_idx",
            "top_k_slot",
            "expert_id",
            "pair_order",
            "grouped_order",
            "tokens_per_expert",
            "expert_offsets",
            "inverse_order",
            "[T, top_k, inter]",
            "Fault-injection checks",
        ]:
            if needle not in text:
                failures.append(f"{DOC.relative_to(ROOT)}: missing {needle!r}")

    fixture = _slot_bridge_fixture()
    pair_order = fixture["pair_order"]
    grouped_order = fixture["grouped_order"]
    counts = fixture["observed_counts"]
    expert_offsets = fixture["expert_offsets"]
    if counts != [32, 64, 64, 96]:
        failures.append(f"unexpected tokens_per_expert counts: {counts}")
    if len(pair_order) != 128 * 2:
        failures.append("pair_order length does not equal T * top_k")
    if len({(pair.token_idx, pair.top_k_slot) for pair in pair_order}) != len(pair_order):
        failures.append("slot identity was duplicated or lost")
    if any(len(set(row)) != 2 for row in fixture["routes"]):
        failures.append("fixture contains duplicate experts within one top-k row")
    if [pair.expert_id for pair in grouped_order] != sorted(pair.expert_id for pair in grouped_order):
        failures.append("grouped_order is not grouped by expert_id")
    for expert, (start, end) in expert_offsets.items():
        if end - start != counts[expert]:
            failures.append(f"expert_offsets mismatch for expert {expert}")
        if any(pair.expert_id != expert for pair in grouped_order[start:end]):
            failures.append(f"expert_offsets slice contains wrong expert {expert}")
    if fixture["scatter"] != fixture["expected"]:
        failures.append("inverse_order scatter did not reconstruct [T, top_k] slots")
    _run_fault_injection_checks(fixture, failures)

    if failures:
        print("Stage 6 R5-C2B slot bridge contract static check FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Stage 6 R5-C2B slot bridge contract static check PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
