#!/usr/bin/env python3
"""Stage 6 R5-C3B real CUTLASS gate/up value materialization smoke.

R5-C3B upgrades R5-C2C from row digests to full gate/up D-row values for a
small selected-expert smoke shape. It proves that real CUTLASS output values can
be scattered into Lynn selected slots and made ready for host-side SwiGLU.

This banks only gate/up value materialization and host-SwiGLU checksum smoke. It
does not bank down projection, full grouped-MoE speed, decode TPS, server/RC
behavior, or runtime defaults.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterator

from r6000_stage6_r5c1_cutlass_numeric_smoke import (  # noqa: E402
    _binary_path,
    _configure_and_build,
    _env,
    _git,
    _parse_run,
    _run,
)
from r6000_stage6_r5c2_selected_expert_gateup_smoke import (  # noqa: E402
    DEFAULT_COUNTS,
    EXAMPLE_REL,
    TARGET_ALIGNMENT,
    _build_routes,
    _parse_counts,
    _parse_groups_seen,
    _route_counts,
    _write_benchmark,
)


PATCH_MARKER = "LYNN_STAGE6_R5C3B_D_ROW_VALUE_PATCH"


@dataclass(frozen=True)
class Pair:
    token_idx: int
    top_k_slot: int
    expert_id: int
    group_row: int


def _pair_order(routes: list[list[int]], experts: int) -> list[Pair]:
    per_expert_row = [0] * experts
    pairs: list[Pair] = []
    for token_idx, row in enumerate(routes):
        for top_k_slot, expert_id in enumerate(row):
            group_row = per_expert_row[expert_id]
            per_expert_row[expert_id] += 1
            pairs.append(Pair(token_idx, top_k_slot, expert_id, group_row))
    return pairs


def _grouped_order(pair_order: list[Pair], experts: int) -> list[Pair]:
    grouped: list[Pair] = []
    for expert in range(experts):
        grouped.extend(pair for pair in pair_order if pair.expert_id == expert)
    return grouped


def _fnv1a_f32(values: list[float]) -> str:
    h = 1469598103934665603
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        h = _fnv1a_u32_bits_update(h, bits)
    return str(h)


def _fnv1a_u32_bits_update(h: int, bits: int) -> int:
    for byte in range(4):
        h ^= (int(bits) >> (byte * 8)) & 0xFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv1a_u32_bits(bits_values: list[int]) -> str:
    h = 1469598103934665603
    for bits in bits_values:
        for byte in range(4):
            h ^= (int(bits) >> (byte * 8)) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return str(h)


def _patch_source_text(text: str) -> str:
    if PATCH_MARKER in text:
        return text
    text = text.replace(
        "#include <float.h>\n",
        "#include <float.h>\n#include <cstdint>\n#include <cstring>\n#include <iomanip>\n",
    )
    text = text.replace(
        "  std::string benchmark_path;\n",
        "  std::string benchmark_path;\n  std::string lynn_d_row_value_path;\n",
    )
    text = text.replace(
        '    cmd.get_cmd_line_argument("benchmark", benchmark_path);\n',
        '    cmd.get_cmd_line_argument("benchmark", benchmark_path);\n'
        '    cmd.get_cmd_line_argument("lynn_d_row_value", lynn_d_row_value_path);\n',
    )
    helper = f"""
// {PATCH_MARKER}: emit real D/ref row values for Lynn R5-C3B.
std::uint64_t lynn_fnv1a_u32_value(std::uint64_t h, std::uint32_t value) {{
  for (int byte = 0; byte < 4; ++byte) {{
    h ^= static_cast<std::uint8_t>((value >> (byte * 8)) & 0xff);
    h *= 1099511628211ull;
  }}
  return h;
}}

float lynn_d_value(HostTensorD& tensor, int row, int col, int cols) {{
  return static_cast<float>(tensor.host_data(row * cols + col));
}}

std::uint32_t lynn_d_bits(HostTensorD& tensor, int row, int col, int cols) {{
  float value = lynn_d_value(tensor, row, col, cols);
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}}

std::uint64_t lynn_hash_d_row_value(HostTensorD& tensor, int row, int cols) {{
  std::uint64_t h = 1469598103934665603ull;
  for (int col = 0; col < cols; ++col) {{
    std::uint32_t bits = lynn_d_bits(tensor, row, col, cols);
    h = lynn_fnv1a_u32_value(h, bits);
  }}
  return h;
}}

void lynn_emit_d_row_values(Options const& options, char const* schedule_name, int group_idx, int rows, int cols) {{
  if (options.lynn_d_row_value_path.empty()) {{
    return;
  }}
  std::ofstream out(options.lynn_d_row_value_path, std::ios::app);
  out << std::setprecision(17);
  for (int row = 0; row < rows; ++row) {{
    auto d_hash = lynn_hash_d_row_value(block_D.at(group_idx), row, cols);
    auto ref_hash = lynn_hash_d_row_value(block_ref_D.at(group_idx), row, cols);
    out << "{{\\\"schedule\\\":\\\"" << schedule_name
        << "\\\",\\\"group\\\":" << group_idx
        << ",\\\"row\\\":" << row
        << ",\\\"cols\\\":" << cols
        << ",\\\"d_hash\\\":\\\"" << d_hash
        << "\\\",\\\"ref_hash\\\":\\\"" << ref_hash
        << "\\\",\\\"d_values\\\":[";
    for (int col = 0; col < cols; ++col) {{
      if (col) out << ",";
      out << lynn_d_value(block_D.at(group_idx), row, col, cols);
    }}
    out << "],\\\"ref_values\\\":[";
    for (int col = 0; col < cols; ++col) {{
      if (col) out << ",";
      out << lynn_d_value(block_ref_D.at(group_idx), row, col, cols);
    }}
    out << "],\\\"d_bits\\\":[";
    for (int col = 0; col < cols; ++col) {{
      if (col) out << ",";
      out << lynn_d_bits(block_D.at(group_idx), row, col, cols);
    }}
    out << "],\\\"ref_bits\\\":[";
    for (int col = 0; col < cols; ++col) {{
      if (col) out << ",";
      out << lynn_d_bits(block_ref_D.at(group_idx), row, col, cols);
    }}
    out << "]}}" << std::endl;
  }}
}}

"""
    text = text.replace("bool verify(const Options &options) {\n", helper + "bool verify(const Options &options, char const* schedule_name) {\n")
    text = text.replace("    block_SFD.at(i).sync_host();\n", "    block_SFD.at(i).sync_host();\n    lynn_emit_d_row_values(options, schedule_name, i, M, N);\n")
    text = text.replace(
        "int run(Options &options, bool host_problem_shapes_available = true)\n",
        "int run(Options &options, bool host_problem_shapes_available = true, char const* schedule_name = \"unknown\")\n",
    )
    text = text.replace("    result.passed = verify(options);\n", "    result.passed = verify(options, schedule_name);")
    text = text.replace(
        "  run<Gemm>(options, false /*host_problem_shapes_available*/);\n",
        "  run<Gemm>(options, false /*host_problem_shapes_available*/, \"cooperative\");\n",
    )
    text = text.replace(
        "  run<GemmPingpong>(options, false /*host_problem_shapes_available*/);\n",
        "  run<GemmPingpong>(options, false /*host_problem_shapes_available*/, \"pingpong\");\n",
    )
    return text


@contextmanager
def _temporary_d_row_value_patch(cutlass_dir: Path, enabled: bool) -> Iterator[dict[str, Any]]:
    source = cutlass_dir / EXAMPLE_REL
    info = {"enabled": enabled, "source": str(source), "applied": False, "restored": False, "error": None}
    original = ""
    try:
        if enabled and source.exists():
            original = source.read_text(encoding="utf-8")
            patched = _patch_source_text(original)
            if patched != original:
                source.write_text(patched, encoding="utf-8")
                info["applied"] = True
        yield info
    except Exception as exc:  # pragma: no cover - surfaced in R6000 artifact
        info["error"] = str(exc)
        raise
    finally:
        if info["applied"]:
            source.write_text(original, encoding="utf-8")
            info["restored"] = True


def _parse_value_records(path: Path) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    by_schedule: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        schedule = str(record["schedule"])
        key = (int(record["group"]), int(record["row"]))
        schedule_records = by_schedule.setdefault(schedule, {})
        if key in schedule_records:
            raise AssertionError(f"duplicate D-row value record for schedule={schedule} key={key}")
        cols = int(record["cols"])
        if len(record.get("d_values") or []) != cols or len(record.get("ref_values") or []) != cols:
            raise AssertionError(f"value length mismatch for schedule={schedule} key={key}")
        if len(record.get("d_bits") or []) != cols or len(record.get("ref_bits") or []) != cols:
            raise AssertionError(f"value bit length mismatch for schedule={schedule} key={key}")
        schedule_records[key] = record
    return by_schedule


def _scatter_values(
    records: dict[tuple[int, int], dict[str, Any]],
    pair_order: list[Pair],
    tokens: int,
    top_k: int,
    *,
    use_ref: bool,
) -> list[list[list[float]]]:
    selected: list[list[list[float] | None]] = [[None for _slot in range(top_k)] for _token in range(tokens)]
    key_name = "ref_values" if use_ref else "d_values"
    for pair in pair_order:
        record = records.get((pair.expert_id, pair.group_row))
        if record is None:
            raise AssertionError(f"missing D-row value for expert={pair.expert_id} row={pair.group_row}")
        if selected[pair.token_idx][pair.top_k_slot] is not None:
            raise AssertionError(f"duplicate selected slot {(pair.token_idx, pair.top_k_slot)}")
        selected[pair.token_idx][pair.top_k_slot] = [float(value) for value in record[key_name]]
    if any(value is None for row in selected for value in row):
        raise AssertionError("missing selected slot after real D-row value scatter")
    return [[list(value) for value in row if value is not None] for row in selected]


def _max_abs(a: list[list[list[float]]], b: list[list[list[float]]]) -> float:
    max_value = 0.0
    for token_a, token_b in zip(a, b):
        for row_a, row_b in zip(token_a, token_b):
            for value_a, value_b in zip(row_a, row_b):
                max_value = max(max_value, abs(value_a - value_b))
    return max_value


def _swiglu_checksum(selected: list[list[list[float]]]) -> float:
    acc = 0.0
    for token in selected:
        for row in token:
            if len(row) % 2:
                raise AssertionError("N_gateup must be even for host SwiGLU")
            half = len(row) // 2
            for gate, up in zip(row[:half], row[half:]):
                acc += (gate / (1.0 + math.exp(-gate))) * up
    if not math.isfinite(acc):
        raise AssertionError("non-finite host SwiGLU checksum")
    return acc


def _fault_injections(records: dict[tuple[int, int], dict[str, Any]], pair_order: list[Pair], tokens: int, top_k: int) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    expected = _scatter_values(records, pair_order, tokens, top_k, use_ref=True)
    keys = sorted(records)
    swapped = {key: dict(value) for key, value in records.items()}
    if len(keys) >= 2:
        swapped[keys[0]]["d_values"], swapped[keys[1]]["d_values"] = swapped[keys[1]]["d_values"], swapped[keys[0]]["d_values"]
    checks["swapped_d_rows_detected"] = _scatter_values(swapped, pair_order, tokens, top_k, use_ref=False) != expected
    missing = {key: dict(value) for key, value in records.items()}
    if keys:
        del missing[keys[0]]
    try:
        _scatter_values(missing, pair_order, tokens, top_k, use_ref=False)
        checks["missing_d_row_rejected"] = False
    except AssertionError:
        checks["missing_d_row_rejected"] = True
    duplicate_pairs = list(pair_order)
    if len(duplicate_pairs) >= 2:
        duplicate_pairs[1] = Pair(
            token_idx=duplicate_pairs[0].token_idx,
            top_k_slot=duplicate_pairs[0].top_k_slot,
            expert_id=duplicate_pairs[1].expert_id,
            group_row=duplicate_pairs[1].group_row,
        )
    try:
        _scatter_values(records, duplicate_pairs, tokens, top_k, use_ref=False)
        checks["duplicate_slot_rejected"] = False
    except AssertionError:
        checks["duplicate_slot_rejected"] = True
    return checks


def _value_report(records_by_schedule: dict[str, dict[tuple[int, int], dict[str, Any]]], pair_order: list[Pair], expected_counts: list[int], tokens: int, top_k: int) -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    for schedule, records in sorted(records_by_schedule.items()):
        row_counts = [sum(1 for expert, _row in records if expert == idx) for idx in range(len(expected_counts))]
        try:
            d_selected = _scatter_values(records, pair_order, tokens, top_k, use_ref=False)
            ref_selected = _scatter_values(records, pair_order, tokens, top_k, use_ref=True)
            value_digest_matches = all(
                _fnv1a_u32_bits([int(v) for v in record["d_bits"]]) == str(record["d_hash"])
                and _fnv1a_u32_bits([int(v) for v in record["ref_bits"]]) == str(record["ref_hash"])
                for record in records.values()
            )
            d_ref_hash_match = all(str(record["d_hash"]) == str(record["ref_hash"]) for record in records.values())
            max_abs = _max_abs(d_selected, ref_selected)
            checksum = _swiglu_checksum(d_selected)
            fault = _fault_injections(records, pair_order, tokens, top_k)
            error = ""
        except (AssertionError, OverflowError, ValueError) as exc:
            value_digest_matches = False
            d_ref_hash_match = False
            max_abs = float("inf")
            checksum = float("nan")
            fault = {}
            error = str(exc)
        schedules[schedule] = {
            "records": len(records),
            "row_counts": row_counts,
            "cols": sorted({int(record.get("cols", 0)) for record in records.values()}),
            "value_digest_matches_r5c2c_digest": value_digest_matches,
            "d_ref_row_hashes_match": d_ref_hash_match,
            "scatter_values_max_abs": max_abs,
            "host_swiglu_checksum": checksum,
            "fault_checks": fault,
            "value_error": error,
            "passes": {
                "row_counts_match": row_counts == expected_counts,
                "full_d_row_values_captured": bool(records) and all(record.get("d_values") and record.get("ref_values") for record in records.values()),
                "full_d_row_value_bits_captured": bool(records) and all(record.get("d_bits") and record.get("ref_bits") for record in records.values()),
                "value_digest_matches_r5c2c_digest": value_digest_matches,
                "scatter_values_d_ref_match": max_abs == 0.0,
                "host_swiglu_checksum_recorded": math.isfinite(checksum),
                "fault_injections_detected": bool(fault) and all(fault.values()),
                "value_error_absent": not error,
            },
        }
    return schedules


def _run_cutlass(args: argparse.Namespace, value_path: Path, bench: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    build_result: dict[str, Any] = {"skipped": True}
    patch_info: dict[str, Any] = {"enabled": bool(args.build), "applied": False, "restored": False}
    if args.build:
        with _temporary_d_row_value_patch(cutlass_dir, True) as patch_info:
            build_result = _configure_and_build(
                cutlass_dir=cutlass_dir,
                build_dir=build_dir,
                cuda_home=args.cuda_home,
                python_bin=args.python_bin,
                timeout_s=args.timeout_s,
                clean_build=args.clean_build,
                atomic_scope_patch=not args.no_atomic_scope_patch,
            )
    if value_path.exists():
        value_path.unlink()
    run_cmd = [
        str(binary),
        f"--benchmark={bench}",
        f"--iterations={args.iterations}",
        "--alpha=1",
        "--beta=0",
        f"--norm_constant={args.norm_constant}",
        f"--lynn_d_row_value={value_path}",
    ]
    if binary.exists():
        run_result = _run(run_cmd, cwd=build_dir, env=_env(args.cuda_home, args.python_bin), timeout_s=args.timeout_s)
    else:
        run_result = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": "binary missing", "cmd": run_cmd}
    return {"build_result": build_result, "patch": patch_info, "run_result": run_result}, _parse_run(
        run_result.get("stdout_tail", ""), run_result.get("stderr_tail", "")
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    example = cutlass_dir / EXAMPLE_REL
    out = Path(args.out).expanduser().resolve()
    bench = Path(args.benchmark_file).expanduser().resolve() if args.benchmark_file else out.with_suffix(".benchmark.txt")
    value_path = Path(args.value_file).expanduser().resolve() if args.value_file else out.with_suffix(".d_row_values.jsonl")
    bench.parent.mkdir(parents=True, exist_ok=True)
    value_path.parent.mkdir(parents=True, exist_ok=True)

    counts = _parse_counts(args.tokens_per_expert)
    routes = _build_routes(args.tokens, args.top_k, counts)
    observed_counts = _route_counts(routes, len(counts))
    benchmark_shapes = _write_benchmark(bench, counts, args.n, args.k)
    pairs = _pair_order(routes, len(counts))
    grouped = _grouped_order(pairs, len(counts))

    cutlass_run, parsed = _run_cutlass(args, value_path, bench)
    groups_seen = _parse_groups_seen(
        cutlass_run["run_result"].get("stdout_tail", ""),
        cutlass_run["run_result"].get("stderr_tail", ""),
    )
    records_by_schedule = _parse_value_records(value_path) if value_path.exists() else {}
    values = _value_report(records_by_schedule, pairs, counts, args.tokens, args.top_k) if records_by_schedule else {}
    schedule_names = sorted(values)
    schedule_passes = {schedule: all((data.get("passes") or {}).values()) for schedule, data in values.items()}
    build_ok = bool((cutlass_run.get("build_result", {}).get("build") or {}).get("ok")) if args.build else False
    n_gateup_even = args.n % 2 == 0
    passes = {
        "cutlass_dir_exists": cutlass_dir.exists(),
        "example_79d_exists": example.exists(),
        "binary_exists": binary.exists(),
        "build_invoked": bool(args.build),
        "build_succeeded": build_ok,
        "d_row_value_patch_applied": bool((cutlass_run.get("patch") or {}).get("applied")),
        "d_row_value_patch_restored": bool((cutlass_run.get("patch") or {}).get("restored")),
        "run_succeeded": bool(cutlass_run["run_result"].get("ok")),
        "no_noop_device_gate": parsed["no_noop_device_gate"],
        "cooperative_passed": parsed["cooperative_seen"],
        "pingpong_passed": parsed["pingpong_seen"],
        "host_reference_seen": parsed["host_reference_seen"],
        "dispositions_passed_count_ge_2": parsed["disposition_passed_count"] >= 2,
        "groups_seen_match_experts": groups_seen == len(counts),
        "route_tokens_match": len(routes) == args.tokens,
        "tokens_per_expert_match": observed_counts == counts,
        "grouped_order_complete": len(grouped) == args.tokens * args.top_k,
        "n_gateup_even_for_swiglu": n_gateup_even,
        "value_file_exists": value_path.exists(),
        "schedules_captured": set(schedule_names) == {"cooperative", "pingpong"},
        "schedule_values_passed": bool(schedule_passes) and all(schedule_passes.values()),
        "banked_gateup_value_materialization": False,
        "banked_host_swiglu_checksum_smoke": False,
        "banked_down_projection_numeric_parity": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }
    required = [
        "cutlass_dir_exists",
        "example_79d_exists",
        "binary_exists",
        "build_invoked",
        "build_succeeded",
        "d_row_value_patch_applied",
        "d_row_value_patch_restored",
        "run_succeeded",
        "no_noop_device_gate",
        "cooperative_passed",
        "pingpong_passed",
        "host_reference_seen",
        "dispositions_passed_count_ge_2",
        "groups_seen_match_experts",
        "route_tokens_match",
        "tokens_per_expert_match",
        "grouped_order_complete",
        "n_gateup_even_for_swiglu",
        "value_file_exists",
        "schedules_captured",
        "schedule_values_passed",
    ]
    passes["banked_gateup_value_materialization"] = all(bool(passes[key]) for key in required)
    passes["banked_host_swiglu_checksum_smoke"] = bool(passes["banked_gateup_value_materialization"])
    passes["all"] = bool(passes["banked_gateup_value_materialization"])
    decision = "PASS_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE" if passes["all"] else "FAIL_R5C3B_GATEUP_VALUE_MATERIALIZATION_SMOKE"
    return {
        "schema": "lynn-stage6-r5c3b-gateup-value-materialization-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "cutlass_dir": str(cutlass_dir),
        "build_dir": str(build_dir),
        "binary": str(binary),
        "example": str(example),
        "benchmark_file": str(bench),
        "d_row_value_file": str(value_path),
        "git": _git(cutlass_dir),
        "selected_expert_shape": {
            "tokens": args.tokens,
            "top_k": args.top_k,
            "experts": len(counts),
            "tokens_per_expert": counts,
            "n_gate_up": args.n,
            "k_hidden": args.k,
            "alignment": TARGET_ALIGNMENT,
            "benchmark_shapes": benchmark_shapes,
        },
        "route_sample": routes[: min(16, len(routes))],
        "grouped_order_sample": [pair.__dict__ for pair in grouped[: min(16, len(grouped))]],
        "route_counts_observed": observed_counts,
        "cutlass_run": cutlass_run,
        "run_parse": parsed,
        "groups_seen": groups_seen,
        "value_schedules": values,
        "passes": passes,
        "promotion_boundary": {
            "gateup_value_materialization": bool(passes["banked_gateup_value_materialization"]),
            "host_swiglu_checksum_smoke": bool(passes["banked_host_swiglu_checksum_smoke"]),
            "down_projection_numeric_parity": False,
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "default_runtime": False,
        },
        "caveats": [
            "This emits full real CUTLASS D/ref row values for a small gate/up smoke shape.",
            "It records a host-side SwiGLU checksum only; down projection is out of scope.",
            "Runtime/TFLOPS are trace-only; R5-C3B does not bank full grouped-MoE speed.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutlass-dir", default="/root/autodl-tmp/src/cutlass")
    ap.add_argument("--build-dir", default="/root/autodl-tmp/src/cutlass/build-r5c3b-sm120a-tools-on")
    ap.add_argument("--cuda-home", default="/usr/local/cuda-12.8")
    ap.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-file", default="")
    ap.add_argument("--value-file", default="")
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--clean-build", action="store_true")
    ap.add_argument("--no-atomic-scope-patch", action="store_true")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--tokens-per-expert", default=DEFAULT_COUNTS)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--norm-constant", type=float, default=1.0)
    args = ap.parse_args()
    data = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": data["decision"], "passes": data["passes"]}, indent=2))
    return 0 if data["passes"]["all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
