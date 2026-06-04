#!/usr/bin/env python3
"""Stage 6 R5-C2C real CUTLASS D-row slot scatter smoke.

R5-C2C is the first R5-C gate that touches real CUTLASS output rows. The script
temporarily patches CUTLASS example 79d to emit per-row D/ref digests after
host-reference verification, rebuilds the example, runs the selected-expert
shape, and scatters those real row digests back to Lynn selected slots using the
R5-C2B inverse-order contract.

This banks only a real D-row slot-scatter smoke. It does not bank an in-epilogue
selected-output CUDA kernel, SwiGLU/down projection, speed, server behavior, RC
quality, or runtime defaults.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
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


PATCH_MARKER = "LYNN_STAGE6_R5C2C_D_ROW_DIGEST_PATCH"


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


def _scatter_schedule(
    records: dict[tuple[int, int], dict[str, Any]],
    pair_order: list[Pair],
    tokens: int,
    top_k: int,
    *,
    use_ref: bool,
) -> list[list[str]]:
    selected: list[list[str | None]] = [[None for _slot in range(top_k)] for _token in range(tokens)]
    key_name = "ref_hash" if use_ref else "d_hash"
    for pair in pair_order:
        record = records.get((pair.expert_id, pair.group_row))
        if record is None:
            raise AssertionError(f"missing D-row record for expert={pair.expert_id} row={pair.group_row}")
        if selected[pair.token_idx][pair.top_k_slot] is not None:
            raise AssertionError(f"duplicate selected slot {(pair.token_idx, pair.top_k_slot)}")
        selected[pair.token_idx][pair.top_k_slot] = str(record[key_name])
    if any(value is None for row in selected for value in row):
        raise AssertionError("missing selected slot after real D-row scatter")
    return [[str(value) for value in row] for row in selected]


def _fault_injections(
    records: dict[tuple[int, int], dict[str, Any]],
    pair_order: list[Pair],
    tokens: int,
    top_k: int,
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    expected_ref = _scatter_schedule(records, pair_order, tokens, top_k, use_ref=True)

    swapped_records = {key: dict(value) for key, value in records.items()}
    keys = sorted(swapped_records)
    if len(keys) >= 2:
        swapped_records[keys[0]]["d_hash"], swapped_records[keys[1]]["d_hash"] = (
            swapped_records[keys[1]]["d_hash"],
            swapped_records[keys[0]]["d_hash"],
        )
    checks["swapped_d_rows_detected"] = (
        _scatter_schedule(swapped_records, pair_order, tokens, top_k, use_ref=False) != expected_ref
    )

    missing_records = {key: dict(value) for key, value in records.items()}
    if keys:
        del missing_records[keys[0]]
    try:
        _scatter_schedule(missing_records, pair_order, tokens, top_k, use_ref=False)
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
        _scatter_schedule(records, duplicate_pairs, tokens, top_k, use_ref=False)
        checks["duplicate_slot_rejected"] = False
    except AssertionError:
        checks["duplicate_slot_rejected"] = True
    return checks


def _patch_source_text(text: str) -> str:
    if PATCH_MARKER in text:
        return text
    text = text.replace(
        "#include <float.h>\n",
        "#include <float.h>\n#include <cstdint>\n#include <cstring>\n",
    )
    text = text.replace(
        "  std::string benchmark_path;\n",
        "  std::string benchmark_path;\n  std::string lynn_d_row_digest_path;\n",
    )
    text = text.replace(
        '    cmd.get_cmd_line_argument("benchmark", benchmark_path);\n',
        '    cmd.get_cmd_line_argument("benchmark", benchmark_path);\n'
        '    cmd.get_cmd_line_argument("lynn_d_row_digest", lynn_d_row_digest_path);\n',
    )
    helper = f"""
// {PATCH_MARKER}: emit real D/ref row digests for Lynn R5-C2C.
std::uint64_t lynn_fnv1a_u32(std::uint64_t h, std::uint32_t value) {{
  for (int byte = 0; byte < 4; ++byte) {{
    h ^= static_cast<std::uint8_t>((value >> (byte * 8)) & 0xff);
    h *= 1099511628211ull;
  }}
  return h;
}}

std::uint64_t lynn_hash_d_row(HostTensorD& tensor, int row, int cols) {{
  std::uint64_t h = 1469598103934665603ull;
  for (int col = 0; col < cols; ++col) {{
    float value = static_cast<float>(tensor.host_data(row * cols + col));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    h = lynn_fnv1a_u32(h, bits);
  }}
  return h;
}}

void lynn_emit_d_row_digests(Options const& options, char const* schedule_name, int group_idx, int rows, int cols) {{
  if (options.lynn_d_row_digest_path.empty()) {{
    return;
  }}
  std::ofstream out(options.lynn_d_row_digest_path, std::ios::app);
  for (int row = 0; row < rows; ++row) {{
    auto d_hash = lynn_hash_d_row(block_D.at(group_idx), row, cols);
    auto ref_hash = lynn_hash_d_row(block_ref_D.at(group_idx), row, cols);
    out << "{{\\\"schedule\\\":\\\"" << schedule_name
        << "\\\",\\\"group\\\":" << group_idx
        << ",\\\"row\\\":" << row
        << ",\\\"cols\\\":" << cols
        << ",\\\"d_hash\\\":\\\"" << d_hash
        << "\\\",\\\"ref_hash\\\":\\\"" << ref_hash
        << "\\\"}}" << std::endl;
  }}
}}

"""
    text = text.replace("bool verify(const Options &options) {\n", helper + "bool verify(const Options &options, char const* schedule_name) {\n")
    text = text.replace("    block_SFD.at(i).sync_host();\n", "    block_SFD.at(i).sync_host();\n    lynn_emit_d_row_digests(options, schedule_name, i, M, N);\n")
    text = text.replace(
        "int run(Options &options, bool host_problem_shapes_available = true)\n",
        "int run(Options &options, bool host_problem_shapes_available = true, char const* schedule_name = \"unknown\")\n",
    )
    text = text.replace("    result.passed = verify(options);\n", "    result.passed = verify(options, schedule_name);\n")
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
def _temporary_d_row_digest_patch(cutlass_dir: Path, enabled: bool) -> Iterator[dict[str, Any]]:
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
    except Exception as exc:  # pragma: no cover - surfaced in result.json on R6000
        info["error"] = str(exc)
        raise
    finally:
        if info["applied"]:
            source.write_text(original, encoding="utf-8")
            info["restored"] = True


def _parse_d_row_records(path: Path) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    by_schedule: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        schedule = str(record["schedule"])
        key = (int(record["group"]), int(record["row"]))
        schedule_records = by_schedule.setdefault(schedule, {})
        if key in schedule_records:
            raise AssertionError(f"duplicate D-row record for schedule={schedule} key={key}")
        schedule_records[key] = record
    return by_schedule


def _run_cutlass(args: argparse.Namespace, digest_path: Path, bench: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    build_result: dict[str, Any] = {"skipped": True}
    patch_info: dict[str, Any] = {"enabled": bool(args.build), "applied": False, "restored": False}
    if args.build:
        with _temporary_d_row_digest_patch(cutlass_dir, True) as patch_info:
            build_result = _configure_and_build(
                cutlass_dir=cutlass_dir,
                build_dir=build_dir,
                cuda_home=args.cuda_home,
                python_bin=args.python_bin,
                timeout_s=args.timeout_s,
                clean_build=args.clean_build,
                atomic_scope_patch=not args.no_atomic_scope_patch,
            )
    if digest_path.exists():
        digest_path.unlink()
    run_cmd = [
        str(binary),
        f"--benchmark={bench}",
        f"--iterations={args.iterations}",
        "--alpha=1",
        "--beta=0",
        f"--norm_constant={args.norm_constant}",
        f"--lynn_d_row_digest={digest_path}",
    ]
    if binary.exists():
        run_result = _run(run_cmd, cwd=build_dir, env=_env(args.cuda_home, args.python_bin), timeout_s=args.timeout_s)
    else:
        run_result = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": "binary missing", "cmd": run_cmd}
    return {"build_result": build_result, "patch": patch_info, "run_result": run_result}, _parse_run(
        run_result.get("stdout_tail", ""), run_result.get("stderr_tail", "")
    )


def _scatter_report(
    records_by_schedule: dict[str, dict[tuple[int, int], dict[str, Any]]],
    pair_order: list[Pair],
    expected_counts: list[int],
    tokens: int,
    top_k: int,
) -> dict[str, Any]:
    schedules: dict[str, Any] = {}
    for schedule, records in sorted(records_by_schedule.items()):
        row_counts = [sum(1 for expert, _row in records if expert == idx) for idx in range(len(expected_counts))]
        try:
            d_selected = _scatter_schedule(records, pair_order, tokens, top_k, use_ref=False)
            ref_selected = _scatter_schedule(records, pair_order, tokens, top_k, use_ref=True)
            fault = _fault_injections(records, pair_order, tokens, top_k)
            row_digest_pairs_match = all(record.get("d_hash") == record.get("ref_hash") for record in records.values())
            scatter_match = d_selected == ref_selected
            fault_ok = all(fault.values())
            error = ""
        except AssertionError as exc:
            row_digest_pairs_match = False
            scatter_match = False
            fault = {}
            fault_ok = False
            error = str(exc)
        schedules[schedule] = {
            "records": len(records),
            "row_counts": row_counts,
            "d_ref_row_digests_match": row_digest_pairs_match,
            "scatter_d_ref_match": scatter_match,
            "fault_checks": fault,
            "scatter_error": error,
            "passes": {
                "row_counts_match": row_counts == expected_counts,
                "d_ref_row_digests_match": row_digest_pairs_match,
                "scatter_d_ref_match": scatter_match,
                "fault_injections_detected": fault_ok,
                "scatter_error_absent": not error,
            },
        }
    return schedules


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    example = cutlass_dir / EXAMPLE_REL
    out = Path(args.out).expanduser().resolve()
    bench = Path(args.benchmark_file).expanduser().resolve() if args.benchmark_file else out.with_suffix(".benchmark.txt")
    digest_path = Path(args.digest_file).expanduser().resolve() if args.digest_file else out.with_suffix(".d_row_digest.jsonl")
    bench.parent.mkdir(parents=True, exist_ok=True)
    digest_path.parent.mkdir(parents=True, exist_ok=True)

    counts = _parse_counts(args.tokens_per_expert)
    routes = _build_routes(args.tokens, args.top_k, counts)
    observed_counts = _route_counts(routes, len(counts))
    benchmark_shapes = _write_benchmark(bench, counts, args.n, args.k)
    pairs = _pair_order(routes, len(counts))
    grouped = _grouped_order(pairs, len(counts))

    cutlass_run, parsed = _run_cutlass(args, digest_path, bench)
    groups_seen = _parse_groups_seen(
        cutlass_run["run_result"].get("stdout_tail", ""),
        cutlass_run["run_result"].get("stderr_tail", ""),
    )
    records_by_schedule = _parse_d_row_records(digest_path) if digest_path.exists() else {}
    scatter = _scatter_report(records_by_schedule, pairs, counts, args.tokens, args.top_k) if records_by_schedule else {}

    build_ok = bool((cutlass_run.get("build_result", {}).get("build") or {}).get("ok")) if args.build else False
    schedule_names = sorted(scatter)
    schedule_passes = {
        schedule: all((data.get("passes") or {}).values()) for schedule, data in scatter.items()
    }
    passes = {
        "cutlass_dir_exists": cutlass_dir.exists(),
        "example_79d_exists": example.exists(),
        "binary_exists": binary.exists(),
        "build_invoked": bool(args.build),
        "build_succeeded": build_ok,
        "d_row_digest_patch_applied": bool((cutlass_run.get("patch") or {}).get("applied")),
        "d_row_digest_patch_restored": bool((cutlass_run.get("patch") or {}).get("restored")),
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
        "digest_file_exists": digest_path.exists(),
        "schedules_captured": set(schedule_names) == {"cooperative", "pingpong"},
        "schedule_scatters_passed": bool(schedule_passes) and all(schedule_passes.values()),
        "banked_real_d_row_slot_scatter": False,
        "banked_selected_output_kernel_epilogue": False,
        "banked_swiglu_or_down_projection": False,
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
        "d_row_digest_patch_applied",
        "d_row_digest_patch_restored",
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
        "digest_file_exists",
        "schedules_captured",
        "schedule_scatters_passed",
    ]
    passes["banked_real_d_row_slot_scatter"] = all(bool(passes[key]) for key in required)
    passes["all"] = bool(passes["banked_real_d_row_slot_scatter"])
    decision = (
        "PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE"
        if passes["all"]
        else "FAIL_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE"
    )
    return {
        "schema": "lynn-stage6-r5c2c-real-d-row-slot-scatter-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision": decision,
        "cutlass_dir": str(cutlass_dir),
        "build_dir": str(build_dir),
        "binary": str(binary),
        "example": str(example),
        "benchmark_file": str(bench),
        "d_row_digest_file": str(digest_path),
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
        "scatter_schedules": scatter,
        "passes": passes,
        "promotion_boundary": {
            "real_d_row_slot_scatter": bool(passes["banked_real_d_row_slot_scatter"]),
            "selected_output_kernel_epilogue": False,
            "swiglu_or_down_projection": False,
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "default_runtime": False,
        },
        "caveats": [
            "This emits and scatters real CUTLASS D/ref row digests, not full D row tensors.",
            "Output shape is [T, top_k, N_gateup]; SwiGLU and down projection are out of scope.",
            "The selected-output scatter is host-side smoke evidence, not an in-epilogue CUDA kernel.",
            "Runtime/TFLOPS are recorded for traceability only; R5-C2C does not bank speed.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutlass-dir", default="/root/autodl-tmp/src/cutlass")
    ap.add_argument("--build-dir", default="/root/autodl-tmp/src/cutlass/build-r5c2c-sm120a-tools-on")
    ap.add_argument("--cuda-home", default="/usr/local/cuda-12.8")
    ap.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-file", default="")
    ap.add_argument("--digest-file", default="")
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
