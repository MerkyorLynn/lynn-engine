#!/usr/bin/env python3
"""Stage 6 R5-C2 selected-expert gate/up numeric smoke.

This gate bridges CUTLASS 79d's SM120 native NVF4 + UE4M3 grouped GEMM to a
minimal selected-expert MoE gate/up shape. It encodes deterministic
tokens_per_expert counts as per-group benchmark shapes:

  M = selected tokens for one expert
  N = gate/up output width for the smoke shape
  K = hidden input width for the smoke shape

It banks only selected-expert numeric smoke with CUTLASS host-reference
verification. It does not bank a full Lynn grouped-MoE kernel, speed, or runtime
default promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from r6000_stage6_r5c1_cutlass_numeric_smoke import (  # noqa: E402
    _binary_path,
    _configure_and_build,
    _env,
    _git,
    _parse_run,
    _run,
)


EXAMPLE_REL = "examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu"
TARGET_ALIGNMENT = 32
DEFAULT_COUNTS = "32,64,64,96"


def _parse_counts(raw: str) -> list[int]:
    counts = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not counts:
        raise ValueError("tokens_per_expert list is empty")
    if any(count <= 0 for count in counts):
        raise ValueError("tokens_per_expert counts must be positive")
    return counts


def _build_routes(tokens: int, top_k: int, counts: list[int]) -> list[list[int]]:
    if tokens <= 0 or top_k <= 0:
        raise ValueError("tokens and top_k must be positive")
    if sum(counts) != tokens * top_k:
        raise ValueError(
            f"sum(tokens_per_expert)={sum(counts)} must equal tokens*top_k={tokens * top_k}"
        )

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
                raise ValueError("could not build unique top-k routing from requested counts")
            row.append(expert)
            remaining[expert] -= 1
        routes.append(row)

    if any(remaining):
        raise ValueError(f"route construction left unmatched counts: {remaining}")
    return routes


def _route_counts(routes: list[list[int]], experts: int) -> list[int]:
    counts = [0] * experts
    for row in routes:
        for expert in row:
            counts[expert] += 1
    return counts


def _aligned(values: list[int], alignment: int = TARGET_ALIGNMENT) -> bool:
    return all(value % alignment == 0 for value in values)


def _write_benchmark(path: Path, counts: list[int], n: int, k: int) -> list[dict[str, int]]:
    shapes = []
    with path.open("w", encoding="utf-8") as handle:
        for expert, count in enumerate(counts):
            handle.write(f"{expert} {count}x{n}x{k}\n")
            shapes.append({"expert": expert, "m": count, "n": n, "k": k})
    return shapes


def _parse_groups_seen(stdout: str, stderr: str) -> int | None:
    text = f"{stdout}\n{stderr}"
    for line in text.splitlines():
        if "Groups" in line and ":" in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                continue
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    cutlass_dir = Path(args.cutlass_dir).expanduser().resolve()
    build_dir = Path(args.build_dir).expanduser().resolve()
    binary = _binary_path(build_dir)
    example = cutlass_dir / EXAMPLE_REL
    out = Path(args.out).expanduser().resolve()
    bench = Path(args.benchmark_file).expanduser().resolve() if args.benchmark_file else out.with_suffix(".benchmark.txt")
    bench.parent.mkdir(parents=True, exist_ok=True)

    counts = _parse_counts(args.tokens_per_expert)
    routes = _build_routes(args.tokens, args.top_k, counts)
    observed_counts = _route_counts(routes, len(counts))
    benchmark_shapes = _write_benchmark(bench, counts, args.n, args.k)

    build_result: dict[str, Any] = {"skipped": True}
    if args.build:
        build_result = _configure_and_build(
            cutlass_dir=cutlass_dir,
            build_dir=build_dir,
            cuda_home=args.cuda_home,
            python_bin=args.python_bin,
            timeout_s=args.timeout_s,
            clean_build=args.clean_build,
            atomic_scope_patch=not args.no_atomic_scope_patch,
        )

    run_cmd = [
        str(binary),
        f"--benchmark={bench}",
        f"--iterations={args.iterations}",
        "--alpha=1",
        "--beta=0",
        f"--norm_constant={args.norm_constant}",
    ]
    if binary.exists():
        run_result = _run(run_cmd, cwd=build_dir, env=_env(args.cuda_home, args.python_bin), timeout_s=args.timeout_s)
    else:
        run_result = {"ok": False, "returncode": None, "stdout_tail": "", "stderr_tail": "binary missing", "cmd": run_cmd}

    parsed = _parse_run(run_result.get("stdout_tail", ""), run_result.get("stderr_tail", ""))
    groups_seen = _parse_groups_seen(run_result.get("stdout_tail", ""), run_result.get("stderr_tail", ""))
    build_ok = bool((build_result.get("build") or {}).get("ok")) if args.build else binary.exists()
    route_topk_unique = all(len(set(row)) == args.top_k for row in routes)
    route_tokens_match = len(routes) == args.tokens
    tokens_per_expert_match = observed_counts == counts
    benchmark_shapes_aligned = _aligned(counts + [args.n, args.k])

    passes = {
        "cutlass_dir_exists": cutlass_dir.exists(),
        "example_79d_exists": example.exists(),
        "binary_exists": binary.exists(),
        "build_invoked": bool(args.build),
        "build_succeeded": build_ok,
        "route_tokens_match": route_tokens_match,
        "route_topk_unique": route_topk_unique,
        "tokens_per_expert_match": tokens_per_expert_match,
        "benchmark_groups_match_experts": len(benchmark_shapes) == len(counts),
        "benchmark_shapes_aligned_32": benchmark_shapes_aligned,
        "groups_seen_match_experts": groups_seen == len(counts),
        "run_succeeded": bool(run_result.get("ok")),
        "no_noop_device_gate": parsed["no_noop_device_gate"],
        "cooperative_passed": parsed["cooperative_seen"],
        "pingpong_passed": parsed["pingpong_seen"],
        "host_reference_seen": parsed["host_reference_seen"],
        "dispositions_passed_count_ge_2": parsed["disposition_passed_count"] >= 2,
        "banked_selected_expert_gate_up_smoke": False,
        "banked_grouped_moe_fp4_mma_poc": False,
        "banked_kernel_speed": False,
        "banked_default_promotion": False,
    }
    required = [
        "cutlass_dir_exists",
        "example_79d_exists",
        "binary_exists",
        "build_succeeded",
        "route_tokens_match",
        "route_topk_unique",
        "tokens_per_expert_match",
        "benchmark_groups_match_experts",
        "benchmark_shapes_aligned_32",
        "groups_seen_match_experts",
        "run_succeeded",
        "no_noop_device_gate",
        "cooperative_passed",
        "pingpong_passed",
        "host_reference_seen",
        "dispositions_passed_count_ge_2",
    ]
    passes["banked_selected_expert_gate_up_smoke"] = all(bool(passes[key]) for key in required)
    passes["all"] = bool(passes["banked_selected_expert_gate_up_smoke"])
    decision = (
        "PASS_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE"
        if passes["all"]
        else "FAIL_R5C2_SELECTED_EXPERT_GATEUP_NUMERIC_SMOKE"
    )
    return {
        "schema": "lynn-stage6-r5c2-selected-expert-gateup-smoke-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutlass_dir": str(cutlass_dir),
        "build_dir": str(build_dir),
        "binary": str(binary),
        "example": str(example),
        "benchmark_file": str(bench),
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
        "route_counts_observed": observed_counts,
        "build_result": build_result,
        "run_result": run_result,
        "run_parse": parsed,
        "groups_seen": groups_seen,
        "passes": passes,
        "decision": decision,
        "promotion_boundary": {
            "selected_expert_gate_up_numeric_smoke": bool(passes["banked_selected_expert_gate_up_smoke"]),
            "grouped_moe_fp4_mma_poc": False,
            "kernel_speed": False,
            "default_runtime": False,
        },
        "caveats": [
            "This maps tokens_per_expert to CUTLASS 79d per-group M shapes; it does not implement Lynn gather/scatter.",
            "This covers selected-expert gate/up numeric smoke only, not down projection or full MoE.",
            "Avg runtime is recorded for traceability only; R5-C2 does not bank speed.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutlass-dir", default="/root/autodl-tmp/src/cutlass")
    ap.add_argument("--build-dir", default="/root/autodl-tmp/src/cutlass/build-r5c1-sm120a-tools-on")
    ap.add_argument("--cuda-home", default="/usr/local/cuda-12.8")
    ap.add_argument("--python-bin", default="/root/miniconda3/bin/python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark-file", default="")
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
