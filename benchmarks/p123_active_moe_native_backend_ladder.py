#!/usr/bin/env python3
"""P123: native active-MoE backend ladder for local parity and runtime gates.

Runs a compact local-vs-runtime comparison for a list of native active-MoE
backends using the same prompt/layer slices and exact-greedy prompts. This is a
Stream-A control surface so strict-fused-boundary can be judged against the
already-known grouped-per16_nonatomic line without editing multiple scripts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BACKENDS = [
    "strict_fused_boundary",
    "grouped_per16_nonatomic",
    "cuda_scalar_contract",
]


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_local(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    return {
        "min_cosine_vs_triton": summary.get("min_cosine_vs_triton"),
        "max_rel_l2_vs_triton": summary.get("max_rel_l2_vs_triton"),
        "max_abs_vs_triton": summary.get("max_abs_vs_triton"),
        "pass": report.get("pass", report.get("subkernel_contract_pass")),
    }


def _summarize_runtime(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("candidate", {})
    summary = candidate.get("summary", {})
    return {
        "new_ids_all_match": report.get("new_ids_all_match"),
        "decode_tps_median": summary.get("decode_tps_median"),
        "median_speedup": report.get("median_speedup"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default="reports/qwen36_35b")
    ap.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 8, 14, 20, 28, 36])
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--linear-block-graph", choices=("0", "1"), default="1")
    ap.add_argument("--native-active-moe-layers", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for backend in args.backends:
        safe_backend = backend.replace("/", "_")
        local_path = out_dir / f"p123_{safe_backend}_local.json"
        runtime_path = out_dir / f"p123_{safe_backend}_runtime.json"

        if backend == "strict_fused_boundary":
            local_cmd = [
                sys.executable,
                "benchmarks/p121_active_moe_strict_boundary_probe.py",
                "--model",
                args.model,
                "--out",
                str(local_path),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--layers",
                *[str(x) for x in args.layers],
            ]
        elif backend == "grouped_per16_nonatomic":
            local_cmd = [
                sys.executable,
                "benchmarks/p73_grouped_per16_nonatomic_reference_probe.py",
                "--model",
                args.model,
                "--out",
                str(local_path),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--layers",
                *[str(x) for x in args.layers],
            ]
        elif backend == "cuda_scalar_contract":
            local_cmd = [
                sys.executable,
                "benchmarks/p45_native_active_moe_contract_gate.py",
                "--model",
                args.model,
                "--out",
                str(local_path),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--layers",
                *[str(x) for x in args.layers],
            ]
        else:
            raise ValueError(f"unsupported backend for p123 ladder: {backend}")

        runtime_cmd = [
            sys.executable,
            "benchmarks/p122_active_moe_strict_boundary_generate_gate.py",
            "--model",
            args.model,
            "--out",
            str(runtime_path),
            "--baseline-backend",
            "triton",
            "--candidate-backend",
            backend,
            "--max-new",
            str(args.max_new),
            "--linear-block-graph",
            args.linear_block_graph,
        ]
        if args.native_active_moe_layers is not None:
            runtime_cmd.extend(["--native-active-moe-layers", args.native_active_moe_layers])

        local_exec = _run(local_cmd)
        runtime_exec = _run(runtime_cmd)
        local_report = _load_json(local_path) if local_path.exists() else None
        runtime_report = _load_json(runtime_path) if runtime_path.exists() else None
        rows.append(
            {
                "backend": backend,
                "local_report": str(local_path),
                "runtime_report": str(runtime_path),
                "local_exec": {
                    "returncode": local_exec["returncode"],
                    "stderr_tail": local_exec["stderr"][-4000:],
                },
                "runtime_exec": {
                    "returncode": runtime_exec["returncode"],
                    "stderr_tail": runtime_exec["stderr"][-4000:],
                },
                "local_summary": None if local_report is None else _summarize_local(local_report),
                "runtime_summary": None if runtime_report is None else _summarize_runtime(runtime_report),
            }
        )

    result = {
        "schema_version": "lynn-engine-p123-active-moe-native-backend-ladder-v1",
        "model": args.model,
        "backends": args.backends,
        "layers": args.layers,
        "linear_block_graph": args.linear_block_graph == "1",
        "native_active_moe_layers": args.native_active_moe_layers,
        "rows": rows,
        "decision": (
            "Use this ladder to compare strict_fused_boundary against existing native MoE baselines. "
            "Only backends with green local parity and green exact-greedy runtime parity should advance."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
