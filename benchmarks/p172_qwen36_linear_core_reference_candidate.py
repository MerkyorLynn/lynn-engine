#!/usr/bin/env python3
"""P172: emit computed reference candidate outputs for P169 linear-core fixtures.

P171 proves the candidate-output-dir plumbing with identity copies.  P172 is the
next rung: it reloads the model once, recomputes the linear-core reference from
each fixture's input tensors, writes candidate safetensors, and then runs the
same P169 candidate gate.  Future fused kernels should beat this timing while
preserving the same output contract.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from statistics import median
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.p169_qwen36_linear_core_fixture_contract import (  # noqa: E402
    CHECK_KEYS,
    _linear_core_reference,
)
from benchmarks.p171_qwen36_linear_core_candidate_output_smoke import (  # noqa: E402
    P169_FINAL_KEYS,
    P169_REFERENCE_KEY_ORDER,
)
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402


def _load_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P169 fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != "lynn-qwen36-linear-core-fixture-v1":
        raise ValueError(f"unexpected P169 fixture schema: {schema!r}")
    return manifest


def _requested_keys(args: argparse.Namespace, candidate: dict[str, torch.Tensor]) -> list[str]:
    if args.only_final:
        keys = list(P169_FINAL_KEYS)
    elif args.keys:
        keys = [part.strip() for part in args.keys.split(",") if part.strip()]
    else:
        keys = [key for key in P169_REFERENCE_KEY_ORDER if key in candidate]
    missing = [key for key in keys if key not in candidate]
    if missing:
        raise KeyError(f"computed candidate is missing requested keys: {missing}")
    return keys


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def emit_reference_candidates(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    out_dir = Path(args.candidate_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(fixtures_dir)

    model_path = args.model or manifest.get("model")
    if not model_path:
        raise ValueError("--model is required when fixture manifest has no model")

    runner = LynnIncrementalRunner(
        model_path,
        device=args.device,
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        verbose=False,
    )

    rows: list[dict[str, Any]] = []
    per_fixture_medians: list[float] = []
    t0 = time.time()
    for item in manifest["fixtures"]:
        fixture_file = str(item["file"])
        layer = int(item["layer_id"])
        tensors = load_file(str(fixtures_dir / fixture_file), device=args.device)
        candidate: dict[str, torch.Tensor] | None = None
        for _ in range(args.warmup_runs):
            candidate = _linear_core_reference(
                tensors["h_norm"],
                runner.layer_weights[layer],
                tensors["recurrent_state_in"],
                tensors["conv_state_in"],
            )
        samples_ms: list[float] = []
        timed_runs = max(args.timed_runs, 1)
        for _ in range(timed_runs):
            _sync_if_cuda(args.device)
            start = time.perf_counter()
            candidate = _linear_core_reference(
                tensors["h_norm"],
                runner.layer_weights[layer],
                tensors["recurrent_state_in"],
                tensors["conv_state_in"],
            )
            _sync_if_cuda(args.device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        assert candidate is not None
        elapsed_ms = sum(samples_ms) / len(samples_ms)
        median_ms = float(median(samples_ms))
        per_fixture_medians.append(median_ms)

        keys = _requested_keys(args, candidate)
        candidate_tensors = {key: candidate[key].detach().contiguous().cpu() for key in keys}
        save_file(candidate_tensors, str(out_dir / fixture_file))
        rows.append(
            {
                "fixture_file": fixture_file,
                "layer_id": layer,
                "prompt_id": int(item["prompt_id"]),
                "compute_ms_mean": elapsed_ms,
                "compute_ms_median": median_ms,
                "compute_ms_samples": samples_ms,
                "keys_written": keys,
                "tensor_shapes": {key: list(candidate_tensors[key].shape) for key in keys},
                "tensor_dtypes": {key: str(candidate_tensors[key].dtype) for key in keys},
            }
        )

    mean_compute_ms = sum(float(row["compute_ms_mean"]) for row in rows) / max(len(rows), 1)
    median_compute_ms = float(median(per_fixture_medians)) if per_fixture_medians else 0.0
    report = {
        "schema": "lynn-qwen36-linear-core-reference-candidate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "fixtures": str(fixtures_dir),
        "candidate_output_dir": str(out_dir),
        "device": args.device,
        "only_final": bool(args.only_final),
        "custom_keys": args.keys,
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "elapsed_seconds": time.time() - t0,
        "summary": {
            "total_fixtures": len(rows),
            "total_tensors_written": sum(len(row["keys_written"]) for row in rows),
            "keys_mode": "only-final" if args.only_final else ("custom" if args.keys else "reference-computed"),
            "all_checked_keys_written": all(all(key in row["keys_written"] for key in CHECK_KEYS) for row in rows),
            "compute_ms_mean": mean_compute_ms,
            "compute_ms_median_of_medians": median_compute_ms,
            "compute_ms_min": min((float(row["compute_ms_median"]) for row in rows), default=0.0),
            "compute_ms_max": max((float(row["compute_ms_median"]) for row in rows), default=0.0),
        },
        "results": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _p169_check_command(args: argparse.Namespace) -> list[str]:
    p169_script = Path(args.p169_script)
    if not p169_script.is_absolute():
        p169_script = ROOT / p169_script
    cmd = [
        sys.executable,
        str(p169_script),
        "--fixtures",
        str(Path(args.fixtures)),
        "--candidate-output-dir",
        str(Path(args.candidate_output_dir)),
        "--out",
        str(Path(args.p169_out)),
        "--check",
        "--device",
        args.device,
        "--max-abs-threshold",
        str(args.max_abs_threshold),
        "--cosine-threshold",
        str(args.cosine_threshold),
    ]
    if args.require_all_keys:
        cmd.append("--require-all-keys")
    return cmd


def _run_p169_check(args: argparse.Namespace) -> dict[str, Any]:
    p169_out = Path(args.p169_out)
    p169_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _p169_check_command(args)
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result: dict[str, Any] = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "report": str(p169_out),
    }
    if p169_out.exists():
        p169_report = json.loads(p169_out.read_text(encoding="utf-8"))
        result["summary"] = p169_report.get("contract", {}).get("summary")
    if completed.returncode != 0:
        raise RuntimeError(f"P169 reference-candidate check failed with exit code {completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit computed reference candidates for P169 linear-core fixtures.")
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--candidate-output-dir", "--out", dest="candidate_output_dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--only-final", action="store_true")
    parser.add_argument("--keys", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--skip-p169-check", action="store_true")
    parser.add_argument("--p169-script", default="benchmarks/p169_qwen36_linear_core_fixture_contract.py")
    parser.add_argument("--p169-out", default="")
    parser.add_argument("--require-all-keys", action="store_true")
    parser.add_argument("--max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.999999)
    args = parser.parse_args()

    if args.only_final and args.keys:
        raise ValueError("--only-final and --keys are mutually exclusive")
    if args.only_final and args.require_all_keys:
        raise ValueError("--only-final cannot be combined with --require-all-keys")
    if not args.p169_out:
        args.p169_out = str(Path(args.candidate_output_dir) / "p169_reference_candidate_check.json")

    report = emit_reference_candidates(args)
    report["p169_check_command"] = _p169_check_command(args)
    if args.skip_p169_check:
        report["p169_check"] = {"skipped": True}
    else:
        report["p169_check"] = _run_p169_check(args)

    manifest_path = Path(args.candidate_output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "candidate_output_dir": report["candidate_output_dir"],
        "summary": report["summary"],
        "p169_check_report": args.p169_out,
        "p169_check_summary": report.get("p169_check", {}).get("summary"),
        "report": args.report or str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
