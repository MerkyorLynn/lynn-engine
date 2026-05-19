#!/usr/bin/env python3
"""P175: candidate recurrent kernel that reads q/k/v from out_conv directly."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _linear, _linear_conv_update_decode  # noqa: E402
from engine.qwen36_linear_attn_block import (  # noqa: E402
    HEAD_K_DIM,
    HEAD_V_DIM,
    KEY_DIM,
    NUM_K_HEADS,
    NUM_V_HEADS,
    VALUE_DIM,
)
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare_from_outconv_gqa  # noqa: E402


OUTPUT_KEYS = ["core_attn_out", "recurrent_state_out", "conv_state_out", "z"]


def _load_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P169 fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != "lynn-qwen36-linear-core-fixture-v1":
        raise ValueError(f"unexpected P169 fixture schema: {schema!r}")
    return manifest


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _candidate_boundary(
    tensors: dict[str, torch.Tensor],
    w: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    h_norm = tensors["h_norm"]
    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")
    times: dict[str, float] = {}

    _sync(device)
    total_start = time.perf_counter()

    start = time.perf_counter()
    proj_all = _linear(h_norm, w[fused_key])
    _sync(device)
    times["in_proj"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    mixed_new, z_raw, b_raw, a_raw = torch.split(
        proj_all,
        [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
        dim=-1,
    )
    z = z_raw.reshape(h_norm.shape[0], 1, NUM_V_HEADS, HEAD_V_DIM)
    _sync(device)
    times["split_z"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    out_conv, conv_state_out = _linear_conv_update_decode(
        mixed_new.transpose(1, 2),
        tensors["conv_state_in"],
        w["linear_attn.conv1d.weight"],
    )
    _sync(device)
    times["conv"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    beta = b_raw.sigmoid()
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -w["linear_attn.A_log"].float().exp()
    g = neg_exp_A_log * F.softplus(a_raw.float() + w["linear_attn.dt_bias"].float())
    _sync(device)
    times["gate"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare_from_outconv_gqa(
        out_conv,
        g,
        beta,
        tensors["recurrent_state_in"].clone(),
    )
    _sync(device)
    times["recurrent_from_outconv"] = (time.perf_counter() - start) * 1000.0
    times["total"] = (time.perf_counter() - total_start) * 1000.0
    return {
        "core_attn_out": core_attn_out.detach().clone(),
        "recurrent_state_out": recurrent_state_out.detach().clone(),
        "conv_state_out": conv_state_out.detach().clone(),
        "z": z.detach().clone(),
    }, times


def emit(args: argparse.Namespace) -> dict[str, Any]:
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
    for item in manifest["fixtures"]:
        fixture_file = str(item["file"])
        layer = int(item["layer_id"])
        tensors = load_file(str(fixtures_dir / fixture_file), device=args.device)
        candidate: dict[str, torch.Tensor] | None = None
        for _ in range(args.warmup_runs):
            candidate, _ = _candidate_boundary(tensors, runner.layer_weights[layer], args.device)
        samples: list[dict[str, float]] = []
        for _ in range(max(args.timed_runs, 1)):
            candidate, times = _candidate_boundary(tensors, runner.layer_weights[layer], args.device)
            samples.append(times)
        assert candidate is not None
        save_file({key: candidate[key].detach().contiguous().cpu() for key in OUTPUT_KEYS}, str(out_dir / fixture_file))
        rows.append(
            {
                "fixture_file": fixture_file,
                "layer_id": layer,
                "prompt_id": int(item["prompt_id"]),
                "stage_ms_median": {key: float(median(sample[key] for sample in samples)) for key in samples[0]},
                "stage_ms_samples": samples,
                "keys_written": list(OUTPUT_KEYS),
            }
        )
    stage_names = list(rows[0]["stage_ms_median"]) if rows else []
    stage_summary = {
        key: {
            "median": float(median(float(row["stage_ms_median"][key]) for row in rows)),
            "mean": sum(float(row["stage_ms_median"][key]) for row in rows) / max(len(rows), 1),
            "min": min((float(row["stage_ms_median"][key]) for row in rows), default=0.0),
            "max": max((float(row["stage_ms_median"][key]) for row in rows), default=0.0),
        }
        for key in stage_names
    }
    report = {
        "schema": "lynn-qwen36-recurrent-from-outconv-candidate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "fixtures": str(fixtures_dir),
        "candidate_output_dir": str(out_dir),
        "device": args.device,
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "summary": {
            "total_fixtures": len(rows),
            "total_tensors_written": len(rows) * len(OUTPUT_KEYS),
            "keys_written": list(OUTPUT_KEYS),
            "stage_ms": stage_summary,
        },
        "results": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _p169_check_command(args: argparse.Namespace) -> list[str]:
    p169_script = Path(args.p169_script)
    if not p169_script.is_absolute():
        p169_script = ROOT / p169_script
    return [
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


def _run_p169_check(args: argparse.Namespace) -> dict[str, Any]:
    p169_out = Path(args.p169_out)
    p169_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = _p169_check_command(args)
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe recurrent-from-outconv candidate on P169 fixtures.")
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--candidate-output-dir", "--out", dest="candidate_output_dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    parser.add_argument("--p169-script", default="benchmarks/p169_qwen36_linear_core_fixture_contract.py")
    parser.add_argument("--p169-out", default="")
    parser.add_argument("--max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.999999)
    args = parser.parse_args()
    if not args.p169_out:
        args.p169_out = str(Path(args.candidate_output_dir) / "p169_recurrent_from_outconv_check.json")
    report = emit(args)
    report["p169_check_command"] = _p169_check_command(args)
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
        "p169_check_returncode": report.get("p169_check", {}).get("returncode"),
        "report": args.report or str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return int(report.get("p169_check", {}).get("returncode", 1))


if __name__ == "__main__":
    raise SystemExit(main())
