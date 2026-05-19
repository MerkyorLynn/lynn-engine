#!/usr/bin/env python3
"""P176: larger recurrent candidate computing beta/g inside Triton."""
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
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.incremental_decode import _linear, _linear_conv_update_decode  # noqa: E402
from engine.qwen36_linear_attn_block import HEAD_V_DIM, KEY_DIM, NUM_V_HEADS, VALUE_DIM  # noqa: E402
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare_from_outconv_ab_gqa  # noqa: E402


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


def _candidate(tensors: dict[str, torch.Tensor], w: dict[str, Any], device: str) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
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
    times["split_zab"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    out_conv, conv_state_out = _linear_conv_update_decode(
        mixed_new.transpose(1, 2),
        tensors["conv_state_in"],
        w["linear_attn.conv1d.weight"],
    )
    _sync(device)
    times["conv"] = (time.perf_counter() - start) * 1000.0

    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -w["linear_attn.A_log"].float().exp()
    start = time.perf_counter()
    core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare_from_outconv_ab_gqa(
        out_conv,
        a_raw,
        b_raw,
        neg_exp_A_log,
        w["linear_attn.dt_bias"].float(),
        tensors["recurrent_state_in"].clone(),
    )
    _sync(device)
    times["recurrent_from_outconv_ab"] = (time.perf_counter() - start) * 1000.0
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
    runner = LynnIncrementalRunner(model_path, device=args.device, dtype=torch.bfloat16, max_seq_len=args.max_seq_len, verbose=False)
    rows: list[dict[str, Any]] = []
    for item in manifest["fixtures"]:
        fixture_file = str(item["file"])
        layer = int(item["layer_id"])
        tensors = load_file(str(fixtures_dir / fixture_file), device=args.device)
        candidate: dict[str, torch.Tensor] | None = None
        for _ in range(args.warmup_runs):
            candidate, _ = _candidate(tensors, runner.layer_weights[layer], args.device)
        samples: list[dict[str, float]] = []
        for _ in range(max(args.timed_runs, 1)):
            candidate, times = _candidate(tensors, runner.layer_weights[layer], args.device)
            samples.append(times)
        assert candidate is not None
        save_file({key: candidate[key].detach().contiguous().cpu() for key in OUTPUT_KEYS}, str(out_dir / fixture_file))
        rows.append({
            "fixture_file": fixture_file,
            "layer_id": layer,
            "prompt_id": int(item["prompt_id"]),
            "stage_ms_median": {key: float(median(sample[key] for sample in samples)) for key in samples[0]},
            "stage_ms_samples": samples,
            "keys_written": list(OUTPUT_KEYS),
        })
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
        "schema": "lynn-qwen36-recurrent-from-outconv-ab-candidate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "fixtures": str(fixtures_dir),
        "candidate_output_dir": str(out_dir),
        "device": args.device,
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


def _run_p169(args: argparse.Namespace) -> dict[str, Any]:
    p169_script = ROOT / "benchmarks/p169_qwen36_linear_core_fixture_contract.py"
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
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    result: dict[str, Any] = {"command": cmd, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    p169_out = Path(args.p169_out)
    if p169_out.exists():
        result["summary"] = json.loads(p169_out.read_text(encoding="utf-8")).get("contract", {}).get("summary")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe outconv+a/b recurrent candidate on P169 fixtures.")
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--candidate-output-dir", "--out", dest="candidate_output_dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--p169-out", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    args = parser.parse_args()
    if not args.p169_out:
        args.p169_out = str(Path(args.candidate_output_dir) / "p169_recurrent_from_outconv_ab_check.json")
    report = emit(args)
    report["p169_check"] = _run_p169(args)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (Path(args.candidate_output_dir) / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_output_dir": report["candidate_output_dir"],
        "summary": report["summary"],
        "p169_check_report": args.p169_out,
        "p169_check_summary": report.get("p169_check", {}).get("summary"),
        "report": args.report or str(Path(args.candidate_output_dir) / "manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

