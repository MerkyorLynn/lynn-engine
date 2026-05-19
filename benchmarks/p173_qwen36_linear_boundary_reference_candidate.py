#!/usr/bin/env python3
"""P173: reference candidate for the first linear-core fused boundary.

This narrows P172 from the full linear core to the intended first native
boundary: `in_proj -> conv -> recurrent/GDN`.  It emits only tensors that the
first fused kernel must reproduce before any resident integration:

  - core_attn_out
  - recurrent_state_out
  - conv_state_out
  - z (debug/downstream input, not part of P169 CHECK_KEYS)

The script is deliberately a benchmark/fixture producer only.  It does not
change serving defaults.
"""
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
    V_PER_K,
)
from engine.resident_runner import LynnIncrementalRunner  # noqa: E402
from triton_kernels.gated_delta import (  # noqa: E402
    recurrent_gated_delta_fused_prepare,
    recurrent_gated_delta_fused_prepare_gqa,
)


BOUNDARY_KEYS = ["core_attn_out", "recurrent_state_out", "conv_state_out", "z"]


def _load_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P169 fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != "lynn-qwen36-linear-core-fixture-v1":
        raise ValueError(f"unexpected P169 fixture schema: {schema!r}")
    return manifest


def _sync_if_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _linear_boundary_reference(
    h_norm: torch.Tensor,
    w: dict[str, Any],
    recurrent_state_in: torch.Tensor,
    conv_state_in: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")
    batch = h_norm.shape[0]
    use_gqa = V_PER_K > 1

    proj_all = _linear(h_norm, w[fused_key])
    mixed_new, z_raw, b_raw, a_raw = torch.split(
        proj_all,
        [KEY_DIM + KEY_DIM + VALUE_DIM, VALUE_DIM, NUM_V_HEADS, NUM_V_HEADS],
        dim=-1,
    )
    out_conv, conv_state_out = _linear_conv_update_decode(
        mixed_new.transpose(1, 2),
        conv_state_in.clone(),
        w["linear_attn.conv1d.weight"],
    )
    q, k, v = torch.split(out_conv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(batch, 1, NUM_K_HEADS, HEAD_K_DIM)
    k = k.reshape(batch, 1, NUM_K_HEADS, HEAD_K_DIM)
    v = v.reshape(batch, 1, NUM_V_HEADS, HEAD_V_DIM)
    z = z_raw.reshape(batch, 1, NUM_V_HEADS, HEAD_V_DIM)
    beta = b_raw.sigmoid()
    neg_exp_A_log = w.get("linear_attn._neg_exp_A_log")
    if neg_exp_A_log is None:
        neg_exp_A_log = -w["linear_attn.A_log"].float().exp()
    g = neg_exp_A_log * F.softplus(a_raw.float() + w["linear_attn.dt_bias"].float())
    state_work = recurrent_state_in.clone()
    if use_gqa:
        core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare_gqa(q, k, v, g, beta, state_work)
    else:
        if V_PER_K > 1:
            q = q.repeat_interleave(V_PER_K, dim=2)
            k = k.repeat_interleave(V_PER_K, dim=2)
        core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare(q, k, v, g, beta, state_work)
    return {
        "core_attn_out": core_attn_out.detach().clone(),
        "recurrent_state_out": recurrent_state_out.detach().clone(),
        "conv_state_out": conv_state_out.detach().clone(),
        "z": z.detach().clone(),
    }


def emit_boundary_candidates(args: argparse.Namespace) -> dict[str, Any]:
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
    t0 = time.time()
    for item in manifest["fixtures"]:
        fixture_file = str(item["file"])
        layer = int(item["layer_id"])
        tensors = load_file(str(fixtures_dir / fixture_file), device=args.device)
        candidate: dict[str, torch.Tensor] | None = None
        for _ in range(args.warmup_runs):
            candidate = _linear_boundary_reference(
                tensors["h_norm"],
                runner.layer_weights[layer],
                tensors["recurrent_state_in"],
                tensors["conv_state_in"],
            )
        samples_ms: list[float] = []
        for _ in range(max(args.timed_runs, 1)):
            _sync_if_cuda(args.device)
            start = time.perf_counter()
            candidate = _linear_boundary_reference(
                tensors["h_norm"],
                runner.layer_weights[layer],
                tensors["recurrent_state_in"],
                tensors["conv_state_in"],
            )
            _sync_if_cuda(args.device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        assert candidate is not None
        candidate_tensors = {key: candidate[key].detach().contiguous().cpu() for key in BOUNDARY_KEYS}
        save_file(candidate_tensors, str(out_dir / fixture_file))
        rows.append(
            {
                "fixture_file": fixture_file,
                "layer_id": layer,
                "prompt_id": int(item["prompt_id"]),
                "compute_ms_mean": sum(samples_ms) / len(samples_ms),
                "compute_ms_median": float(median(samples_ms)),
                "compute_ms_samples": samples_ms,
                "keys_written": list(BOUNDARY_KEYS),
                "tensor_shapes": {key: list(candidate_tensors[key].shape) for key in BOUNDARY_KEYS},
                "tensor_dtypes": {key: str(candidate_tensors[key].dtype) for key in BOUNDARY_KEYS},
            }
        )

    medians = [float(row["compute_ms_median"]) for row in rows]
    report = {
        "schema": "lynn-qwen36-linear-boundary-reference-candidate-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "fixtures": str(fixtures_dir),
        "candidate_output_dir": str(out_dir),
        "device": args.device,
        "boundary": "in_proj->conv->recurrent_gdn",
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "elapsed_seconds": time.time() - t0,
        "summary": {
            "total_fixtures": len(rows),
            "total_tensors_written": len(rows) * len(BOUNDARY_KEYS),
            "keys_written": list(BOUNDARY_KEYS),
            "compute_ms_mean": sum(float(row["compute_ms_mean"]) for row in rows) / max(len(rows), 1),
            "compute_ms_median_of_medians": float(median(medians)) if medians else 0.0,
            "compute_ms_min": min(medians, default=0.0),
            "compute_ms_max": max(medians, default=0.0),
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
        raise RuntimeError(f"P169 boundary-candidate check failed with exit code {completed.returncode}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit P173 boundary reference candidates for P169 fixtures.")
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--candidate-output-dir", "--out", dest="candidate_output_dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--skip-p169-check", action="store_true")
    parser.add_argument("--p169-script", default="benchmarks/p169_qwen36_linear_core_fixture_contract.py")
    parser.add_argument("--p169-out", default="")
    parser.add_argument("--max-abs-threshold", type=float, default=0.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.999999)
    args = parser.parse_args()

    if not args.p169_out:
        args.p169_out = str(Path(args.candidate_output_dir) / "p169_boundary_candidate_check.json")

    report = emit_boundary_candidates(args)
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

