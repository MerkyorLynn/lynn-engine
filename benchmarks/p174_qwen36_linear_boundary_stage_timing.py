#!/usr/bin/env python3
"""P174: stage timing for the P173 linear boundary.

Measures the exact P173 boundary on P169 fixtures, split into:

  - fused native FP4 in-proj
  - tensor split/view preparation
  - conv update
  - q/k/v/z reshape + beta/g prep
  - recurrent/GDN

This is a design probe for the next fused candidate. It does not modify serving
defaults or resident paths.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

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
from triton_kernels.gated_delta import recurrent_gated_delta_fused_prepare_gqa  # noqa: E402


STAGE_NAMES = [
    "in_proj",
    "split",
    "conv",
    "reshape_gate",
    "recurrent_gdn",
    "total",
]


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


def _time_one_boundary(tensors: dict[str, torch.Tensor], w: dict[str, Any], device: str) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    h_norm = tensors["h_norm"]
    recurrent_state_in = tensors["recurrent_state_in"]
    conv_state_in = tensors["conv_state_in"]
    fused_key = "linear_attn._in_proj_qkv_z_b_a.weight"
    if fused_key not in w:
        raise RuntimeError(f"{fused_key} missing; set LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1")
    batch = h_norm.shape[0]
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
    mixed_for_conv = mixed_new.transpose(1, 2)
    _sync(device)
    times["split"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    out_conv, conv_state_out = _linear_conv_update_decode(
        mixed_for_conv,
        conv_state_in.clone(),
        w["linear_attn.conv1d.weight"],
    )
    _sync(device)
    times["conv"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
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
    _sync(device)
    times["reshape_gate"] = (time.perf_counter() - start) * 1000.0

    if V_PER_K <= 1:
        raise RuntimeError("P174 currently expects Qwen3.6 GQA linear attention")
    start = time.perf_counter()
    core_attn_out, recurrent_state_out = recurrent_gated_delta_fused_prepare_gqa(
        q,
        k,
        v,
        g,
        beta,
        recurrent_state_in.clone(),
    )
    _sync(device)
    times["recurrent_gdn"] = (time.perf_counter() - start) * 1000.0

    times["total"] = (time.perf_counter() - total_start) * 1000.0
    return times, {
        "core_attn_out": core_attn_out.detach(),
        "conv_state_out": conv_state_out.detach(),
        "recurrent_state_out": recurrent_state_out.detach(),
        "z": z.detach(),
    }


def _metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().reshape(-1)
    gf = got.float().reshape(-1)
    diff = rf - gf
    max_abs = float(diff.abs().max().item())
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(gf).clamp_min(1e-12)
    return {
        "max_abs": max_abs,
        "cosine": float((torch.dot(rf, gf) / (ref_norm * got_norm)).item()),
        "exact": 1 if max_abs == 0.0 else 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
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
        for _ in range(args.warmup_runs):
            _time_one_boundary(tensors, runner.layer_weights[layer], args.device)
        samples: list[dict[str, float]] = []
        last_outputs: dict[str, torch.Tensor] | None = None
        for _ in range(max(args.timed_runs, 1)):
            times, last_outputs = _time_one_boundary(tensors, runner.layer_weights[layer], args.device)
            samples.append(times)
        assert last_outputs is not None
        med = {name: float(median(sample[name] for sample in samples)) for name in STAGE_NAMES}
        exact_metrics = {
            key: _metrics(tensors[key], last_outputs[key])
            for key in ("core_attn_out", "conv_state_out", "recurrent_state_out")
        }
        rows.append(
            {
                "fixture_file": fixture_file,
                "layer_id": layer,
                "prompt_id": int(item["prompt_id"]),
                "stage_ms_median": med,
                "stage_ms_samples": samples,
                "exact_metrics": exact_metrics,
                "passed_exact": all(metric["exact"] == 1 for metric in exact_metrics.values()),
            }
        )
    stage_summary: dict[str, dict[str, float]] = {}
    for name in STAGE_NAMES:
        vals = [float(row["stage_ms_median"][name]) for row in rows]
        stage_summary[name] = {
            "median": float(median(vals)) if vals else 0.0,
            "mean": sum(vals) / max(len(vals), 1),
            "min": min(vals, default=0.0),
            "max": max(vals, default=0.0),
        }
    return {
        "schema": "lynn-qwen36-linear-boundary-stage-timing-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(model_path),
        "fixtures": str(fixtures_dir),
        "device": args.device,
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "summary": {
            "total_fixtures": len(rows),
            "passed_exact": sum(1 for row in rows if row["passed_exact"]),
            "all_exact": all(row["passed_exact"] for row in rows),
            "stage_ms": stage_summary,
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile P173 linear boundary stages on P169 fixtures.")
    parser.add_argument("--model", default="")
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=5)
    args = parser.parse_args()
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

