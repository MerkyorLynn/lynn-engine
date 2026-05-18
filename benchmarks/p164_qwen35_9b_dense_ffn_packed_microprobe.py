#!/usr/bin/env python3
"""P164 · Qwen3.5-9B dense-FFN packed/native microprobe.

This research probe consumes P159 dense-FFN fixtures and emits a P160/P161-style
candidate output directory.  It reloads the packed NVFP4 gate/up/down linears
from the model artifact, runs one fixture token through the packed wrappers, and
records timing plus fixture-output drift metrics.

No serving integration is performed here.  Numeric drift is expected to be
reported, not hidden.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import (  # noqa: E402
    PackedNVFP4Linear,
    dual_scalar_bridge,
    load_packed_nvfp4_linear,
)


@dataclass
class PackedLayer:
    gate: PackedNVFP4Linear
    up: PackedNVFP4Linear
    down: PackedNVFP4Linear


@dataclass
class ProbeRow:
    fixture_file: str
    output_file: str
    layer_id: int
    prompt_id: int
    gate_up_backend: str
    down_backend: str
    gate_up_ms_mean: float
    activation_ms_mean: float
    down_ms_mean: float
    total_ms_mean: float
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int
    output_shape: list[int]
    output_dtype: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    max_abs = float(diff.abs().max())
    if max_abs == 0.0:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rel_l2": 0.0,
            "cosine": 1.0,
            "exact": 1,
        }
    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cand_norm = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)
    cosine = torch.dot(rf, cf) / (ref_norm * cand_norm)
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": float(cosine),
        "exact": 1 if max_abs == 0.0 else 0,
    }


def _cuda_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return device


def _time_ms(fn, *, warmup: int, repeat: int) -> tuple[Any, float]:
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end)) / float(repeat)
    t0 = time.time()
    for _ in range(repeat):
        out = fn()
    return out, (time.time() - t0) * 1000.0 / float(repeat)


def _base_key(layer_id: int, proj: str) -> str:
    return f"model.language_model.layers.{layer_id}.mlp.{proj}_proj"


def _load_packed_layer(model: str, layer_id: int, device: str, backend: str) -> PackedLayer:
    return PackedLayer(
        gate=load_packed_nvfp4_linear(
            model,
            _base_key(layer_id, "gate"),
            name=_base_key(layer_id, "gate"),
            device=device,
            default_backend=backend,
        ),
        up=load_packed_nvfp4_linear(
            model,
            _base_key(layer_id, "up"),
            name=_base_key(layer_id, "up"),
            device=device,
            default_backend=backend,
        ),
        down=load_packed_nvfp4_linear(
            model,
            _base_key(layer_id, "down"),
            name=_base_key(layer_id, "down"),
            device=device,
            default_backend=backend,
        ),
    )


def _linear_native(linear: PackedNVFP4Linear, x: torch.Tensor) -> torch.Tensor:
    return linear.forward_native_fast_2d(x.reshape(1, x.shape[-1]))


def _linear_scalar(linear: PackedNVFP4Linear, x: torch.Tensor) -> torch.Tensor:
    return linear.forward(x, output_dtype=torch.float32, backend="scalar_bridge").reshape(1, -1)


def _choose_down_backend(requested: str) -> str:
    if requested in ("auto", "native_fast_2d", "dual_scalar_bridge"):
        return "native_fast_2d"
    if requested == "scalar_bridge":
        return "scalar_bridge"
    raise ValueError(f"unknown backend: {requested}")


def _gate_up(
    ffn_in: torch.Tensor,
    layer: PackedLayer,
    *,
    requested: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    can_dual = layer.gate.weight_packed.shape == layer.up.weight_packed.shape
    if requested in ("auto", "dual_scalar_bridge") and can_dual:
        gate, up = dual_scalar_bridge(ffn_in, layer.gate, layer.up, output_dtype=torch.float32)
        return gate.reshape(1, -1), up.reshape(1, -1), "dual_scalar_bridge"
    if requested == "scalar_bridge":
        return _linear_scalar(layer.gate, ffn_in), _linear_scalar(layer.up, ffn_in), "scalar_bridge"
    return _linear_native(layer.gate, ffn_in), _linear_native(layer.up, ffn_in), "native_fast_2d"


def _down(intermediate: torch.Tensor, layer: PackedLayer, *, backend: str) -> torch.Tensor:
    if backend == "scalar_bridge":
        return _linear_scalar(layer.down, intermediate)
    return _linear_native(layer.down, intermediate)


def _out_dtype(candidate: torch.Tensor, fixture: torch.Tensor, requested: str) -> torch.Tensor:
    if requested == "fixture":
        return candidate.to(fixture.dtype)
    return candidate.to(_dtype(requested))


def _run_one(
    ffn_in: torch.Tensor,
    layer: PackedLayer,
    *,
    backend: str,
    warmup: int,
    repeat: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    gate_up_out, gate_up_ms = _time_ms(
        lambda: _gate_up(ffn_in, layer, requested=backend),
        warmup=warmup,
        repeat=repeat,
    )
    gate, up, gate_up_backend = gate_up_out

    intermediate, activation_ms = _time_ms(lambda: F.silu(gate) * up, warmup=warmup, repeat=repeat)
    down_backend = _choose_down_backend(backend)
    output, down_ms = _time_ms(
        lambda: _down(intermediate, layer, backend=down_backend),
        warmup=warmup,
        repeat=repeat,
    )

    def full() -> torch.Tensor:
        g, u, _ = _gate_up(ffn_in, layer, requested=backend)
        return _down(F.silu(g) * u, layer, backend=down_backend)

    output, total_ms = _time_ms(full, warmup=warmup, repeat=repeat)
    timings = {
        "gate_up_backend": gate_up_backend,
        "down_backend": down_backend,
        "gate_up_ms_mean": gate_up_ms,
        "activation_ms_mean": activation_ms,
        "down_ms_mean": down_ms,
        "total_ms_mean": total_ms,
    }
    return output, timings


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype(args.dtype)
    fixtures_dir = Path(args.fixtures)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = fixtures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    layer_cache: dict[int, PackedLayer] = {}
    rows: list[ProbeRow] = []
    t0 = time.time()

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        if layer_id not in layer_cache:
            layer_cache[layer_id] = _load_packed_layer(
                args.model,
                layer_id,
                args.device,
                "scalar_bridge" if args.backend == "scalar_bridge" else "native_scaled_mm",
            )
            if args.prepare_native:
                for linear in (
                    layer_cache[layer_id].gate,
                    layer_cache[layer_id].up,
                    layer_cache[layer_id].down,
                ):
                    linear._native_scale_b()
                    linear._native_weight_t()

        fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
        ffn_in = fixture["ffn_in"].to(dtype).reshape(1, -1)
        expected = fixture["ffn_output"].to(dtype)
        candidate, stats = _run_one(
            ffn_in,
            layer_cache[layer_id],
            backend=args.backend,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        out_tensor = _out_dtype(candidate, fixture["ffn_output"], args.output_dtype)
        metrics = _metrics(expected, out_tensor)
        output_file = Path(item["file"]).name
        save_file({"candidate_output": out_tensor.detach().contiguous().cpu()}, str(out_dir / output_file))
        rows.append(
            ProbeRow(
                fixture_file=item["file"],
                output_file=output_file,
                layer_id=layer_id,
                prompt_id=int(item["prompt_id"]),
                gate_up_backend=str(stats["gate_up_backend"]),
                down_backend=str(stats["down_backend"]),
                gate_up_ms_mean=float(stats["gate_up_ms_mean"]),
                activation_ms_mean=float(stats["activation_ms_mean"]),
                down_ms_mean=float(stats["down_ms_mean"]),
                total_ms_mean=float(stats["total_ms_mean"]),
                max_abs=float(metrics["max_abs"]),
                mean_abs=float(metrics["mean_abs"]),
                rel_l2=float(metrics["rel_l2"]),
                cosine=float(metrics["cosine"]),
                exact=int(metrics["exact"]),
                output_shape=list(out_tensor.shape),
                output_dtype=str(out_tensor.dtype),
            )
        )

    total_ms = [r.total_ms_mean for r in rows]
    gate_up_ms = [r.gate_up_ms_mean for r in rows]
    down_ms = [r.down_ms_mean for r in rows]
    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-p164-packed-microprobe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "fixture_manifest": str(manifest_path),
        "model": args.model,
        "out": str(out_dir),
        "p160_candidate_output_dir": str(out_dir),
        "backend": args.backend,
        "dtype": args.dtype,
        "output_dtype": args.output_dtype,
        "device": _cuda_name(args.device),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "prepare_native": bool(args.prepare_native),
        "total": len(rows),
        "elapsed_seconds": time.time() - t0,
        "exact": sum(r.exact for r in rows),
        "max_abs_max": max((r.max_abs for r in rows), default=None),
        "cosine_min": min((r.cosine for r in rows), default=None),
        "total_ms_mean": statistics.mean(total_ms) if total_ms else None,
        "total_ms_median": statistics.median(total_ms) if total_ms else None,
        "gate_up_ms_mean": statistics.mean(gate_up_ms) if gate_up_ms else None,
        "down_ms_mean": statistics.mean(down_ms) if down_ms else None,
        "decision": "PACKED_RESEARCH_EXACT" if rows and sum(r.exact for r in rows) == len(rows) else "PACKED_RESEARCH_DRIFT",
        "results": [r.to_dict() for r in rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run P164 packed/native dense-FFN microprobe over P159 fixtures.")
    ap.add_argument("--fixtures", required=True, help="P159 fixture directory containing manifest.json.")
    ap.add_argument("--model", required=True, help="Packed NVFP4 model artifact.")
    ap.add_argument("--out", required=True, help="P160/P161-compatible candidate output directory.")
    ap.add_argument(
        "--backend",
        choices=["auto", "native_fast_2d", "dual_scalar_bridge", "scalar_bridge"],
        default="auto",
        help="auto uses dual_scalar_bridge for gate/up when possible and native_fast_2d for down.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output-dtype", choices=["fixture", "bf16", "fp16", "fp32"], default="fixture")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--prepare-native", action="store_true", help="Precompute native scale/weight views before timing.")
    ap.add_argument("--report", default="", help="Optional copy of the probe manifest JSON.")
    args = ap.parse_args()

    report = run_probe(args)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": report["out"],
                "decision": report["decision"],
                "backend": report["backend"],
                "total": report["total"],
                "exact": report["exact"],
                "max_abs_max": report["max_abs_max"],
                "cosine_min": report["cosine_min"],
                "total_ms_mean": report["total_ms_mean"],
                "p160_candidate_output_dir": report["p160_candidate_output_dir"],
                "report": args.report or str(Path(args.out) / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
