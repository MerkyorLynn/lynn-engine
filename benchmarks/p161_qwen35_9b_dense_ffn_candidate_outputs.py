#!/usr/bin/env python3
"""P161 · Emit Qwen3.5-9B dense-FFN candidate outputs from P159 fixtures.

This is the candidate-output side of the P159/P160 fixture contract.  It reads
P159 dense-FFN fixtures, computes one output per fixture, and writes a directory
that can be passed directly to P160 via ``--candidate-output-dir``.

The initial backend is a reference PyTorch dense FFN:

    down_proj(silu(gate_proj(ffn_in)) * up_proj(ffn_in))

The CLI intentionally exposes backend hooks so future torch.compile or native
implementations can keep the same output-dir contract.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402


@dataclass
class CandidateRow:
    fixture_file: str
    output_file: str
    layer_id: int
    prompt_id: int
    backend: str
    ms_mean: float
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


def _load_weights_from_fixture_dir(
    fixtures_dir: Path,
    layer_id: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    weights_path = fixtures_dir / f"layer_{layer_id:02d}_dense_ffn_weights.safetensors"
    if not weights_path.exists():
        return None
    data = load_file(str(weights_path), device=device)
    return {
        "mlp.gate_proj.weight": data["mlp.gate_proj.weight"].to(dtype),
        "mlp.up_proj.weight": data["mlp.up_proj.weight"].to(dtype),
        "mlp.down_proj.weight": data["mlp.down_proj.weight"].to(dtype),
    }


def _load_layer_weights(
    *,
    fixtures_dir: Path,
    model: str,
    layer_id: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    from_fixtures = _load_weights_from_fixture_dir(fixtures_dir, layer_id, device, dtype)
    if from_fixtures is not None:
        return from_fixtures
    if not model:
        raise ValueError(
            "P159 fixtures do not contain dense FFN weights; pass --model or export fixtures with --export-weights"
        )
    weights, inferred = load_qwen36_layer(
        model,
        layer_id,
        num_experts=0,
        device=device,
        dequant_dtype=dtype,
    )
    if inferred.get("is_moe", False):
        raise RuntimeError(f"layer {layer_id} loaded as MoE; expected dense")
    return weights


def _dense_forward(ffn_in: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
    gate = F.linear(ffn_in, weights["mlp.gate_proj.weight"])
    up = F.linear(ffn_in, weights["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, weights["mlp.down_proj.weight"])


def _build_backend(
    backend: str,
    weights: dict[str, Any],
    *,
    compile_mode: str,
    fullgraph: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if backend == "reference":
        return lambda ffn_in: _dense_forward(ffn_in, weights)
    if backend == "torch_compile":
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable in this PyTorch build")

        def fn(ffn_in: torch.Tensor) -> torch.Tensor:
            return _dense_forward(ffn_in, weights)

        return torch.compile(fn, mode=compile_mode, fullgraph=fullgraph)  # type: ignore[attr-defined]
    if backend == "native":
        raise NotImplementedError(
            "native backend hook is reserved; implement it behind this contract and keep P160 compatibility"
        )
    raise ValueError(f"unknown backend: {backend}")


def _time_candidate(
    fn: Callable[[torch.Tensor], torch.Tensor],
    ffn_in: torch.Tensor,
    *,
    warmup: int,
    repeat: int,
) -> tuple[torch.Tensor, float]:
    out = fn(ffn_in)
    if ffn_in.is_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn(ffn_in)
    if ffn_in.is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = fn(ffn_in)
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end)) / float(repeat)
    t0 = time.time()
    for _ in range(repeat):
        out = fn(ffn_in)
    return out, (time.time() - t0) * 1000.0 / float(repeat)


def emit_candidate_outputs(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype(args.dtype)
    fixtures_dir = Path(args.fixtures)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = fixtures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    rows: list[CandidateRow] = []
    layer_cache: dict[int, dict[str, Any]] = {}
    backend_cache: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    t0 = time.time()

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        if layer_id not in layer_cache:
            weights = _load_layer_weights(
                fixtures_dir=fixtures_dir,
                model=args.model,
                layer_id=layer_id,
                device=args.device,
                dtype=dtype,
            )
            layer_cache[layer_id] = weights
            backend_cache[layer_id] = _build_backend(
                args.backend,
                weights,
                compile_mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
            )

        fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
        ffn_in = fixture["ffn_in"].to(dtype)
        candidate, ms_mean = _time_candidate(
            backend_cache[layer_id],
            ffn_in,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        if args.output_dtype == "fixture":
            out_tensor = candidate.to(fixture["ffn_output"].dtype)
        else:
            out_tensor = candidate.to(_dtype(args.output_dtype))

        output_file = Path(item["file"]).name
        save_file({"candidate_output": out_tensor.detach().contiguous().cpu()}, str(out_dir / output_file))
        rows.append(
            CandidateRow(
                fixture_file=item["file"],
                output_file=output_file,
                layer_id=layer_id,
                prompt_id=int(item["prompt_id"]),
                backend=args.backend,
                ms_mean=ms_mean,
                output_shape=list(out_tensor.shape),
                output_dtype=str(out_tensor.dtype),
            )
        )

    times = [r.ms_mean for r in rows]
    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-candidate-outputs-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "fixture_manifest": str(manifest_path),
        "model": args.model,
        "out": str(out_dir),
        "backend": args.backend,
        "dtype": args.dtype,
        "output_dtype": args.output_dtype,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "total": len(rows),
        "elapsed_seconds": time.time() - t0,
        "candidate_ms_mean": statistics.mean(times) if times else None,
        "candidate_ms_median": statistics.median(times) if times else None,
        "candidate_ms_min": min(times) if times else None,
        "candidate_ms_max": max(times) if times else None,
        "p160_candidate_output_dir": str(out_dir),
        "results": [r.to_dict() for r in rows],
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit P160-compatible dense FFN candidate outputs.")
    ap.add_argument("--fixtures", required=True, help="P159 fixture directory containing manifest.json.")
    ap.add_argument("--out", required=True, help="Candidate output directory for P160 --candidate-output-dir.")
    ap.add_argument("--model", default="", help="Model artifact used when fixtures do not include exported weights.")
    ap.add_argument("--backend", choices=["reference", "torch_compile", "native"], default="reference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--output-dtype", choices=["fixture", "bf16", "fp16", "fp32"], default="fixture")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--compile-mode", default="default")
    ap.add_argument("--compile-fullgraph", action="store_true")
    ap.add_argument("--report", default="", help="Optional copy of the candidate manifest JSON.")
    args = ap.parse_args()

    report = emit_candidate_outputs(args)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": report["out"],
        "backend": report["backend"],
        "total": report["total"],
        "candidate_ms_mean": report["candidate_ms_mean"],
        "p160_candidate_output_dir": report["p160_candidate_output_dir"],
        "report": args.report or str(Path(args.out) / "manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
