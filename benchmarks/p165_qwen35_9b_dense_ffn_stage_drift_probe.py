#!/usr/bin/env python3
"""P165 · Qwen3.5-9B dense FFN packed stage-drift probe.

P164 showed that existing packed NVFP4 wrappers are not a resident candidate:
`scalar_bridge` is close but slow, while `native_fast_2d` is fast but drifts.
This probe localizes the drift by comparing each FFN stage against P159 fixture
tensors:

    gate_output, up_output, intermediate, ffn_output

No serving integration is performed here.
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
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import PackedNVFP4Linear, load_packed_nvfp4_linear  # noqa: E402


@dataclass
class PackedLayer:
    gate: PackedNVFP4Linear
    up: PackedNVFP4Linear
    down: PackedNVFP4Linear


@dataclass
class StageRow:
    fixture_file: str
    layer_id: int
    prompt_id: int
    backend: str
    gate_ms: float
    up_ms: float
    act_ms: float
    down_ms: float
    total_ms: float
    gate_max_abs: float
    gate_mean_abs: float
    gate_rel_l2: float
    gate_cosine: float
    gate_exact: int
    up_max_abs: float
    up_mean_abs: float
    up_rel_l2: float
    up_cosine: float
    up_exact: int
    inter_max_abs: float
    inter_mean_abs: float
    inter_rel_l2: float
    inter_cosine: float
    inter_exact: int
    out_max_abs: float
    out_mean_abs: float
    out_rel_l2: float
    out_cosine: float
    out_exact: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    max_abs = float(diff.abs().max())
    mean_abs = float(diff.abs().mean())
    if max_abs == 0.0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0, "exact": 1}
    rn = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cn = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": float(torch.linalg.vector_norm(diff) / rn),
        "cosine": float(torch.dot(rf, cf) / (rn * cn)),
        "exact": 0,
    }


def _bench_ms(fn: Callable[[], torch.Tensor], warmup: int, repeat: int) -> tuple[torch.Tensor, float]:
    out = fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end)) / float(repeat)


def _base_key(layer_id: int, proj: str) -> str:
    return f"model.language_model.layers.{layer_id}.mlp.{proj}_proj"


def _load_layer(model: str, layer_id: int, device: str, *, prepare_native: bool, load_backend: str) -> PackedLayer:
    layer = PackedLayer(
        gate=load_packed_nvfp4_linear(model, _base_key(layer_id, "gate"), device=device, default_backend=load_backend),
        up=load_packed_nvfp4_linear(model, _base_key(layer_id, "up"), device=device, default_backend=load_backend),
        down=load_packed_nvfp4_linear(model, _base_key(layer_id, "down"), device=device, default_backend=load_backend),
    )
    if prepare_native:
        for linear in (layer.gate, layer.up, layer.down):
            linear._native_scale_b()
            linear._native_weight_t()
    return layer


def _linear(linear: PackedNVFP4Linear, x: torch.Tensor, backend: str) -> torch.Tensor:
    if backend == "scalar_bridge":
        return linear.forward(x, output_dtype=torch.float32, backend="scalar_bridge").reshape(1, -1)
    if backend == "native_fast_2d":
        return linear.forward_native_fast_2d(x.reshape(1, -1))
    raise ValueError(f"unsupported backend: {backend}")


def _run_fixture(
    fixture_path: Path,
    item: dict[str, Any],
    layer: PackedLayer,
    *,
    backend: str,
    device: str,
    warmup: int,
    repeat: int,
) -> StageRow:
    fixture = load_file(str(fixture_path), device=device)
    x = fixture["ffn_in"].to(torch.bfloat16).reshape(1, -1)

    gate, gate_ms = _bench_ms(lambda: _linear(layer.gate, x, backend), warmup, repeat)
    up, up_ms = _bench_ms(lambda: _linear(layer.up, x, backend), warmup, repeat)
    inter, act_ms = _bench_ms(lambda: (F.silu(gate) * up).to(torch.bfloat16), warmup, repeat)
    out, down_ms = _bench_ms(lambda: _linear(layer.down, inter, backend).to(torch.bfloat16), warmup, repeat)

    def full() -> torch.Tensor:
        g = _linear(layer.gate, x, backend)
        u = _linear(layer.up, x, backend)
        return _linear(layer.down, (F.silu(g) * u).to(torch.bfloat16), backend).to(torch.bfloat16)

    out, total_ms = _bench_ms(full, warmup, repeat)
    gate_m = _metric(fixture["gate_output"].to(torch.bfloat16), gate.to(torch.bfloat16))
    up_m = _metric(fixture["up_output"].to(torch.bfloat16), up.to(torch.bfloat16))
    inter_m = _metric(fixture["intermediate"].to(torch.bfloat16), inter.to(torch.bfloat16))
    out_m = _metric(fixture["ffn_output"].to(torch.bfloat16), out.to(torch.bfloat16))

    return StageRow(
        fixture_file=item["file"],
        layer_id=int(item["layer_id"]),
        prompt_id=int(item["prompt_id"]),
        backend=backend,
        gate_ms=gate_ms,
        up_ms=up_ms,
        act_ms=act_ms,
        down_ms=down_ms,
        total_ms=total_ms,
        gate_max_abs=float(gate_m["max_abs"]),
        gate_mean_abs=float(gate_m["mean_abs"]),
        gate_rel_l2=float(gate_m["rel_l2"]),
        gate_cosine=float(gate_m["cosine"]),
        gate_exact=int(gate_m["exact"]),
        up_max_abs=float(up_m["max_abs"]),
        up_mean_abs=float(up_m["mean_abs"]),
        up_rel_l2=float(up_m["rel_l2"]),
        up_cosine=float(up_m["cosine"]),
        up_exact=int(up_m["exact"]),
        inter_max_abs=float(inter_m["max_abs"]),
        inter_mean_abs=float(inter_m["mean_abs"]),
        inter_rel_l2=float(inter_m["rel_l2"]),
        inter_cosine=float(inter_m["cosine"]),
        inter_exact=int(inter_m["exact"]),
        out_max_abs=float(out_m["max_abs"]),
        out_mean_abs=float(out_m["mean_abs"]),
        out_rel_l2=float(out_m["rel_l2"]),
        out_cosine=float(out_m["cosine"]),
        out_exact=int(out_m["exact"]),
    )


def _summary(rows: list[StageRow]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(getattr(r, key)) for r in rows]

    def exacts(key: str) -> int:
        return sum(int(getattr(r, key)) for r in rows)

    out: dict[str, Any] = {"total": len(rows)}
    for prefix in ("gate", "up", "inter", "out"):
        out[f"{prefix}_exact"] = exacts(f"{prefix}_exact")
        out[f"{prefix}_max_abs_max"] = max(vals(f"{prefix}_max_abs"), default=None)
        out[f"{prefix}_mean_abs_mean"] = statistics.mean(vals(f"{prefix}_mean_abs")) if rows else None
        out[f"{prefix}_rel_l2_max"] = max(vals(f"{prefix}_rel_l2"), default=None)
        out[f"{prefix}_cosine_min"] = min(vals(f"{prefix}_cosine"), default=None)
    for key in ("gate_ms", "up_ms", "act_ms", "down_ms", "total_ms"):
        out[f"{key}_mean"] = statistics.mean(vals(key)) if rows else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe packed dense FFN stage drift over P159 fixtures.")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["scalar_bridge", "native_fast_2d", "both"], default="")
    ap.add_argument("--backends", default="", help="Comma-separated backends; wrapper-compatible alias for --backend.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16"], default="bf16")
    ap.add_argument("--load-backend", default="native_scaled_mm")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--prepare-native", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fixtures = Path(args.fixtures)
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    if args.backends:
        backends = [part.strip() for part in args.backends.split(",") if part.strip()]
    else:
        backend = args.backend or "both"
        backends = ["scalar_bridge", "native_fast_2d"] if backend == "both" else [backend]
    allowed = {"scalar_bridge", "native_fast_2d"}
    bad = [backend for backend in backends if backend not in allowed]
    if bad:
        raise ValueError(f"unsupported backends: {bad}")
    layer_cache: dict[int, PackedLayer] = {}
    rows: list[StageRow] = []
    started = time.time()

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        if layer_id not in layer_cache:
            layer_cache[layer_id] = _load_layer(
                args.model,
                layer_id,
                args.device,
                prepare_native=args.prepare_native,
                load_backend=args.load_backend,
            )
        for backend in backends:
            rows.append(
                _run_fixture(
                    fixtures / item["file"],
                    item,
                    layer_cache[layer_id],
                    backend=backend,
                    device=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                )
            )

    by_backend = {
        backend: _summary([r for r in rows if r.backend == backend])
        for backend in backends
    }
    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-p165-stage-drift-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures),
        "model": args.model,
        "backend": args.backend or None,
        "backends": backends,
        "device": torch.cuda.get_device_name(args.device),
        "dtype": args.dtype,
        "load_backend": args.load_backend,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "prepare_native": bool(args.prepare_native),
        "elapsed_seconds": time.time() - started,
        "summary_by_backend": by_backend,
        "decision": "STAGE_DRIFT_PROBE_ONLY",
        "results": [r.to_dict() for r in rows],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "summary_by_backend": by_backend}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
