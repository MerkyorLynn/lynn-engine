#!/usr/bin/env python3
"""P168 · Qwen3.5-9B dense FFN fused gate/up probe.

This is the first exact-first dense FFN boundary with a plausible speedup:
replace two independent gate/up matmuls with one matmul over a pre-concatenated
`[gate; up]` BF16 weight, then chunk the result.  The down projection and
activation order stay unchanged.
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

from engine.loader import load_qwen36_layer  # noqa: E402


@dataclass
class Row:
    fixture_file: str
    layer_id: int
    prompt_id: int
    ref_ms: float
    fused_ms: float
    speedup: float
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
    rf = ref.float().flatten()
    cf = cand.float().flatten()
    diff = rf - cf
    max_abs = float(diff.abs().max())
    if max_abs == 0.0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0, "cosine": 1.0, "exact": 1}
    rn = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    cn = torch.linalg.vector_norm(cf).clamp_min(1e-12)
    return {
        "max_abs": max_abs,
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / rn),
        "cosine": float(torch.dot(rf, cf) / (rn * cn)),
        "exact": 0,
    }


def _bench(fn: Callable[[], torch.Tensor], warmup: int, repeat: int) -> tuple[torch.Tensor, float]:
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


def _load_weights(fixtures_dir: Path, model: str, layer_id: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    weights_path = fixtures_dir / f"layer_{layer_id:02d}_dense_ffn_weights.safetensors"
    if weights_path.exists():
        data = load_file(str(weights_path), device=device)
        return {
            "mlp.gate_proj.weight": data["mlp.gate_proj.weight"].to(dtype),
            "mlp.up_proj.weight": data["mlp.up_proj.weight"].to(dtype),
            "mlp.down_proj.weight": data["mlp.down_proj.weight"].to(dtype),
        }
    w, inferred = load_qwen36_layer(model, layer_id, num_experts=0, device=device, dequant_dtype=dtype)
    if inferred.get("is_moe", False):
        raise RuntimeError(f"layer {layer_id} loaded as MoE; expected dense")
    return w


def _dense_ref(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = F.linear(x, w["mlp.gate_proj.weight"])
    up = F.linear(x, w["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.down_proj.weight"])


def _dense_fused_gateup(x: torch.Tensor, fused_gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    gate_up = F.linear(x, fused_gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fixtures_dir = Path(args.fixtures)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    layer_cache: dict[int, dict[str, torch.Tensor]] = {}
    fused_cache: dict[int, torch.Tensor] = {}
    rows: list[Row] = []
    started = time.time()

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
        x = fixture["ffn_in"].to(dtype)
        expected = fixture["ffn_output"].to(dtype)
        if layer_id not in layer_cache:
            w = _load_weights(fixtures_dir, args.model, layer_id, args.device, dtype)
            layer_cache[layer_id] = w
            fused_cache[layer_id] = torch.cat(
                [w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]],
                dim=0,
            ).contiguous()
        w = layer_cache[layer_id]
        fused = fused_cache[layer_id]
        ref, ref_ms = _bench(lambda: _dense_ref(x, w), args.warmup, args.repeat)
        cand, fused_ms = _bench(
            lambda: _dense_fused_gateup(x, fused, w["mlp.down_proj.weight"]),
            args.warmup,
            args.repeat,
        )
        m = _metrics(expected, cand)
        rows.append(
            Row(
                fixture_file=item["file"],
                layer_id=layer_id,
                prompt_id=int(item["prompt_id"]),
                ref_ms=ref_ms,
                fused_ms=fused_ms,
                speedup=ref_ms / max(fused_ms, 1e-9),
                max_abs=float(m["max_abs"]),
                mean_abs=float(m["mean_abs"]),
                rel_l2=float(m["rel_l2"]),
                cosine=float(m["cosine"]),
                exact=int(m["exact"]),
            )
        )
        _ = ref

    speedups = [r.speedup for r in rows]
    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-p168-fused-gateup-probe-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "model": args.model,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(args.device),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "elapsed_seconds": time.time() - started,
        "total": len(rows),
        "exact": sum(r.exact for r in rows),
        "max_abs_max": max((r.max_abs for r in rows), default=None),
        "cosine_min": min((r.cosine for r in rows), default=None),
        "ref_ms_mean": statistics.mean([r.ref_ms for r in rows]) if rows else None,
        "fused_ms_mean": statistics.mean([r.fused_ms for r in rows]) if rows else None,
        "speedup_mean": statistics.mean(speedups) if rows else None,
        "speedup_min": min(speedups, default=None),
        "decision": "FUSED_GATEUP_CANDIDATE" if rows and all(r.exact for r in rows) else "FUSED_GATEUP_CLOSED_NUMERIC",
        "results": [r.to_dict() for r in rows],
    }
    if report["decision"] == "FUSED_GATEUP_CANDIDATE" and (report["speedup_mean"] or 0.0) <= 1.01:
        report["decision"] = "FUSED_GATEUP_CLOSED_FLAT"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe dense FFN fused gate/up boundary.")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "decision": report["decision"],
        "exact": report["exact"],
        "total": report["total"],
        "ref_ms_mean": report["ref_ms_mean"],
        "fused_ms_mean": report["fused_ms_mean"],
        "speedup_mean": report["speedup_mean"],
        "speedup_min": report["speedup_min"],
        "max_abs_max": report["max_abs_max"],
        "cosine_min": report["cosine_min"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
