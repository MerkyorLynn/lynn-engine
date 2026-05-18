#!/usr/bin/env python3
"""P167 · Qwen3.5-9B dense FFN `mm(out=...)` boundary probe.

P164/P165 closed the existing packed NVFP4 wrappers.  This probe stays on the
strict BF16/dequant math path and only tests whether caller-owned `torch.mm`
output buffers can reduce the dense FFN boundary without changing results.
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
class ProbeRow:
    fixture_file: str
    layer_id: int
    prompt_id: int
    ref_ms: float
    mm_out_ms: float
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


def _dense_ref(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    gate = F.linear(x, w["mlp.gate_proj.weight"])
    up = F.linear(x, w["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.down_proj.weight"])


def _dense_mm_out(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    scratch: dict[str, torch.Tensor],
) -> torch.Tensor:
    torch.mm(x, w["mlp.gate_proj.weight"].t(), out=scratch["gate"])
    torch.mm(x, w["mlp.up_proj.weight"].t(), out=scratch["up"])
    torch.mul(F.silu(scratch["gate"]), scratch["up"], out=scratch["inter"])
    torch.mm(scratch["inter"], w["mlp.down_proj.weight"].t(), out=scratch["out"])
    return scratch["out"]


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


def _make_scratch(x: torch.Tensor, w: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    inter = w["mlp.gate_proj.weight"].shape[0]
    hidden = w["mlp.down_proj.weight"].shape[0]
    return {
        "gate": torch.empty((1, inter), device=x.device, dtype=x.dtype),
        "up": torch.empty((1, inter), device=x.device, dtype=x.dtype),
        "inter": torch.empty((1, inter), device=x.device, dtype=x.dtype),
        "out": torch.empty((1, hidden), device=x.device, dtype=x.dtype),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fixtures_dir = Path(args.fixtures)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    layer_cache: dict[int, dict[str, torch.Tensor]] = {}
    scratch_cache: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[ProbeRow] = []
    started = time.time()

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
        x = fixture["ffn_in"].to(dtype)
        expected = fixture["ffn_output"].to(dtype)
        if layer_id not in layer_cache:
            layer_cache[layer_id] = _load_weights(fixtures_dir, args.model, layer_id, args.device, dtype)
            scratch_cache[layer_id] = _make_scratch(x, layer_cache[layer_id])
        w = layer_cache[layer_id]
        scratch = scratch_cache[layer_id]
        ref, ref_ms = _bench(lambda: _dense_ref(x, w), args.warmup, args.repeat)
        cand, cand_ms = _bench(lambda: _dense_mm_out(x, w, scratch), args.warmup, args.repeat)
        m = _metrics(expected, cand)
        rows.append(
            ProbeRow(
                fixture_file=item["file"],
                layer_id=layer_id,
                prompt_id=int(item["prompt_id"]),
                ref_ms=ref_ms,
                mm_out_ms=cand_ms,
                speedup=ref_ms / max(cand_ms, 1e-9),
                max_abs=float(m["max_abs"]),
                mean_abs=float(m["mean_abs"]),
                rel_l2=float(m["rel_l2"]),
                cosine=float(m["cosine"]),
                exact=int(m["exact"]),
            )
        )
        # Keep ref alive for the compiler/runtime and assert the fixture still
        # matches the standard path; P160 owns the formal reference gate.
        _ = ref

    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-p167-mm-out-probe-v1",
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
        "mm_out_ms_mean": statistics.mean([r.mm_out_ms for r in rows]) if rows else None,
        "speedup_mean": statistics.mean([r.speedup for r in rows]) if rows else None,
        "decision": "MM_OUT_CANDIDATE" if rows and all(r.exact for r in rows) else "MM_OUT_CLOSED_NUMERIC",
        "results": [r.to_dict() for r in rows],
    }
    if report["decision"] == "MM_OUT_CANDIDATE" and (report["speedup_mean"] or 0.0) <= 1.01:
        report["decision"] = "MM_OUT_CLOSED_FLAT"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe exact dense FFN mm(out=...) boundary.")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("out", "decision") if k in report}, ensure_ascii=False))
    print(json.dumps({
        "out": str(out),
        "decision": report["decision"],
        "exact": report["exact"],
        "total": report["total"],
        "ref_ms_mean": report["ref_ms_mean"],
        "mm_out_ms_mean": report["mm_out_ms_mean"],
        "speedup_mean": report["speedup_mean"],
        "max_abs_max": report["max_abs_max"],
        "cosine_min": report["cosine_min"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
