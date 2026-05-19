#!/usr/bin/env python3
"""P185 · Qwen3.5-9B dense W4A8 fixture gate.

Spark quality tests showed 9B W4A8 fake-quant is essentially lossless versus
W4A16.  This R6000-side gate checks the same contract on Lynn-native dense FFN
fixtures before any resident or native FP8 kernel work is promoted.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.full_forward import _fake_quant_fp8_activation  # noqa: E402
from engine.loader import load_qwen36_layer  # noqa: E402


@dataclass
class FixtureModeRow:
    fixture_file: str
    layer_id: int
    prompt_id: int
    mode: str
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int
    ref_ms_mean: float
    candidate_ms_mean: float
    speedup_vs_ref: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float | int]:
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


def _load_weights_from_fixture_dir(fixtures_dir: Path, layer_id: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor] | None:
    weights_path = fixtures_dir / f"layer_{layer_id:02d}_dense_ffn_weights.safetensors"
    if not weights_path.exists():
        return None
    data = load_file(str(weights_path), device=device)
    return {
        "mlp.gate_proj.weight": data["mlp.gate_proj.weight"].to(dtype),
        "mlp.up_proj.weight": data["mlp.up_proj.weight"].to(dtype),
        "mlp.down_proj.weight": data["mlp.down_proj.weight"].to(dtype),
    }


def _dense_forward(ffn_in: torch.Tensor, weights: dict[str, Any], mode: str) -> torch.Tensor:
    active = _fake_quant_fp8_activation(ffn_in) if mode in {"gateup", "full"} else ffn_in
    gate = F.linear(active, weights["mlp.gate_proj.weight"])
    up = F.linear(active, weights["mlp.up_proj.weight"])
    inter = F.silu(gate) * up
    if mode == "full":
        inter = _fake_quant_fp8_activation(inter)
    return F.linear(inter, weights["mlp.down_proj.weight"])


def _bench_ms(fn, warmup: int, repeat: int) -> tuple[torch.Tensor, float]:
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


def _summarize(rows: list[FixtureModeRow], mode: str) -> dict[str, Any]:
    selected = [r for r in rows if r.mode == mode]
    return {
        "mode": mode,
        "total": len(selected),
        "exact": sum(r.exact for r in selected),
        "max_abs_max": max((r.max_abs for r in selected), default=None),
        "mean_abs_mean": statistics.mean([r.mean_abs for r in selected]) if selected else None,
        "rel_l2_max": max((r.rel_l2 for r in selected), default=None),
        "rel_l2_mean": statistics.mean([r.rel_l2 for r in selected]) if selected else None,
        "cosine_min": min((r.cosine for r in selected), default=None),
        "ref_ms_mean": statistics.mean([r.ref_ms_mean for r in selected]) if selected else None,
        "candidate_ms_mean": statistics.mean([r.candidate_ms_mean for r in selected]) if selected else None,
        "speedup_vs_ref_mean": statistics.mean([r.speedup_vs_ref for r in selected]) if selected else None,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fixtures_dir = Path(args.fixtures)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    old_fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT")
    old_gran = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY")
    os.environ["LYNN_W4A8_FAKE_QUANT_FORMAT"] = args.fp8_format
    os.environ["LYNN_W4A8_FAKE_QUANT_GRANULARITY"] = args.granularity

    try:
        layer_cache: dict[int, dict[str, Any]] = {}
        ref_cache: dict[str, tuple[torch.Tensor, float]] = {}
        rows: list[FixtureModeRow] = []
        for item in manifest["fixtures"]:
            layer_id = int(item["layer_id"])
            if layer_id not in layer_cache:
                weights = _load_weights_from_fixture_dir(fixtures_dir, layer_id, args.device, dtype)
                if weights is None:
                    if not args.model:
                        raise ValueError("fixtures do not include weights; pass --model")
                    weights, inferred = load_qwen36_layer(
                        args.model,
                        layer_id,
                        num_experts=0,
                        device=args.device,
                        dequant_dtype=dtype,
                    )
                    if inferred.get("is_moe", False):
                        raise RuntimeError(f"layer {layer_id} loaded as MoE; expected dense")
                layer_cache[layer_id] = weights

            fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
            ffn_in = fixture["ffn_in"].to(dtype)
            expected = fixture["ffn_output"].to(dtype)
            weights = layer_cache[layer_id]

            ref_out, ref_ms = _bench_ms(lambda: _dense_forward(ffn_in, weights, "off"), args.warmup, args.repeat)
            ref_m = _metrics(expected, ref_out)
            if ref_m["max_abs"] != 0.0:
                raise RuntimeError(f"fixture reference is not exact for {item['file']}: {ref_m}")
            ref_cache[item["file"]] = (ref_out, ref_ms)

            for mode in ("gateup", "full"):
                cand, cand_ms = _bench_ms(lambda mode=mode: _dense_forward(ffn_in, weights, mode), args.warmup, args.repeat)
                m = _metrics(expected, cand.to(dtype))
                rows.append(
                    FixtureModeRow(
                        fixture_file=item["file"],
                        layer_id=layer_id,
                        prompt_id=int(item["prompt_id"]),
                        mode=mode,
                        max_abs=float(m["max_abs"]),
                        mean_abs=float(m["mean_abs"]),
                        rel_l2=float(m["rel_l2"]),
                        cosine=float(m["cosine"]),
                        exact=int(m["exact"]),
                        ref_ms_mean=ref_ms,
                        candidate_ms_mean=cand_ms,
                        speedup_vs_ref=ref_ms / cand_ms if cand_ms > 0 else 0.0,
                    )
                )

        summaries = [_summarize(rows, "gateup"), _summarize(rows, "full")]
        full_summary = next(s for s in summaries if s["mode"] == "full")
        gateup_summary = next(s for s in summaries if s["mode"] == "gateup")
        if full_summary["cosine_min"] is not None and full_summary["cosine_min"] >= args.cosine_green and full_summary["rel_l2_max"] <= args.rel_l2_green:
            decision = "DENSE_W4A8_FIXTURE_GREEN"
        elif gateup_summary["cosine_min"] is not None and gateup_summary["cosine_min"] >= args.cosine_green and gateup_summary["rel_l2_max"] <= args.rel_l2_green:
            decision = "DENSE_W4A8_GATEUP_GREEN_FULL_AMBER"
        elif full_summary["cosine_min"] is not None and full_summary["cosine_min"] >= args.cosine_amber:
            decision = "DENSE_W4A8_FIXTURE_AMBER"
        else:
            decision = "DENSE_W4A8_FIXTURE_RED"

        return {
            "schema": "lynn-qwen35-9b-dense-w4a8-fixture-gate-v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fixtures": str(fixtures_dir),
            "model": args.model,
            "dtype": args.dtype,
            "fp8_format": args.fp8_format,
            "granularity": args.granularity,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "thresholds": {
                "cosine_green": args.cosine_green,
                "cosine_amber": args.cosine_amber,
                "rel_l2_green": args.rel_l2_green,
            },
            "decision": decision,
            "summaries": summaries,
            "results": [r.to_dict() for r in rows],
        }
    finally:
        if old_fmt is None:
            os.environ.pop("LYNN_W4A8_FAKE_QUANT_FORMAT", None)
        else:
            os.environ["LYNN_W4A8_FAKE_QUANT_FORMAT"] = old_fmt
        if old_gran is None:
            os.environ.pop("LYNN_W4A8_FAKE_QUANT_GRANULARITY", None)
        else:
            os.environ["LYNN_W4A8_FAKE_QUANT_GRANULARITY"] = old_gran


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.5-9B dense W4A8 fixture gate.")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--fp8-format", choices=["e4m3", "e5m2"], default="e4m3")
    ap.add_argument("--granularity", choices=["tensor", "row", "per16"], default="per16")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--cosine-green", type=float, default=0.999)
    ap.add_argument("--cosine-amber", type=float, default=0.995)
    ap.add_argument("--rel-l2-green", type=float, default=0.05)
    args = ap.parse_args()

    report = run_gate(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": report["decision"],
        "summaries": report["summaries"],
        "out": str(out_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
