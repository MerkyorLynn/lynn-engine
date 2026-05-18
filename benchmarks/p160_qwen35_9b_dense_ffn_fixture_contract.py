#!/usr/bin/env python3
"""P160 · Qwen3.5-9B dense-FFN fixture contract.

Validates p159 fixtures by recomputing dense FFN output with the selected
layer weights and optionally compares precomputed candidate outputs.  This is
the first admission gate for 9B dense-FFN kernel/repack work.
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
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402


@dataclass
class FixtureResult:
    fixture_file: str
    layer_id: int
    prompt_id: int
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int
    ref_ms_mean: float
    candidate_max_abs: float | None
    candidate_cosine: float | None
    candidate_exact: int | None
    passed: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _dense_forward(ffn_in: torch.Tensor, w: dict[str, Any]) -> torch.Tensor:
    gate = F.linear(ffn_in, w["mlp.gate_proj.weight"])
    up = F.linear(ffn_in, w["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, w["mlp.down_proj.weight"])


def _time_dense(ffn_in: torch.Tensor, w: dict[str, Any], warmup: int, repeat: int) -> tuple[torch.Tensor, float]:
    out = _dense_forward(ffn_in, w)
    if ffn_in.is_cuda:
        torch.cuda.synchronize()
    for _ in range(warmup):
        out = _dense_forward(ffn_in, w)
    if ffn_in.is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            out = _dense_forward(ffn_in, w)
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end)) / float(repeat)
    t0 = time.time()
    for _ in range(repeat):
        out = _dense_forward(ffn_in, w)
    return out, (time.time() - t0) * 1000.0 / float(repeat)


def _candidate_path(candidate_dir: Path, fixture_file: str) -> Path | None:
    name = Path(fixture_file).name
    stem = Path(fixture_file).stem
    for p in (candidate_dir / name, candidate_dir / f"{stem}.safetensors"):
        if p.exists():
            return p
    return None


def _load_candidate(candidate_dir: Path, fixture_file: str, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    p = _candidate_path(candidate_dir, fixture_file)
    if p is None:
        return None
    data = load_file(str(p), device=device)
    for key in ("ffn_output", "candidate_output", "output"):
        if key in data:
            return data[key].to(dtype)
    raise KeyError(f"{p} has no ffn_output/candidate_output/output tensor")


def run_contract(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fixtures_dir = Path(args.fixtures)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "lynn-qwen35-9b-dense-ffn-fixture-v1":
        raise ValueError(f"unexpected fixture schema: {manifest.get('schema')}")

    layer_cache: dict[int, dict[str, Any]] = {}
    results: list[FixtureResult] = []
    candidate_dir = Path(args.candidate_output_dir) if args.candidate_output_dir else None

    for item in manifest["fixtures"]:
        layer_id = int(item["layer_id"])
        fixture = load_file(str(fixtures_dir / item["file"]), device=args.device)
        ffn_in = fixture["ffn_in"].to(dtype)
        expected = fixture["ffn_output"].to(dtype)

        if layer_id not in layer_cache:
            w = _load_weights_from_fixture_dir(fixtures_dir, layer_id, args.device, dtype)
            if w is None:
                if not args.model:
                    raise ValueError("fixtures do not contain weights; pass --model to reload layer weights")
                w, inferred = load_qwen36_layer(
                    args.model,
                    layer_id,
                    num_experts=0,
                    device=args.device,
                    dequant_dtype=dtype,
                )
                if inferred.get("is_moe", False):
                    raise RuntimeError(f"layer {layer_id} loaded as MoE; expected dense")
            layer_cache[layer_id] = w

        out, ref_ms = _time_dense(ffn_in, layer_cache[layer_id], args.warmup, args.repeat)
        m = _metrics(expected, out)
        fail_reasons: list[str] = []
        if args.require_exact and m["exact"] != 1:
            fail_reasons.append("not exact")
        if m["max_abs"] > args.max_abs_threshold:
            fail_reasons.append(f"max_abs {m['max_abs']:.6g} > {args.max_abs_threshold}")
        if m["cosine"] < args.cosine_threshold:
            fail_reasons.append(f"cosine {m['cosine']:.9f} < {args.cosine_threshold}")

        cand_max_abs = None
        cand_cosine = None
        cand_exact = None
        if candidate_dir is not None:
            cand = _load_candidate(candidate_dir, item["file"], args.device, dtype)
            if cand is None:
                fail_reasons.append("candidate output missing")
            else:
                cm = _metrics(expected, cand)
                cand_max_abs = cm["max_abs"]
                cand_cosine = cm["cosine"]
                cand_exact = int(cm["exact"])
                if cand_max_abs > args.candidate_max_abs_threshold:
                    fail_reasons.append(
                        f"candidate max_abs {cand_max_abs:.6g} > {args.candidate_max_abs_threshold}"
                    )
                if cand_cosine < args.candidate_cosine_threshold:
                    fail_reasons.append(
                        f"candidate cosine {cand_cosine:.9f} < {args.candidate_cosine_threshold}"
                    )

        results.append(
            FixtureResult(
                fixture_file=item["file"],
                layer_id=layer_id,
                prompt_id=int(item["prompt_id"]),
                max_abs=m["max_abs"],
                mean_abs=m["mean_abs"],
                rel_l2=m["rel_l2"],
                cosine=m["cosine"],
                exact=int(m["exact"]),
                ref_ms_mean=ref_ms,
                candidate_max_abs=cand_max_abs,
                candidate_cosine=cand_cosine,
                candidate_exact=cand_exact,
                passed=not fail_reasons,
                fail_reasons=fail_reasons,
            )
        )

    ref_ms = [r.ref_ms_mean for r in results]
    passed = sum(int(r.passed) for r in results)
    exact = sum(r.exact for r in results)
    candidate_exact_vals = [r.candidate_exact for r in results if r.candidate_exact is not None]
    report = {
        "schema": "lynn-qwen35-9b-dense-ffn-fixture-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "model": args.model,
        "dtype": args.dtype,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "thresholds": {
            "require_exact": args.require_exact,
            "max_abs": args.max_abs_threshold,
            "cosine": args.cosine_threshold,
            "candidate_max_abs": args.candidate_max_abs_threshold,
            "candidate_cosine": args.candidate_cosine_threshold,
        },
        "passed": passed,
        "total": len(results),
        "exact": exact,
        "max_abs_max": max((r.max_abs for r in results), default=None),
        "cosine_min": min((r.cosine for r in results), default=None),
        "ref_ms_mean": statistics.mean(ref_ms) if ref_ms else None,
        "ref_ms_median": statistics.median(ref_ms) if ref_ms else None,
        "candidate_exact": sum(int(x or 0) for x in candidate_exact_vals) if candidate_exact_vals else None,
        "decision": "DENSE_FFN_FIXTURE_GREEN" if passed == len(results) else "DENSE_FFN_FIXTURE_RED",
        "results": [r.to_dict() for r in results],
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Qwen3.5-9B dense FFN fixtures.")
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--candidate-output-dir", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=32)
    ap.add_argument("--max-abs-threshold", type=float, default=0.0)
    ap.add_argument("--cosine-threshold", type=float, default=0.999999)
    ap.add_argument("--candidate-max-abs-threshold", type=float, default=0.0)
    ap.add_argument("--candidate-cosine-threshold", type=float, default=1.0)
    ap.add_argument("--require-exact", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run_contract(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "decision": report["decision"],
        "passed": report["passed"],
        "total": report["total"],
        "exact": report["exact"],
        "ref_ms_mean": report["ref_ms_mean"],
        "max_abs_max": report["max_abs_max"],
        "cosine_min": report["cosine_min"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["passed"] == report["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
