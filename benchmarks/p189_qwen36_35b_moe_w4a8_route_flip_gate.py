#!/usr/bin/env python3
"""P189 - Qwen3.6-35B MoE W4A8 route-flip gate.

Checks whether FP8 fake-quant of hidden_in changes router top-K expert
selections in Qwen3.6-35B MoE layers.  If the router flips, W4A8 activation
noise is too high for safe MoE deployment.

Consumes p133 MoE fixtures (hidden_in, expert_ids, routing_weights).
Produces per-fixture route comparison and aggregate verdict.

Verdicts:
  CLOSED_ROUTE_FLIP    - any fixture has topk_exact=0 (experts changed)
  MOE_W4A8_ROUTE_AMBER - some fixtures have flips but high Jaccard
  MOE_W4A8_ROUTE_GREEN - all fixtures have topk_exact=1

Usage:
  python benchmarks/p189_qwen36_35b_moe_w4a8_route_flip_gate.py \\
    --fixtures /root/autodl-tmp/reports/qwen36_35b/p133_fixtures_official_w4a16 \\
    --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \\
    --out /root/autodl-tmp/reports/qwen36_35b/p189_moe_w4a8_route_flip_gate.json
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


# ─────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────
JACCARD_GREEN = 1.0    # all experts match
JACCARD_AMBER = 0.75   # at least 6/8 match
WEIGHT_MAX_ABS_GREEN = 0.01
WEIGHT_MAX_ABS_AMBER = 0.05
COSINE_GREEN = 0.9999
COSINE_AMBER = 0.999


@dataclass
class FixtureRow:
    fixture_file: str
    layer_id: int
    prompt_id: int
    topk_exact: int          # 1 if expert_ids identical, 0 otherwise
    topk_jaccard: float      # |intersection| / |union|
    route_flip_count: int    # number of expert slots that differ
    routing_weight_max_abs: float  # max |delta| on routing weights
    routing_weight_cosine: float   # cosine sim of routing weight vectors
    baseline_expert_ids: list[int]
    w4a8_expert_ids: list[int]
    flipped_slots: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    an = torch.linalg.vector_norm(af).clamp_min(1e-12)
    bn = torch.linalg.vector_norm(bf).clamp_min(1e-12)
    return float(torch.dot(af, bf) / (an * bn))


def _route(hidden_in: torch.Tensor, gate_weight: torch.Tensor, top_k: int):
    """Run router on hidden_in, return (expert_ids, routing_weights)."""
    logits = F.linear(hidden_in, gate_weight)  # [1, num_experts]
    rw, ei = torch.topk(logits, top_k, dim=-1)
    rw = F.softmax(rw, dim=-1, dtype=torch.float32)
    return ei[0], rw[0]


def _load_gate_weight(model_path: str, layer_id: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Load router gate.weight for a single MoE layer."""
    weights, _ = load_qwen36_layer(model_path, layer_id, num_experts=256, device=device, dequant_dtype=dtype)
    return weights["mlp.gate.weight"].to(dtype)


# ─────────────────────────────────────────────────────────────
# Gate logic
# ─────────────────────────────────────────────────────────────
def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fixtures_dir = Path(args.fixtures)

    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found in {fixtures_dir}", file=sys.stderr)
        print(f"Run p133 first: python benchmarks/p133_export_active_moe_fixtures.py", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema", "")
    if "moe-fixture" not in schema and "fixture" not in schema:
        print(f"WARNING: unexpected fixture schema: {schema}", file=sys.stderr)

    top_k = manifest.get("top_k", 8)
    num_experts = manifest.get("num_experts", 256)

    # Set W4A8 fake-quant env
    old_fmt = os.environ.get("LYNN_W4A8_FAKE_QUANT_FORMAT")
    old_gran = os.environ.get("LYNN_W4A8_FAKE_QUANT_GRANULARITY")
    os.environ["LYNN_W4A8_FAKE_QUANT_FORMAT"] = args.fp8_format
    os.environ["LYNN_W4A8_FAKE_QUANT_GRANULARITY"] = args.granularity

    try:
        gate_weight_cache: dict[int, torch.Tensor] = {}
        rows: list[FixtureRow] = []

        for item in manifest["fixtures"]:
            layer_id = int(item["layer_id"])
            fixture_file = item["file"]
            prompt_id = int(item["prompt_id"])

            # Load fixture
            fixture = load_file(str(fixtures_dir / fixture_file), device=args.device)
            hidden_in = fixture["hidden_in"].to(dtype).unsqueeze(0) if fixture["hidden_in"].dim() == 1 else fixture["hidden_in"].to(dtype)
            baseline_ei = fixture["expert_ids"].tolist() if isinstance(fixture["expert_ids"], torch.Tensor) else list(fixture["expert_ids"])
            baseline_rw = fixture["routing_weights"].to(torch.float32)

            # Load gate weight (cached per layer)
            if layer_id not in gate_weight_cache:
                if not args.model:
                    print(f"ERROR: --model required to load router weights for layer {layer_id}", file=sys.stderr)
                    sys.exit(1)
                gate_weight_cache[layer_id] = _load_gate_weight(args.model, layer_id, args.device, dtype)
            gate_w = gate_weight_cache[layer_id]

            # W4A8 route: fake-quant hidden_in then route
            hidden_in_q = _fake_quant_fp8_activation(hidden_in)
            w4a8_ei_t, w4a8_rw_t = _route(hidden_in_q, gate_w, top_k)
            w4a8_ei = w4a8_ei_t.tolist()
            w4a8_rw = w4a8_rw_t

            # Compare
            baseline_set = set(baseline_ei)
            w4a8_set = set(w4a8_ei)
            intersection = baseline_set & w4a8_set
            union = baseline_set | w4a8_set
            jaccard = len(intersection) / len(union) if union else 1.0
            topk_exact = 1 if baseline_ei == w4a8_ei else 0
            flipped = [i for i in range(top_k) if baseline_ei[i] != w4a8_ei[i]]

            # Routing weight comparison
            rw_diff = (baseline_rw.float() - w4a8_rw.float()).abs()
            rw_max_abs = float(rw_diff.max())
            rw_cosine = _cosine(baseline_rw, w4a8_rw)

            rows.append(FixtureRow(
                fixture_file=fixture_file,
                layer_id=layer_id,
                prompt_id=prompt_id,
                topk_exact=topk_exact,
                topk_jaccard=jaccard,
                route_flip_count=len(flipped),
                routing_weight_max_abs=rw_max_abs,
                routing_weight_cosine=rw_cosine,
                baseline_expert_ids=baseline_ei,
                w4a8_expert_ids=w4a8_ei,
                flipped_slots=flipped,
            ))

        # Aggregate
        total = len(rows)
        exact_count = sum(r.topk_exact for r in rows)
        any_flip = any(r.topk_exact == 0 for r in rows)
        jaccard_min = min((r.topk_jaccard for r in rows), default=1.0)
        jaccard_mean = statistics.mean([r.topk_jaccard for r in rows]) if rows else 1.0
        flip_count_total = sum(r.route_flip_count for r in rows)
        rw_max_abs_max = max((r.routing_weight_max_abs for r in rows), default=0.0)
        rw_cosine_min = min((r.routing_weight_cosine for r in rows), default=1.0)

        # Per-layer aggregation
        layer_ids = sorted(set(r.layer_id for r in rows))
        per_layer = {}
        for lid in layer_ids:
            lr = [r for r in rows if r.layer_id == lid]
            per_layer[str(lid)] = {
                "total": len(lr),
                "exact": sum(r.topk_exact for r in lr),
                "jaccard_min": min(r.topk_jaccard for r in lr),
                "flip_count": sum(r.route_flip_count for r in lr),
            }

        # Verdict
        if not any_flip and rw_cosine_min >= COSINE_GREEN:
            verdict = "MOE_W4A8_ROUTE_GREEN"
        elif any_flip and jaccard_min >= JACCARD_AMBER and rw_cosine_min >= COSINE_AMBER:
            verdict = "MOE_W4A8_ROUTE_AMBER"
        elif any_flip:
            verdict = "CLOSED_ROUTE_FLIP"
        else:
            verdict = "MOE_W4A8_ROUTE_GREEN"

        return {
            "schema": "lynn-qwen36-35b-moe-w4a8-route-flip-gate-v1",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "fixtures": str(fixtures_dir),
            "model": args.model,
            "dtype": args.dtype,
            "fp8_format": args.fp8_format,
            "granularity": args.granularity,
            "top_k": top_k,
            "num_experts": num_experts,
            "thresholds": {
                "jaccard_green": JACCARD_GREEN,
                "jaccard_amber": JACCARD_AMBER,
                "cosine_green": COSINE_GREEN,
                "cosine_amber": COSINE_AMBER,
            },
            "verdict": verdict,
            "summary": {
                "total": total,
                "exact": exact_count,
                "any_flip": any_flip,
                "jaccard_min": jaccard_min,
                "jaccard_mean": jaccard_mean,
                "flip_count_total": flip_count_total,
                "routing_weight_max_abs": rw_max_abs_max,
                "routing_weight_cosine_min": rw_cosine_min,
            },
            "per_layer": per_layer,
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


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", required=True, help="p133 fixture directory")
    ap.add_argument("--model", default="", help="Model path for loading router weights")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--fp8-format", choices=["e4m3", "e5m2"], default="e4m3")
    ap.add_argument("--granularity", choices=["tensor", "row", "per16"], default="per16")
    args = ap.parse_args()

    report = run_gate(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Console summary
    s = report["summary"]
    print(f"verdict: {report['verdict']}")
    print(f"fixtures: {s['total']}, exact: {s['exact']}/{s['total']}, "
          f"flips: {s['flip_count_total']}, jaccard_min: {s['jaccard_min']:.3f}")
    print(f"rw_cosine_min: {s['routing_weight_cosine_min']:.6f}, "
          f"rw_max_abs: {s['routing_weight_max_abs']:.6f}")
    print(f"out: {out_path}")

    # Print flipped fixtures if any
    for r in report["results"]:
        if r["topk_exact"] == 0:
            print(f"  FLIP: layer={r['layer_id']} prompt={r['prompt_id']} "
                  f"baseline={r['baseline_expert_ids']} w4a8={r['w4a8_expert_ids']} "
                  f"jaccard={r['topk_jaccard']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
