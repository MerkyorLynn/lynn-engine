#!/usr/bin/env python3
"""P136 · MoE slot repack contract test.

Purpose: Load p135 slot-repacked fixtures and verify that the slot-only
computation reproduces the stored routed_output within tolerance.

Contract metrics:
  - max_abs:  max|stored_routed - slot_computed|
  - mean_abs: mean|stored_routed - slot_computed|
  - rel_l2:   ||stored - slot||_2 / ||stored||_2
  - cosine:   cos(stored, slot)
  - exact:    1 if max_abs == 0 else 0
  - ref_ms:   reference kernel latency (unique-active-expert path)
  - slot_repack_ms: slot-only kernel latency
  - pass/fail: max_abs <= threshold AND cosine >= threshold

Usage:
  # Self-check (slot-only vs stored reference):
  python benchmarks/p136_moe_slot_repack_contract.py \
    --fixtures reports/qwen36_35b/p135_repacked_fixtures \
    --model-dir /path/to/model \
    --device cuda

  # With custom thresholds:
  python benchmarks/p136_moe_slot_repack_contract.py \
    --fixtures reports/qwen36_35b/p135_repacked_fixtures \
    --model-dir /path/to/model \
    --max-abs-threshold 1e-3 \
    --cosine-threshold 0.999999
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Contract metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class ContractResult:
    fixture_file: str
    layer_id: int
    prompt_id: int
    max_abs: float
    mean_abs: float
    rel_l2: float
    cosine: float
    exact: int
    ref_ms: float
    slot_repack_ms: float
    passed: bool
    fail_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_metrics(ref: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Compute all contract metrics between reference and candidate outputs."""
    rf = ref.float().flatten()
    cf = candidate.float().flatten()
    diff = rf - cf

    ref_norm = torch.linalg.vector_norm(rf).clamp_min(1e-12)
    diff_norm = torch.linalg.vector_norm(diff)

    cosine = torch.dot(rf, cf) / (
        torch.linalg.vector_norm(rf).clamp_min(1e-12)
        * torch.linalg.vector_norm(cf).clamp_min(1e-12)
    )

    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rel_l2": float(diff_norm / ref_norm),
        "cosine": float(cosine),
        "exact": 1 if float(diff.abs().max()) == 0.0 else 0,
    }


# ─────────────────────────────────────────────────────────────
# Computation kernels
# ─────────────────────────────────────────────────────────────

def _moe_reference_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
) -> torch.Tensor:
    """Reference routed-only path (unique active experts)."""
    K = expert_ids.shape[0]
    h_flat = hidden_in

    active_experts = torch.unique(expert_ids).tolist()
    moe_out = torch.zeros_like(h_flat)

    expert_indices = expert_ids.unsqueeze(0).long()
    routing_w = routing_weights.unsqueeze(0).to(h_flat.dtype)

    for e in active_experts:
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]

        if "mlp.experts.gate_up_proj" in layer_weights and "mlp.experts.down_proj" in layer_weights:
            gate_up = F.linear(x_e, layer_weights["mlp.experts.gate_up_proj"][e])
            gate, up = gate_up.chunk(2, dim=-1)
            ffn_e = F.linear(F.silu(gate) * up, layer_weights["mlp.experts.down_proj"][e])
        else:
            gate = F.linear(x_e, layer_weights[f"mlp.experts.{e}.gate_proj.weight"])
            up = F.linear(x_e, layer_weights[f"mlp.experts.{e}.up_proj.weight"])
            ffn_e = F.linear(F.silu(gate) * up, layer_weights[f"mlp.experts.{e}.down_proj.weight"])

        weight_e = routing_w[token_idx, slot_idx].unsqueeze(-1)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    return moe_out


def _moe_slot_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    slot_gate_up: torch.Tensor,
    slot_down: torch.Tensor,
) -> torch.Tensor:
    """Slot-only routed path — no full expert table, only 8 slot weights."""
    K = expert_ids.shape[0]
    h = hidden_in  # [1, hidden]
    out = torch.zeros_like(h)

    for k in range(K):
        gate_up = F.linear(h, slot_gate_up[k])
        gate, up = gate_up.chunk(2, dim=-1)
        ffn = F.linear(F.silu(gate) * up, slot_down[k])
        out += ffn * routing_weights[k].to(h.dtype)

    return out


# ─────────────────────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────────────────────

def _bench_fn(fn: callable, warmup: int = 3, iters: int = 10) -> float:
    """Benchmark a function, return mean ms per call."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / iters)
    else:
        t0 = time.time()
        for _ in range(iters):
            fn()
        return (time.time() - t0) * 1000 / iters


# ─────────────────────────────────────────────────────────────
# Main contract runner
# ─────────────────────────────────────────────────────────────

def run_contract(
    fixtures_dir: str,
    model_dir: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_abs_threshold: float = 1e-3,
    cosine_threshold: float = 0.999999,
    warmup: int = 3,
    iters: int = 10,
) -> list[ContractResult]:
    """Run the slot repack contract test suite."""
    from safetensors.torch import load_file

    fixtures_path = Path(fixtures_dir)
    manifest_path = fixtures_path / "manifest.json"

    if not manifest_path.exists():
        print(f"[p136] ERROR: manifest.json not found in {fixtures_path}", flush=True)
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[p136] Fixtures: {fixtures_path}")
    print(f"[p136] Fixtures count: {manifest['num_fixtures']}")

    effective_model_dir = model_dir or manifest.get("model_dir")
    if effective_model_dir is None:
        print("[p136] WARNING: No model_dir provided or found in manifest; ref_ms will be 0.0")

    # Load reference layer weights if model dir available
    layer_cache: dict[int, dict[str, Any]] = {}
    if effective_model_dir and Path(effective_model_dir).exists():
        needed_layers = sorted({entry["layer_id"] for entry in manifest["fixtures"]})
        print(f"[p136] Loading layer weights for ref timing: {needed_layers}")
        for layer_id in needed_layers:
            print(f"[p136] Loading layer {layer_id}...", flush=True)
            w, _ = load_qwen36_layer(
                effective_model_dir,
                layer_id,
                num_experts=manifest.get("num_experts", 256),
                device=device,
                dequant_dtype=dtype,
            )
            layer_cache[layer_id] = w
    else:
        print("[p136] Model dir not available; skipping reference timing.")

    results: list[ContractResult] = []

    print(f"\n{'='*80}")
    print(
        f"{'FIXTURE':<40} {'MAX_ABS':>10} {'MEAN_ABS':>10} "
        f"{'COS':>8} {'EXACT':>5} {'REF_MS':>8} {'SLOT_MS':>8} {'PASS':>5}"
    )
    print(f"{'='*80}")

    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        fixture_data = load_file(str(fixtures_path / fixture_file), device=device)
        hidden_in = fixture_data["hidden_in"].to(dtype)
        expert_ids = fixture_data["expert_ids"]
        routing_weights = fixture_data["routing_weights"]
        slot_gate_up = fixture_data["slot_gate_up_weight"].to(dtype)
        slot_down = fixture_data["slot_down_weight"].to(dtype)
        expected_routed = fixture_data["routed_output"].to(dtype)

        # Slot-only computation
        slot_out = _moe_slot_only(
            hidden_in, expert_ids, routing_weights, slot_gate_up, slot_down
        )
        slot_ms = _bench_fn(
            lambda: _moe_slot_only(
                hidden_in, expert_ids, routing_weights, slot_gate_up, slot_down
            ),
            warmup=warmup,
            iters=iters,
        )

        # Reference timing (if layer weights available)
        ref_ms = 0.0
        if layer_id in layer_cache:
            ref_ms = _bench_fn(
                lambda: _moe_reference_routed_only(
                    hidden_in, expert_ids, routing_weights, layer_cache[layer_id]
                ),
                warmup=warmup,
                iters=iters,
            )

        # Metrics: compare slot-computed vs stored routed_output
        metrics = _compute_metrics(expected_routed, slot_out)

        fail_reasons = []
        if metrics["max_abs"] > max_abs_threshold:
            fail_reasons.append(f"max_abs {metrics['max_abs']:.6e} > {max_abs_threshold}")
        if metrics["exact"] == 0 and metrics["cosine"] < cosine_threshold:
            fail_reasons.append(f"cosine {metrics['cosine']:.6f} < {cosine_threshold}")

        passed = len(fail_reasons) == 0

        result = ContractResult(
            fixture_file=fixture_file,
            layer_id=layer_id,
            prompt_id=prompt_id,
            max_abs=metrics["max_abs"],
            mean_abs=metrics["mean_abs"],
            rel_l2=metrics["rel_l2"],
            cosine=metrics["cosine"],
            exact=metrics["exact"],
            ref_ms=ref_ms,
            slot_repack_ms=slot_ms,
            passed=passed,
            fail_reasons=fail_reasons,
        )
        results.append(result)

        status = "GREEN" if passed else "RED"
        fixture_label = f"L{layer_id:02d}/P{prompt_id:02d}"
        print(
            f"{fixture_label:<40} "
            f"{metrics['max_abs']:>10.2e} "
            f"{metrics['mean_abs']:>10.2e} "
            f"{metrics['cosine']:>8.6f} "
            f"{metrics['exact']:>5d} "
            f"{ref_ms:>8.3f} "
            f"{slot_ms:>8.3f} "
            f"{status:>5}",
            flush=True,
        )

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    max_abs_max = max(r.max_abs for r in results)
    ref_ms_mean = sum(r.ref_ms for r in results) / total if total else 0.0
    slot_ms_mean = sum(r.slot_repack_ms for r in results) / total if total else 0.0

    print(f"\n{'='*80}")
    print("CONTRACT SUMMARY")
    print(f"  Thresholds: max_abs <= {max_abs_threshold}, cosine >= {cosine_threshold}")
    print(f"  Total:      {total}")
    print(f"  Passed:     {passed_count} GREEN")
    print(f"  Failed:     {total - passed_count} RED")
    print(f"  max_abs_max: {max_abs_max:.6e}")
    print(f"  ref_ms_mean: {ref_ms_mean:.3f} ms")
    print(f"  slot_repack_ms_mean: {slot_ms_mean:.3f} ms")
    if passed_count == total:
        print(f"\n  VERDICT: ALL {total} fixtures GREEN.")
        print(f"  Slot repack is mathematically consistent with reference.")
    else:
        print(f"\n  VERDICT: {total - passed_count} fixture(s) FAILED.")
        for r in results:
            if not r.passed:
                print(f"    {r.fixture_file}: {'; '.join(r.fail_reasons)}")
    print(f"{'='*80}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="MoE slot repack contract test.")
    ap.add_argument("--fixtures", required=True, help="Path to p135 repacked fixture directory.")
    ap.add_argument("--model-dir", default=None, help="Model dir for reference timing. Defaults to manifest.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--max-abs-threshold", type=float, default=1e-3)
    ap.add_argument("--cosine-threshold", type=float, default=0.999999)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None, help="Output JSON report path.")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    results = run_contract(
        fixtures_dir=args.fixtures,
        model_dir=args.model_dir,
        device=args.device,
        dtype=dtype,
        max_abs_threshold=args.max_abs_threshold,
        cosine_threshold=args.cosine_threshold,
        warmup=args.warmup,
        iters=args.iters,
    )

    if not results:
        return 1

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    max_abs_max = max(r.max_abs for r in results)
    ref_ms_mean = sum(r.ref_ms for r in results) / total
    slot_ms_mean = sum(r.slot_repack_ms for r in results) / total

    out_path = args.out or str(Path(args.fixtures) / "p136_slot_repack_contract_report.json")
    report = {
        "schema": "lynn-moe-slot-repack-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixtures_dir": args.fixtures,
        "model_dir": args.model_dir,
        "max_abs_threshold": args.max_abs_threshold,
        "cosine_threshold": args.cosine_threshold,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "verdict": "GREEN" if passed == total else "RED",
        "max_abs_max": max_abs_max,
        "ref_ms_mean": ref_ms_mean,
        "slot_repack_ms_mean": slot_ms_mean,
        "results": [r.to_dict() for r in results],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p136] Report written: {out_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
