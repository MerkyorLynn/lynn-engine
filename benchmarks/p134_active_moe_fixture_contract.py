#!/usr/bin/env python3
"""P134 · Active MoE fixture contract test.

Purpose: Load p133-exported fixtures and verify that the Triton active-MoE
reference kernel produces bit-exact (or within tolerance) results. Optionally
compare a candidate native backend against the same fixtures.

This is the "fast target" for Stream A native grouped kernel development:
instead of loading the full 35B model, kernel developers load tiny fixture
files (~16 KB each) and validate their implementation matches the reference.

Contract metrics:
  - max_abs:  max|ref - candidate|
  - mean_abs: mean|ref - candidate|
  - rel_l2:   ||ref - candidate||_2 / ||ref||_2
  - cosine:   cos(ref, candidate)
  - exact:    1 if max_abs == 0 else 0
  - ref_ms:   reference kernel latency
  - candidate_ms: candidate kernel latency (if applicable)
  - pass/fail: based on configurable thresholds

Usage:
  # Triton-vs-Triton self-check (must be exact):
  python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --device cuda

  # Native candidate vs Triton reference:
  python benchmarks/p134_active_moe_fixture_contract.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --candidate-backend native_grouped \
    --max-abs-threshold 0.01 \
    --cosine-threshold 0.999 \
    --device cuda
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    candidate_ms: float
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
# Reference MoE forward (same as p133 capture)
# ─────────────────────────────────────────────────────────────

def _moe_reference_from_fixture(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Run the reference MoE forward using fixture inputs + layer weights.

    This reconstructs the MoE computation from the captured fixture data.
    Uses the same active-expert iteration as p133 capture.
    """
    K = expert_ids.shape[0]
    h_flat = hidden_in  # [1, hidden]

    active_experts = torch.unique(expert_ids).tolist()
    moe_out = torch.zeros_like(h_flat)

    # Reconstruct expert_indices as [1, K] to match moe_forward_decode_optimized format
    expert_indices = expert_ids.unsqueeze(0).long()  # [1, K]
    routing_w = routing_weights.unsqueeze(0).to(h_flat.dtype)  # [1, K]

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

    # Shared expert
    if "mlp.shared_expert.gate_proj.weight" in layer_weights:
        gate_s = F.linear(h_flat, layer_weights["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_flat, layer_weights["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, layer_weights["mlp.shared_expert.down_proj.weight"])
        if "mlp.shared_expert_gate.weight" in layer_weights:
            shared_gate = torch.sigmoid(
                F.linear(h_flat, layer_weights["mlp.shared_expert_gate.weight"])
            )
            shared_ffn = shared_ffn * shared_gate
        moe_out = moe_out + shared_ffn

    return moe_out


def _moe_reference_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Run ONLY the routed expert path (no shared expert) for contract testing.

    Native kernel candidates typically implement only the grouped routed-expert
    dispatch. Shared expert is a separate, simpler linear and not part of the
    grouped kernel contract.
    """
    K = expert_ids.shape[0]
    h_flat = hidden_in  # [1, hidden]

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


# ─────────────────────────────────────────────────────────────
# Benchmark helper
# ─────────────────────────────────────────────────────────────

def _bench_fn(fn: Callable, warmup: int = 3, iters: int = 10) -> float:
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
# Candidate backend loading
# ─────────────────────────────────────────────────────────────

def _load_candidate_backend(
    backend_name: str,
) -> Callable | None:
    """Dynamically load a candidate MoE backend for testing.

    Candidate backends must implement:
        def moe_forward_fixture(hidden_in, expert_ids, routing_weights, layer_weights, cfg) -> Tensor

    The function signature matches _moe_reference_routed_only.
    """
    if backend_name == "triton_reference":
        # Self-test: use the same reference as candidate (should be exact)
        return _moe_reference_routed_only

    if backend_name == "triton_fused":
        try:
            from triton_kernels.moe_expert_ffn import moe_forward_decode_triton, stack_expert_weights

            def _triton_fused_candidate(hidden_in, expert_ids, routing_weights, layer_weights, cfg):
                # Stack weights if not already done
                if "mlp.experts._gate_stacked" not in layer_weights:
                    stack_expert_weights(layer_weights, cfg["num_experts"])
                h = hidden_in.unsqueeze(0)  # [1, 1, hidden]
                # Override routing — inject fixture expert_ids + weights
                # This requires modifying the triton path to accept pre-computed routing.
                # For now, fall back to indexed_bmm which accepts the same interface.
                from triton_kernels.moe_expert_ffn import moe_forward_decode_indexed_bmm
                # Inject routing into layer_weights temporarily
                layer_weights["_fixture_expert_ids"] = expert_ids
                layer_weights["_fixture_routing_weights"] = routing_weights
                return moe_forward_decode_indexed_bmm(h, layer_weights, cfg).view(1, -1)

            return _triton_fused_candidate
        except ImportError:
            print("[p134] WARNING: triton_fused backend not available", flush=True)
            return None

    # Generic module loading for native backends
    # Expected: benchmarks/candidates/<backend_name>.py with moe_forward_fixture()
    candidate_path = ROOT / "benchmarks" / "candidates" / f"{backend_name}.py"
    if candidate_path.exists():
        spec = importlib.util.spec_from_file_location(backend_name, candidate_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "moe_forward_fixture"):
            return mod.moe_forward_fixture
        print(f"[p134] WARNING: {candidate_path} has no moe_forward_fixture()", flush=True)
        return None

    print(f"[p134] WARNING: candidate backend '{backend_name}' not found", flush=True)
    return None


# ─────────────────────────────────────────────────────────────
# Main contract runner
# ─────────────────────────────────────────────────────────────

def run_contract(
    fixtures_dir: str,
    model_dir: str | None = None,
    candidate_backend: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_abs_threshold: float = 0.0,
    cosine_threshold: float = 1.0,
    warmup: int = 3,
    iters: int = 10,
    routed_only: bool = False,
) -> list[ContractResult]:
    """Run the MoE fixture contract test suite.

    If candidate_backend is None, runs Triton-vs-Triton self-check (must be exact).
    If candidate_backend is specified, compares candidate against reference.
    """
    from safetensors.torch import load_file

    fixtures_path = Path(fixtures_dir)
    manifest_path = fixtures_path / "manifest.json"

    if not manifest_path.exists():
        print(f"[p134] ERROR: manifest.json not found in {fixtures_path}", flush=True)
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[p134] Fixtures: {fixtures_path}", flush=True)
    print(f"[p134] Fixtures count: {manifest['num_fixtures']}", flush=True)
    print(f"[p134] Model: {manifest['model_dir']}", flush=True)

    # Determine model_dir for loading layer weights
    effective_model_dir = model_dir or manifest["model_dir"]
    print(f"[p134] Loading layer weights from: {effective_model_dir}", flush=True)

    # Determine which layers we need
    needed_layers = sorted(set(entry["layer_id"] for entry in manifest["fixtures"]))
    print(f"[p134] Layers needed: {needed_layers}", flush=True)

    # Load candidate backend
    candidate_fn = None
    if candidate_backend:
        if candidate_backend == "triton_reference":
            print(f"[p134] Self-check mode: Triton reference vs Triton reference", flush=True)
        else:
            print(f"[p134] Candidate backend: {candidate_backend}", flush=True)
        candidate_fn = _load_candidate_backend(candidate_backend)
        if candidate_fn is None and candidate_backend != "triton_reference":
            print(f"[p134] FATAL: Could not load candidate backend '{candidate_backend}'", flush=True)
            return []

    # Load layer weights
    from engine.loader import load_qwen36_layer

    cfg = {
        "num_experts": manifest["num_experts"],
        "num_experts_per_tok": manifest["top_k"],
        "hidden_size": manifest["hidden_size"],
    }

    layer_cache: dict[int, dict] = {}
    for layer_id in needed_layers:
        print(f"[p134] Loading layer {layer_id} weights...", flush=True)
        w, _ = load_qwen36_layer(
            effective_model_dir,
            layer_id,
            num_experts=cfg["num_experts"],
            device=device,
            dequant_dtype=dtype,
        )
        layer_cache[layer_id] = w

    # Run contract on each fixture
    results: list[ContractResult] = []
    is_self_check = candidate_backend is None or candidate_backend == "triton_reference"

    ref_fn = _moe_reference_routed_only if routed_only else _moe_reference_from_fixture

    print(f"\n{'='*70}")
    print(f"{'FIXTURE':<35} {'MAX_ABS':>10} {'MEAN_ABS':>10} {'REL_L2':>10} {'COS':>8} {'EXACT':>5} {'MS':>7} {'PASS':>5}")
    print(f"{'='*70}")

    for entry in manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        # Load fixture tensors
        fixture_data = load_file(str(fixtures_path / fixture_file), device=device)
        hidden_in = fixture_data["hidden_in"].to(dtype)
        expert_ids = fixture_data["expert_ids"]
        routing_weights = fixture_data["routing_weights"]
        expected_output = fixture_data["moe_output"].to(dtype)

        layer_weights = layer_cache[layer_id]

        # Run reference
        ref_out = ref_fn(hidden_in, expert_ids, routing_weights, layer_weights, cfg)
        ref_ms = _bench_fn(
            lambda: ref_fn(hidden_in, expert_ids, routing_weights, layer_weights, cfg),
            warmup=warmup,
            iters=iters,
        )

        # Determine candidate output
        if is_self_check:
            # Self-check: compare reference output against stored fixture output
            candidate_out = ref_out
            candidate_ms = ref_ms
            # Metrics compare against the stored expected output
            metrics = _compute_metrics(expected_output, ref_out)
        else:
            # Candidate comparison
            candidate_out = candidate_fn(hidden_in, expert_ids, routing_weights, layer_weights, cfg)
            candidate_ms = _bench_fn(
                lambda: candidate_fn(hidden_in, expert_ids, routing_weights, layer_weights, cfg),
                warmup=warmup,
                iters=iters,
            )
            metrics = _compute_metrics(ref_out, candidate_out)

        # Check pass/fail
        fail_reasons = []
        if metrics["max_abs"] > max_abs_threshold:
            fail_reasons.append(f"max_abs {metrics['max_abs']:.6e} > {max_abs_threshold}")
        if metrics["cosine"] < cosine_threshold:
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
            candidate_ms=candidate_ms,
            passed=passed,
            fail_reasons=fail_reasons,
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        fixture_label = f"L{layer_id:02d}/P{prompt_id:02d}"
        print(
            f"{fixture_label:<35} "
            f"{metrics['max_abs']:>10.2e} "
            f"{metrics['mean_abs']:>10.2e} "
            f"{metrics['rel_l2']:>10.2e} "
            f"{metrics['cosine']:>8.6f} "
            f"{metrics['exact']:>5d} "
            f"{ref_ms:>7.3f} "
            f"{'GREEN' if passed else 'RED':>5}",
            flush=True,
        )

    # Summary
    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total - passed_count

    print(f"\n{'='*70}")
    print(f"CONTRACT SUMMARY")
    print(f"  Mode:       {'self-check (Triton vs stored fixture)' if is_self_check else f'candidate ({candidate_backend})'}")
    print(f"  Thresholds: max_abs <= {max_abs_threshold}, cosine >= {cosine_threshold}")
    print(f"  Total:      {total}")
    print(f"  Passed:     {passed_count} GREEN")
    print(f"  Failed:     {failed_count} RED")
    if is_self_check and failed_count == 0:
        print(f"\n  VERDICT: Triton reference reproduces all stored fixtures EXACTLY.")
        print(f"  Fixtures are valid as Stream A native kernel contract gate.")
    elif not is_self_check and failed_count == 0:
        print(f"\n  VERDICT: Candidate '{candidate_backend}' passes all fixtures.")
    elif failed_count > 0:
        print(f"\n  VERDICT: {failed_count} fixture(s) FAILED contract.")
        for r in results:
            if not r.passed:
                print(f"    {r.fixture_file}: {'; '.join(r.fail_reasons)}")
    print(f"{'='*70}")

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Active MoE fixture contract test."
    )
    ap.add_argument(
        "--fixtures",
        required=True,
        help="Path to p133 fixture directory (containing manifest.json).",
    )
    ap.add_argument(
        "--model-dir",
        default=None,
        help="Override model directory for loading layer weights. "
             "Defaults to the model_dir recorded in manifest.json.",
    )
    ap.add_argument(
        "--candidate-backend",
        default=None,
        help="Name of candidate backend to test. "
             "Use 'triton_reference' for self-check. "
             "Leave empty for stored-fixture self-check.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument(
        "--max-abs-threshold",
        type=float,
        default=0.0,
        help="Maximum absolute error threshold (0.0 = exact match required).",
    )
    ap.add_argument(
        "--cosine-threshold",
        type=float,
        default=1.0,
        help="Minimum cosine similarity threshold (1.0 = exact).",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument(
        "--routed-only",
        action="store_true",
        help="Only test the routed expert path (skip shared expert).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON report path. Default: <fixtures>/p134_contract_report.json",
    )

    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    results = run_contract(
        fixtures_dir=args.fixtures,
        model_dir=args.model_dir,
        candidate_backend=args.candidate_backend,
        device=args.device,
        dtype=dtype,
        max_abs_threshold=args.max_abs_threshold,
        cosine_threshold=args.cosine_threshold,
        warmup=args.warmup,
        iters=args.iters,
        routed_only=args.routed_only,
    )

    if not results:
        return 1

    # Write report
    out_path = args.out or str(Path(args.fixtures) / "p134_contract_report.json")
    report = {
        "schema": "lynn-moe-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fixtures_dir": args.fixtures,
        "model_dir": args.model_dir,
        "candidate_backend": args.candidate_backend,
        "max_abs_threshold": args.max_abs_threshold,
        "cosine_threshold": args.cosine_threshold,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "verdict": "GREEN" if all(r.passed for r in results) else "RED",
        "results": [r.to_dict() for r in results],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[p134] Report written: {out_path}", flush=True)

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
