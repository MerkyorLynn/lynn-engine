#!/usr/bin/env python3
"""P134 candidate backend: native output-owned BF16 MoE kernel.

This candidate operates on BF16 fused weights (gate_up_proj, down_proj) — the
same format as p134 fixtures. It calls the native CUDA kernel for the routed
expert path only (shared expert handled separately by p134 framework).

Usage with p134:
    python benchmarks/p134_active_moe_fixture_contract.py \
        --fixtures reports/qwen36_35b/p133_fixtures \
        --model-dir /path/to/bf16/model \
        --candidate-backend native_output_owned_bf16 \
        --max-abs-threshold 0.0 \
        --cosine-threshold 1.0

Or standalone:
    python benchmarks/candidates/native_output_owned_bf16.py \
        --fixtures reports/qwen36_35b/p133_fixtures \
        --model-dir /path/to/bf16/model
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_native_extension():
    """Load the lynn native CUDA extension with output-owned BF16 kernel."""
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    if not hasattr(ext, "moe_output_owned_bf16"):
        raise RuntimeError(
            "Native extension missing moe_output_owned_bf16. "
            "Rebuild with moe_output_owned_bf16.cu included."
        )
    return ext


def moe_forward_fixture(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """p134 candidate interface: run native BF16 MoE on fixture inputs.

    Returns full MoE output (routed + shared expert) to match p134 expectations.
    """
    ext = _load_native_extension()

    # hidden_in is [1, hidden] — flatten to [hidden]
    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
    routing_w = routing_weights.to(torch.float32).contiguous()

    # Get BF16 weights
    if "mlp.experts.gate_up_proj" not in layer_weights:
        raise ValueError("Candidate requires fused BF16 gate_up_proj weights")

    gate_up_w = layer_weights["mlp.experts.gate_up_proj"].contiguous()  # [E, 1024, 2048]
    down_w = layer_weights["mlp.experts.down_proj"].contiguous()        # [E, 2048, 512]

    # Run native kernel (routed experts only)
    routed_out = ext.moe_output_owned_bf16(
        x, expert_ids_i32, routing_w, gate_up_w, down_w
    )

    # Add shared expert (same as reference — not part of native kernel contract)
    moe_out = routed_out.view(1, -1)
    h_flat = hidden_in.to(torch.bfloat16)

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


def moe_forward_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Routed-only version for isolated kernel testing."""
    ext = _load_native_extension()

    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
    routing_w = routing_weights.to(torch.float32).contiguous()

    gate_up_w = layer_weights["mlp.experts.gate_up_proj"].contiguous()
    down_w = layer_weights["mlp.experts.down_proj"].contiguous()

    routed_out = ext.moe_output_owned_bf16(
        x, expert_ids_i32, routing_w, gate_up_w, down_w
    )
    return routed_out.view(1, -1)


def benchmark_kernel(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
    warmup: int = 10,
    iters: int = 100,
) -> float:
    """Benchmark native kernel latency (ms). Routed experts only."""
    ext = _load_native_extension()

    x = hidden_in.view(-1).contiguous().to(torch.bfloat16)
    expert_ids_i32 = expert_ids.to(torch.int32).contiguous()
    routing_w = routing_weights.to(torch.float32).contiguous()
    gate_up_w = layer_weights["mlp.experts.gate_up_proj"].contiguous()
    down_w = layer_weights["mlp.experts.down_proj"].contiguous()

    for _ in range(warmup):
        ext.moe_output_owned_bf16(x, expert_ids_i32, routing_w, gate_up_w, down_w)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.moe_output_owned_bf16(x, expert_ids_i32, routing_w, gate_up_w, down_w)
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> int:
    """Standalone: run candidate on all fixtures, report metrics + latency."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file
    from engine.loader import load_qwen36_layer

    fixtures_path = Path(args.fixtures)
    with open(fixtures_path / "manifest.json") as f:
        manifest = json.load(f)

    print(f"[candidate] Native output-owned BF16 MoE")
    print(f"[candidate] Fixtures: {fixtures_path} ({manifest['num_fixtures']})")
    print(f"[candidate] Model: {args.model_dir}")

    # Build extension
    print(f"[candidate] Building native CUDA extension...")
    ext = _load_native_extension()
    print(f"[candidate] Extension loaded OK")

    results = []
    needed_layers = sorted(set(e["layer_id"] for e in manifest["fixtures"]))

    for layer_id in needed_layers:
        print(f"\n[candidate] Loading layer {layer_id}...")
        w, _ = load_qwen36_layer(
            args.model_dir, layer_id,
            num_experts=manifest["num_experts"],
            device=args.device, dequant_dtype=torch.bfloat16,
        )

        layer_fixtures = [e for e in manifest["fixtures"] if e["layer_id"] == layer_id]
        for entry in layer_fixtures:
            fixture_data = load_file(str(fixtures_path / entry["fixture_file"]), device=args.device)
            hidden_in = fixture_data["hidden_in"].to(torch.bfloat16)
            expert_ids = fixture_data["expert_ids"]
            routing_weights = fixture_data["routing_weights"]
            expected = fixture_data["moe_output"].to(torch.bfloat16)

            # Run candidate
            candidate_out = moe_forward_fixture(hidden_in, expert_ids, routing_weights, w, {})

            # Compute metrics
            rf = expected.float().flatten()
            cf = candidate_out.float().flatten()
            diff = rf - cf
            max_abs = float(diff.abs().max())
            mean_abs = float(diff.abs().mean())
            cosine = float(torch.dot(rf, cf) / (
                torch.linalg.vector_norm(rf).clamp_min(1e-12) *
                torch.linalg.vector_norm(cf).clamp_min(1e-12)
            ))

            # Benchmark
            latency_ms = benchmark_kernel(hidden_in, expert_ids, routing_weights, w,
                                          warmup=args.warmup, iters=args.iters)

            result = {
                "fixture": entry["fixture_file"],
                "layer_id": layer_id,
                "prompt_id": entry["prompt_id"],
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "cosine": cosine,
                "exact": 1 if max_abs == 0.0 else 0,
                "candidate_ms": latency_ms,
            }
            results.append(result)

            status = "GREEN" if max_abs == 0.0 else ("AMBER" if max_abs < 1e-3 else "RED")
            print(
                f"  L{layer_id:02d}/P{entry['prompt_id']:02d}: "
                f"max_abs={max_abs:.2e} cos={cosine:.8f} "
                f"latency={latency_ms:.4f}ms {status}"
            )

        del w
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    all_exact = all(r["exact"] == 1 for r in results)
    avg_latency = sum(r["candidate_ms"] for r in results) / len(results) if results else 0
    max_latency = max(r["candidate_ms"] for r in results) if results else 0

    print(f"\n{'='*60}")
    print(f"CANDIDATE SUMMARY: native_output_owned_bf16")
    print(f"  Fixtures:    {len(results)}")
    print(f"  All exact:   {'YES' if all_exact else 'NO'}")
    print(f"  Avg latency: {avg_latency:.4f} ms")
    print(f"  Max latency: {max_latency:.4f} ms")
    print(f"  Triton ref:  0.059 ms (target)")
    if avg_latency < 0.059:
        print(f"  VERDICT: FASTER than Triton — proceed to integration")
    elif avg_latency < 0.075:
        print(f"  VERDICT: AMBER — marginal, research only")
    else:
        print(f"  VERDICT: SLOWER — close this approach")
    print(f"{'='*60}")

    # Write report
    out_path = args.out or str(fixtures_path / "native_output_owned_bf16_report.json")
    report = {
        "candidate": "native_output_owned_bf16",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "all_exact": all_exact,
        "avg_latency_ms": avg_latency,
        "max_latency_ms": max_latency,
        "triton_baseline_ms": 0.059,
        "results": results,
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[candidate] Report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
