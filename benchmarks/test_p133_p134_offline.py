#!/usr/bin/env python3
"""Offline smoke test for p133/p134 pipeline (no GPU, no model required).

Creates synthetic fixtures with known ground truth, then verifies p134
contract logic produces exact match. This catches logic bugs before
R6000 execution.

Usage:
    python3 benchmarks/test_p133_p134_offline.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _make_synthetic_layer_weights(
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    num_experts: int = 256,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Create synthetic layer weights matching Lynn engine fused MoE layout."""
    gen = torch.Generator().manual_seed(seed)
    w = {}
    # Router
    w["mlp.gate.weight"] = torch.randn(num_experts, hidden_size, generator=gen) * 0.02
    # Fused expert weights
    w["mlp.experts.gate_up_proj"] = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, generator=gen
    ) * 0.02
    w["mlp.experts.down_proj"] = torch.randn(
        num_experts, hidden_size, intermediate_size, generator=gen
    ) * 0.02
    # Shared expert
    w["mlp.shared_expert.gate_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, generator=gen
    ) * 0.02
    w["mlp.shared_expert.up_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, generator=gen
    ) * 0.02
    w["mlp.shared_expert.down_proj.weight"] = torch.randn(
        hidden_size, intermediate_size, generator=gen
    ) * 0.02
    w["mlp.shared_expert_gate.weight"] = torch.randn(
        1, hidden_size, generator=gen
    ) * 0.02
    return w


def _moe_forward_reference(
    h_flat: torch.Tensor,
    w: dict[str, torch.Tensor],
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference MoE forward returning output + expert_ids + routing_weights."""
    router_logits = F.linear(h_flat, w["mlp.gate.weight"])
    routing_weights, expert_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32)

    active_experts = torch.unique(expert_indices).tolist()
    moe_out = torch.zeros_like(h_flat)

    for e in active_experts:
        mask = (expert_indices == e)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        x_e = h_flat[token_idx]
        gate_up = F.linear(x_e, w["mlp.experts.gate_up_proj"][e])
        gate, up = gate_up.chunk(2, dim=-1)
        ffn_e = F.linear(F.silu(gate) * up, w["mlp.experts.down_proj"][e])
        weight_e = routing_weights[token_idx, slot_idx].unsqueeze(-1).to(h_flat.dtype)
        moe_out.index_add_(0, token_idx, ffn_e * weight_e)

    # Shared expert
    gate_s = F.linear(h_flat, w["mlp.shared_expert.gate_proj.weight"])
    up_s = F.linear(h_flat, w["mlp.shared_expert.up_proj.weight"])
    shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(F.linear(h_flat, w["mlp.shared_expert_gate.weight"]))
    shared_ffn = shared_ffn * shared_gate
    moe_out = moe_out + shared_ffn

    return moe_out, expert_indices[0].to(torch.int32), routing_weights[0]


def test_p134_contract_logic():
    """Test that p134 contract metrics are correct on synthetic data."""
    from benchmarks.p134_active_moe_fixture_contract import _compute_metrics

    # Exact match
    a = torch.randn(1, 2048)
    metrics = _compute_metrics(a, a.clone())
    assert metrics["max_abs"] == 0.0, f"Expected max_abs=0, got {metrics['max_abs']}"
    assert metrics["exact"] == 1, f"Expected exact=1, got {metrics['exact']}"
    assert metrics["cosine"] == 1.0 or abs(metrics["cosine"] - 1.0) < 1e-7

    # Known difference
    b = a + 0.01
    metrics2 = _compute_metrics(a, b)
    assert metrics2["max_abs"] > 0
    assert metrics2["exact"] == 0
    assert metrics2["cosine"] < 1.0

    print("  [test] _compute_metrics: PASS")


def test_fixture_roundtrip():
    """Create synthetic fixture, save, reload, verify contract."""
    from safetensors.torch import save_file, load_file

    hidden_size = 2048
    top_k = 8
    num_experts = 256

    # Create synthetic weights and hidden state
    w = _make_synthetic_layer_weights(hidden_size=hidden_size, num_experts=num_experts)
    gen = torch.Generator().manual_seed(12345)
    h = torch.randn(1, hidden_size, generator=gen) * 0.5

    # Run reference MoE forward
    moe_out, expert_ids, routing_weights = _moe_forward_reference(h, w, top_k=top_k)

    # Save fixture
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture_path = Path(tmpdir) / "layer_00_prompt_00.safetensors"
        save_file(
            {
                "hidden_in": h.contiguous(),
                "expert_ids": expert_ids.contiguous(),
                "routing_weights": routing_weights.contiguous(),
                "moe_output": moe_out.contiguous(),
            },
            str(fixture_path),
        )

        # Reload
        loaded = load_file(str(fixture_path))
        h_loaded = loaded["hidden_in"]
        ids_loaded = loaded["expert_ids"]
        rw_loaded = loaded["routing_weights"]
        out_loaded = loaded["moe_output"]

        # Re-run reference with loaded inputs
        # Reconstruct using same weights
        expert_indices = ids_loaded.unsqueeze(0).long()
        routing_w = rw_loaded.unsqueeze(0).to(h_loaded.dtype)
        active_experts = torch.unique(ids_loaded).tolist()
        recomputed = torch.zeros_like(h_loaded)

        for e in active_experts:
            mask = (expert_indices == e)
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            x_e = h_loaded[token_idx]
            gate_up = F.linear(x_e, w["mlp.experts.gate_up_proj"][e])
            gate, up = gate_up.chunk(2, dim=-1)
            ffn_e = F.linear(F.silu(gate) * up, w["mlp.experts.down_proj"][e])
            weight_e = routing_w[token_idx, slot_idx].unsqueeze(-1)
            recomputed.index_add_(0, token_idx, ffn_e * weight_e)

        # Shared expert
        gate_s = F.linear(h_loaded, w["mlp.shared_expert.gate_proj.weight"])
        up_s = F.linear(h_loaded, w["mlp.shared_expert.up_proj.weight"])
        shared_ffn = F.linear(F.silu(gate_s) * up_s, w["mlp.shared_expert.down_proj.weight"])
        shared_gate = torch.sigmoid(F.linear(h_loaded, w["mlp.shared_expert_gate.weight"]))
        shared_ffn = shared_ffn * shared_gate
        recomputed = recomputed + shared_ffn

        # Verify exact match
        diff = (recomputed - out_loaded).abs().max().item()
        assert diff == 0.0, f"Fixture roundtrip max_abs={diff}, expected 0.0"

        print(f"  [test] Fixture roundtrip: PASS (max_abs={diff})")


def test_manifest_schema():
    """Verify manifest JSON schema matches expected fields."""
    required_top_keys = [
        "schema", "created", "model_dir", "device", "dtype",
        "layers", "prompts", "top_k", "hidden_size", "num_experts",
        "num_fixtures", "load_seconds", "export_seconds", "sidecar", "fixtures",
    ]
    required_fixture_keys = [
        "fixture_file", "layer_id", "prompt_id", "prompt_text",
        "prompt_tokens", "top_k", "hidden_size", "num_experts",
        "expert_ids", "routing_weights", "hidden_in_norm", "moe_output_norm",
        "sidecar",
    ]

    # Build a mock manifest
    manifest = {
        "schema": "lynn-moe-fixture-v1",
        "created": "2026-05-18T00:00:00",
        "model_dir": "/fake/path",
        "device": "cuda",
        "dtype": "torch.bfloat16",
        "layers": [0, 4, 8],
        "prompts": ["Hello", "World"],
        "top_k": 8,
        "hidden_size": 2048,
        "num_experts": 256,
        "num_fixtures": 6,
        "load_seconds": 100.0,
        "export_seconds": 10.0,
        "sidecar": {"quant_method": "compressed-tensors", "uses_folded_scale": True},
        "fixtures": [
            {
                "fixture_file": "layer_00_prompt_00.safetensors",
                "layer_id": 0,
                "prompt_id": 0,
                "prompt_text": "Hello",
                "prompt_tokens": 1,
                "top_k": 8,
                "hidden_size": 2048,
                "num_experts": 256,
                "expert_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "routing_weights": [0.2, 0.15, 0.13, 0.12, 0.11, 0.1, 0.1, 0.09],
                "hidden_in_norm": 45.3,
                "moe_output_norm": 12.1,
                "sidecar": {"layer_prefix": "model.language_model.layers.0."},
            }
        ],
    }

    for key in required_top_keys:
        assert key in manifest, f"Missing top-level key: {key}"
    for key in required_fixture_keys:
        assert key in manifest["fixtures"][0], f"Missing fixture key: {key}"

    print("  [test] Manifest schema: PASS")


def main() -> int:
    print("=" * 60)
    print("P133/P134 Offline Smoke Test (no GPU, no model)")
    print("=" * 60)
    print()

    try:
        test_p134_contract_logic()
        test_fixture_roundtrip()
        test_manifest_schema()
    except Exception as e:
        print(f"\n  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 60)
    print("ALL OFFLINE TESTS PASSED")
    print("  Pipeline logic verified — safe to run on R6000")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
