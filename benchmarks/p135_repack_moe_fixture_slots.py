#!/usr/bin/env python3
"""P135 · Fixture-local MoE slot repack.

Purpose: Read p133-exported fixtures and repack each fixture's top-8 expert
weights into slot-ordered tensors.  This eliminates dynamic gather and
full expert-table lookup for candidate kernels: the kernel receives
`slot_gate_up_weight` and `slot_down_weight` directly in dispatch order.

Per-fixture output (safetensors):
  - hidden_in:            [1, hidden] BF16
  - expert_ids:           [8] int32
  - routing_weights:      [8] float32
  - slot_gate_up_weight:  [8, 2*intermediate, hidden] BF16 (F.linear layout)
  - slot_down_weight:     [8, hidden, intermediate] BF16 (F.linear layout)
  - routed_output:        [1, hidden] BF16 — ground truth routed-only output

Metadata manifest records source fixture sha256, layer_id, prompt_id,
shapes, dtype, and timing.

Usage:
  python benchmarks/p135_repack_moe_fixture_slots.py \
    --fixtures reports/qwen36_35b/p133_fixtures \
    --model-dir /path/to/Qwen3.6-35B-A3B-BF16 \
    --out reports/qwen36_35b/p135_repacked_fixtures \
    --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.loader import load_qwen36_layer  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Reference routed-only MoE (matches p134 _moe_reference_routed_only)
# ─────────────────────────────────────────────────────────────

def _moe_reference_routed_only(
    hidden_in: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, Any],
) -> torch.Tensor:
    """Run ONLY the routed expert path (no shared expert)."""
    K = expert_ids.shape[0]
    h_flat = hidden_in  # [1, hidden]

    active_experts = torch.unique(expert_ids).tolist()
    moe_out = torch.zeros_like(h_flat)

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

    return moe_out


# ─────────────────────────────────────────────────────────────
# Slot weight extraction
# ─────────────────────────────────────────────────────────────

def _extract_slot_weights(
    expert_ids: torch.Tensor,
    layer_weights: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract top-K expert weights in slot order.

    Returns:
        slot_gate_up: [K, 2*intermediate, hidden]
        slot_down:    [K, hidden, intermediate]
    """
    ids = expert_ids.tolist()

    if "mlp.experts.gate_up_proj" in layer_weights and "mlp.experts.down_proj" in layer_weights:
        gate_up_full = layer_weights["mlp.experts.gate_up_proj"]
        down_full = layer_weights["mlp.experts.down_proj"]
        slot_gate_up = torch.stack([gate_up_full[e] for e in ids])
        slot_down = torch.stack([down_full[e] for e in ids])
    else:
        slot_gate_up_list = []
        slot_down_list = []
        for e in ids:
            gate = layer_weights[f"mlp.experts.{e}.gate_proj.weight"]
            up = layer_weights[f"mlp.experts.{e}.up_proj.weight"]
            slot_gate_up_list.append(torch.cat([gate, up], dim=0))
            slot_down_list.append(layer_weights[f"mlp.experts.{e}.down_proj.weight"])
        slot_gate_up = torch.stack(slot_gate_up_list)
        slot_down = torch.stack(slot_down_list)

    return slot_gate_up, slot_down


# ─────────────────────────────────────────────────────────────
# Main repack pipeline
# ─────────────────────────────────────────────────────────────

def repack_fixtures(
    fixtures_dir: str,
    model_dir: str,
    out_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Repack all fixtures in fixtures_dir into slot-ordered tensors."""
    from safetensors.torch import load_file, save_file

    fixtures_path = Path(fixtures_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = fixtures_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"p133 manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        src_manifest = json.load(f)

    print(f"[p135] Source fixtures: {fixtures_path}")
    print(f"[p135] Model dir: {model_dir}")
    print(f"[p135] Output: {out_path}")
    print(f"[p135] Fixtures count: {src_manifest['num_fixtures']}")

    # Determine layers needed
    needed_layers = sorted({entry["layer_id"] for entry in src_manifest["fixtures"]})
    print(f"[p135] Layers needed: {needed_layers}")

    # Cache layer weights
    layer_cache: dict[int, dict[str, Any]] = {}
    for layer_id in needed_layers:
        print(f"[p135] Loading layer {layer_id} weights...", flush=True)
        w, _ = load_qwen36_layer(
            model_dir,
            layer_id,
            num_experts=src_manifest.get("num_experts", 256),
            device=device,
            dequant_dtype=dtype,
        )
        layer_cache[layer_id] = w

    manifest_entries = []
    repack_t0 = time.time()

    for entry in src_manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        src_fixture_path = fixtures_path / fixture_file
        fixture_data = load_file(str(src_fixture_path), device=device)

        hidden_in = fixture_data["hidden_in"].to(dtype)
        expert_ids = fixture_data["expert_ids"]
        routing_weights = fixture_data["routing_weights"]
        layer_weights = layer_cache[layer_id]

        # Compute reference routed-only output
        t_ref = time.time()
        routed_output = _moe_reference_routed_only(
            hidden_in, expert_ids, routing_weights, layer_weights
        )
        ref_ms = (time.time() - t_ref) * 1000.0

        # Extract slot weights
        t_extract = time.time()
        slot_gate_up, slot_down = _extract_slot_weights(expert_ids, layer_weights)
        extract_ms = (time.time() - t_extract) * 1000.0

        # Source file sha256
        sha256 = hashlib.sha256(src_fixture_path.read_bytes()).hexdigest()

        # Build repacked safetensors
        repacked_name = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slots.safetensors"
        repacked_path = out_path / repacked_name

        tensors_to_save = {
            "hidden_in": hidden_in.contiguous().cpu(),
            "expert_ids": expert_ids.contiguous().cpu(),
            "routing_weights": routing_weights.contiguous().cpu(),
            "slot_gate_up_weight": slot_gate_up.contiguous().cpu(),
            "slot_down_weight": slot_down.contiguous().cpu(),
            "routed_output": routed_output.contiguous().cpu(),
        }
        save_file(tensors_to_save, str(repacked_path))

        entry_meta = {
            "fixture_file": repacked_name,
            "source_fixture": fixture_file,
            "source_fixture_sha256": sha256,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "expert_ids": expert_ids.tolist(),
            "routing_weights": [round(float(x), 6) for x in routing_weights.tolist()],
            "shapes": {
                k: list(v.shape) for k, v in tensors_to_save.items()
            },
            "dtype": str(dtype),
            "ref_compute_ms": round(ref_ms, 4),
            "extract_ms": round(extract_ms, 4),
        }
        manifest_entries.append(entry_meta)

        print(
            f"  [p135] L{layer_id:02d}/P{prompt_id:02d} -> {repacked_name} "
            f"ref={ref_ms:.3f}ms extract={extract_ms:.3f}ms",
            flush=True,
        )

    total_time = time.time() - repack_t0

    manifest = {
        "schema": "lynn-moe-slot-repack-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_fixtures_dir": str(fixtures_path),
        "source_schema": src_manifest.get("schema", "unknown"),
        "model_dir": str(model_dir),
        "device": device,
        "dtype": str(dtype),
        "num_fixtures": len(manifest_entries),
        "total_repack_seconds": total_time,
        "fixtures": manifest_entries,
    }
    out_manifest_path = out_path / "manifest.json"
    out_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[p135] Manifest written: {out_manifest_path}")
    print(f"[p135] {len(manifest_entries)} fixtures repacked in {total_time:.1f}s")

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Repack MoE fixtures into slot-ordered expert weights.")
    ap.add_argument("--fixtures", required=True, help="Path to p133 fixture directory.")
    ap.add_argument("--model-dir", required=True, help="Path to model directory for loading layer weights.")
    ap.add_argument("--out", default="reports/qwen36_35b/p135_repacked_fixtures", help="Output directory.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    manifest = repack_fixtures(
        fixtures_dir=args.fixtures,
        model_dir=args.model_dir,
        out_dir=args.out,
        device=args.device,
        dtype=dtype,
    )

    print(f"\n{'='*60}")
    print(f"P135 DONE: {manifest['num_fixtures']} fixtures repacked")
    print(f"  Output: {args.out}/")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
