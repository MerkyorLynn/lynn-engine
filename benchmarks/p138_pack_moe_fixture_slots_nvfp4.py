#!/usr/bin/env python3
"""P138 · Pack MoE fixture slots into NVFP4 packed weights.

Purpose: Read p133-exported fixtures and repack each fixture's top-8 expert
weights into slot-ordered NVFP4 packed tensors.  This keeps the fixture
size small (~15.6 MB vs ~49 MB BF16) while preserving exact dequant
round-trip fidelity.

Per-fixture output (safetensors):
  - hidden_in:                    [1, hidden] BF16
  - expert_ids:                   [8] int32
  - routing_weights:              [8] float32
  - slot_gate_up_packed:          [8, 1024, 1024] uint8
  - slot_gate_up_scale:           [8, 1024, 128] FP16
  - slot_gate_up_global_scale:    scalar FP16
  - slot_down_packed:             [8, 2048, 256] uint8
  - slot_down_scale:              [8, 2048, 32] FP16
  - slot_down_global_scale:       scalar FP16

Optional --compress writes .safetensors.gz (~12-14 MB).

Metadata manifest records source fixture sha256, layer_id, prompt_id,
shapes, dtypes, packed bytes, and BF16-equivalent bytes.

Usage:
  python benchmarks/p138_pack_moe_fixture_slots_nvfp4.py \
    --fixtures reports/qwen36_35b/p133_fixtures_official_w4a16 \
    --model-dir /path/to/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-v0 \
    --out reports/qwen36_35b/p138_packed_slot_fixtures \
    --device cuda
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import load_grouped_nvfp4_weight  # noqa: E402


def _load_p133_fixture_data(
    fixtures_path: Path,
    fixture_file: str,
    device: str,
) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(fixtures_path / fixture_file), device=device)


def _extract_slot_packed_weights(
    expert_ids: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract top-K expert packed weights in slot order.

    Scale tensors are downcast from fp32 (loader upcast) back to fp16,
    which is lossless because the original checkpoint stores them as fp16.
    """
    ids = expert_ids.long()
    return {
        "slot_gate_up_packed": gate_up_packed[ids].contiguous(),
        "slot_gate_up_scale": gate_up_scale[ids].half().contiguous(),
        "slot_gate_up_global_scale": gate_up_global_scale.half(),
        "slot_down_packed": down_packed[ids].contiguous(),
        "slot_down_scale": down_scale[ids].half().contiguous(),
        "slot_down_global_scale": down_global_scale.half(),
    }


def pack_fixtures(
    fixtures_dir: str,
    model_dir: str,
    out_dir: str,
    device: str = "cuda",
    compress: bool = False,
) -> dict[str, Any]:
    """Pack all fixtures into slot-ordered NVFP4 tensors."""
    from safetensors.torch import load_file, save_file, save as save_safetensors_buffer

    fixtures_path = Path(fixtures_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_path = fixtures_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"p133 manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        src_manifest = json.load(f)

    print(f"[p138] Source fixtures: {fixtures_path}")
    print(f"[p138] Model dir: {model_dir}")
    print(f"[p138] Output: {out_path}")
    print(f"[p138] Fixtures count: {src_manifest['num_fixtures']}")

    # Determine layers needed
    needed_layers = sorted({entry["layer_id"] for entry in src_manifest["fixtures"]})
    print(f"[p138] Layers needed: {needed_layers}")

    # Cache layer packed weights
    layer_cache: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in needed_layers:
        print(f"[p138] Loading layer {layer_id} packed weights...", flush=True)
        gup, gus, gugs = load_grouped_nvfp4_weight(
            model_dir,
            f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj",
            device=device,
        )
        dp, ds, dgs = load_grouped_nvfp4_weight(
            model_dir,
            f"model.language_model.layers.{layer_id}.mlp.experts.down_proj",
            device=device,
        )
        layer_cache[layer_id] = {
            "gate_up_packed": gup,
            "gate_up_scale": gus,
            "gate_up_global_scale": gugs,
            "down_packed": dp,
            "down_scale": ds,
            "down_global_scale": dgs,
        }

    manifest_entries = []
    pack_t0 = time.time()

    for entry in src_manifest["fixtures"]:
        fixture_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        src_fixture_path = fixtures_path / fixture_file
        fixture_data = _load_p133_fixture_data(fixtures_path, fixture_file, device)

        hidden_in = fixture_data["hidden_in"]
        expert_ids = fixture_data["expert_ids"]
        routing_weights = fixture_data["routing_weights"]
        layer_weights = layer_cache[layer_id]

        # Extract slot packed weights
        t_extract = time.time()
        slot_weights = _extract_slot_packed_weights(
            expert_ids,
            layer_weights["gate_up_packed"],
            layer_weights["gate_up_scale"],
            layer_weights["gate_up_global_scale"],
            layer_weights["down_packed"],
            layer_weights["down_scale"],
            layer_weights["down_global_scale"],
        )
        extract_ms = (time.time() - t_extract) * 1000.0

        # Source file sha256
        sha256 = hashlib.sha256(src_fixture_path.read_bytes()).hexdigest()

        # Build packed safetensors
        repacked_name = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slot_packed.safetensors"
        if compress:
            repacked_name += ".gz"
        repacked_path = out_path / repacked_name

        tensors_to_save = {
            "hidden_in": hidden_in.contiguous().cpu(),
            "expert_ids": expert_ids.contiguous().cpu(),
            "routing_weights": routing_weights.contiguous().cpu(),
            **{k: v.contiguous().cpu() for k, v in slot_weights.items()},
        }

        # Save (optionally gzip)
        buf = save_safetensors_buffer(tensors_to_save)
        if compress:
            with gzip.open(str(repacked_path), "wb", compresslevel=6) as f:
                f.write(buf)
        else:
            repacked_path.write_bytes(buf)

        packed_bytes = repacked_path.stat().st_size
        bf16_equiv = (
            8 * 1024 * 2048 * 2  # gate_up
            + 8 * 2048 * 512 * 2  # down
        )

        entry_meta = {
            "fixture_file": repacked_name,
            "source_fixture": fixture_file,
            "source_fixture_sha256": sha256,
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "expert_ids": expert_ids.tolist(),
            "routing_weights": [round(float(x), 6) for x in routing_weights.tolist()],
            "shapes": {k: list(v.shape) for k, v in tensors_to_save.items()},
            "dtypes": {k: str(v.dtype) for k, v in tensors_to_save.items()},
            "packed_bytes": packed_bytes,
            "bf16_equiv_bytes": bf16_equiv,
            "compress": compress,
            "extract_ms": round(extract_ms, 4),
        }
        manifest_entries.append(entry_meta)

        print(
            f"  [p138] L{layer_id:02d}/P{prompt_id:02d} -> {repacked_name} "
            f"{packed_bytes / 1e6:.2f} MB (bf16={bf16_equiv / 1e6:.2f} MB) "
            f"extract={extract_ms:.3f}ms",
            flush=True,
        )

    total_time = time.time() - pack_t0

    manifest = {
        "schema": "lynn-moe-slot-packed-nvfp4-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_fixtures_dir": str(fixtures_path),
        "source_schema": src_manifest.get("schema", "unknown"),
        "model_dir": str(model_dir),
        "device": device,
        "num_fixtures": len(manifest_entries),
        "total_pack_seconds": total_time,
        "fixtures": manifest_entries,
    }
    out_manifest_path = out_path / "manifest.json"
    out_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[p138] Manifest written: {out_manifest_path}")
    print(f"[p138] {len(manifest_entries)} fixtures packed in {total_time:.1f}s")

    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Pack MoE fixtures into slot-ordered NVFP4 packed weights.")
    ap.add_argument("--fixtures", required=True, help="Path to p133 fixture directory.")
    ap.add_argument("--model-dir", required=True, help="Path to NVFP4 model directory.")
    ap.add_argument("--out", default="reports/qwen36_35b/p138_packed_slot_fixtures", help="Output directory.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compress", action="store_true", help="Gzip compress each fixture.")
    args = ap.parse_args()

    manifest = pack_fixtures(
        fixtures_dir=args.fixtures,
        model_dir=args.model_dir,
        out_dir=args.out,
        device=args.device,
        compress=args.compress,
    )

    print(f"\n{'='*60}")
    print(f"P138 DONE: {manifest['num_fixtures']} fixtures packed")
    print(f"  Output: {args.out}/")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
