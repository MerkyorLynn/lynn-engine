#!/usr/bin/env python3
"""P141 · Packed NVFP4 slot MoE stage diagnostics.

Isolates which stage introduces drift vs the slot-order PyTorch reference:
  A. gate_up → inter[8,512]: native vs Python F.linear dequant
  B. down → [8,2048]: native down_out vs Python F.linear on same inter
  C. route reduce: native weighted sum vs Python sum

Uses p138 packed fixtures + p135 slotorder BF16 reference.
Also runs a pure-Python dequant path to establish the "achievable ceiling"
for packed NVFP4 → slot-order BF16 comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _e2m1_dequant_tensor(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Python reference: dequant packed NVFP4 to BF16 using E2M1 table.

    packed: [..., K/2] uint8  (two nibbles per byte)
    scale: [..., K/16] fp16  (per-group-of-16 scale)
    global_scale: scalar fp16

    Returns: [..., K] BF16
    """
    E2M1_TABLE = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=packed.device)

    # Unpack nibbles: low nibble first, then high
    low = (packed & 0x0F).to(torch.int32)
    high = ((packed >> 4) & 0x0F).to(torch.int32)
    # Interleave: [... , K/2] → [..., K] with low at even, high at odd
    shape = list(packed.shape)
    K_half = shape[-1]
    K = K_half * 2

    # mag + sign decode
    low_mag = low & 0x07
    low_sign = ((low >> 3) & 1).float()
    high_mag = high & 0x07
    high_sign = ((high >> 3) & 1).float()

    low_val = E2M1_TABLE[low_mag] * (1 - 2 * low_sign)
    high_val = E2M1_TABLE[high_mag] * (1 - 2 * high_sign)

    # Interleave to [..., K]
    result = torch.zeros(*shape[:-1], K, dtype=torch.float32, device=packed.device)
    result[..., 0::2] = low_val
    result[..., 1::2] = high_val

    # Apply per-16-group scale
    # scale shape: [..., K/16]
    groups = K // 16
    inv_global = 1.0 / global_scale.float().item()
    # Expand scale to match elements
    scale_expanded = scale.float().unsqueeze(-1).expand(*scale.shape, 16).reshape(*shape[:-1], K)
    result = result * scale_expanded * inv_global

    return result.to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-fixtures", required=True)
    ap.add_argument("--ref-fixtures", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file

    packed_dir = Path(args.packed_fixtures)
    ref_dir = Path(args.ref_fixtures)

    with open(packed_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"[p141] Stage diagnostics: packed NVFP4 vs slot-order PyTorch")
    print(f"[p141] Packed: {packed_dir}")
    print(f"[p141] Ref: {ref_dir}\n")

    # Load native extension for kernel comparison
    from engine.native_cuda import load_lynn_native_extension
    ext = load_lynn_native_extension(verbose=False)
    has_native = hasattr(ext, "moe_slot_packed_nvfp4_probe")
    print(f"[p141] Native kernel: {'OK' if has_native else 'MISSING'}\n")

    results = []
    for entry in manifest["fixtures"]:
        packed_file = entry["fixture_file"]
        layer_id = entry["layer_id"]
        prompt_id = entry["prompt_id"]

        pd = load_file(str(packed_dir / packed_file), device="cuda")
        x = pd["hidden_in"].to(torch.bfloat16).view(1, 2048)
        rw = pd["routing_weights"]
        gu_packed = pd["slot_gate_up_packed"]       # [8, 1024, 1024] uint8
        gu_scale = pd["slot_gate_up_scale"]         # [8, 1024, 128] fp16
        gu_global = pd["slot_gate_up_global_scale"] # scalar fp16
        d_packed = pd["slot_down_packed"]           # [8, 2048, 256] uint8
        d_scale = pd["slot_down_scale"]             # [8, 2048, 32] fp16
        d_global = pd["slot_down_global_scale"]     # scalar fp16

        # Load BF16 reference
        ref_file = f"layer_{layer_id:02d}_prompt_{prompt_id:02d}_slots.safetensors"
        ref_path = ref_dir / ref_file
        if not ref_path.exists():
            continue
        rd = load_file(str(ref_path), device="cuda")
        ref_gate_up_w = rd["slot_gate_up_weight"].to(torch.bfloat16)  # [8, 1024, 2048]
        ref_down_w = rd["slot_down_weight"].to(torch.bfloat16)        # [8, 2048, 512]
        ref_routed = rd["routed_output"].to(torch.bfloat16)           # [1, 2048]

        # ═══ Stage A: gate_up dequant ═══
        # Python dequant: unpack → BF16 weight → F.linear
        dequant_gu = _e2m1_dequant_tensor(gu_packed, gu_scale, gu_global)  # [8, 1024, 2048]

        # Compare dequant vs BF16 reference weights
        dequant_vs_ref_w = (dequant_gu.float() - ref_gate_up_w.float()).abs().max().item()

        # Python slot-order inter using dequant weights
        inter_dequant = torch.zeros(8, 1, 512, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            gate_up = F.linear(x, dequant_gu[k])  # [1, 1024]
            gate, up = gate_up.chunk(2, dim=1)
            inter_dequant[k] = F.silu(gate) * up

        # Python slot-order inter using BF16 ref weights
        inter_ref = torch.zeros(8, 1, 512, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            gate_up = F.linear(x, ref_gate_up_w[k])
            gate, up = gate_up.chunk(2, dim=1)
            inter_ref[k] = F.silu(gate) * up

        inter_dequant_vs_ref = (inter_dequant.float() - inter_ref.float()).abs().max().item()

        # ═══ Stage B: down projection ═══
        # Dequant down weights
        dequant_down = _e2m1_dequant_tensor(d_packed, d_scale, d_global)  # [8, 2048, 512]
        dequant_vs_ref_down_w = (dequant_down.float() - ref_down_w.float()).abs().max().item()

        # Down using dequant weights + dequant inter
        down_dequant = torch.zeros(8, 1, 2048, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            down_dequant[k] = F.linear(inter_dequant[k], dequant_down[k])

        # Down using ref weights + ref inter (the gold standard)
        down_ref = torch.zeros(8, 1, 2048, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            down_ref[k] = F.linear(inter_ref[k], ref_down_w[k])

        down_dequant_vs_ref = (down_dequant.float() - down_ref.float()).abs().max().item()

        # ═══ Stage C: route reduce ═══
        # Full dequant path
        out_dequant = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            out_dequant += down_dequant[k] * rw[k].to(torch.bfloat16)

        # Full ref path
        out_ref = torch.zeros(1, 2048, device="cuda", dtype=torch.bfloat16)
        for k in range(8):
            out_ref += down_ref[k] * rw[k].to(torch.bfloat16)

        full_dequant_vs_ref = (out_dequant.float() - ref_routed.float()).abs().max().item()
        full_ref_vs_stored = (out_ref.float() - ref_routed.float()).abs().max().item()

        # ═══ Native kernel vs dequant Python ═══
        native_vs_dequant = None
        if has_native:
            native_out = ext.moe_slot_packed_nvfp4_probe(
                x.view(-1).contiguous(),
                rw.float().contiguous(),
                gu_packed.contiguous(),
                gu_scale.to(torch.float16).contiguous(),
                gu_global.to(torch.float16).contiguous(),
                d_packed.contiguous(),
                d_scale.to(torch.float16).contiguous(),
                d_global.to(torch.float16).contiguous(),
            )
            native_vs_dequant = (native_out.view(1, -1).float() - out_dequant.float()).abs().max().item()

        result = {
            "layer_id": layer_id,
            "prompt_id": prompt_id,
            "dequant_vs_ref_gate_up_w": dequant_vs_ref_w,
            "inter_dequant_vs_inter_ref": inter_dequant_vs_ref,
            "dequant_vs_ref_down_w": dequant_vs_ref_down_w,
            "down_dequant_vs_down_ref": down_dequant_vs_ref,
            "full_dequant_vs_stored_ref": full_dequant_vs_ref,
            "full_slotorder_ref_vs_stored": full_ref_vs_stored,
            "native_vs_dequant_python": native_vs_dequant,
        }
        results.append(result)

        print(f"  L{layer_id:02d}/P{prompt_id:02d}:")
        print(f"    gate_up weight dequant err:  {dequant_vs_ref_w:.2e}")
        print(f"    inter (dequant vs ref):      {inter_dequant_vs_ref:.2e}")
        print(f"    down weight dequant err:     {dequant_vs_ref_down_w:.2e}")
        print(f"    down_out (dequant vs ref):   {down_dequant_vs_ref:.2e}")
        print(f"    full (dequant→routed):       {full_dequant_vs_ref:.2e}")
        print(f"    full (slotorder ref→stored): {full_ref_vs_stored:.2e}")
        if native_vs_dequant is not None:
            print(f"    native vs dequant Python:    {native_vs_dequant:.2e}")
        print()

    # Summary
    print(f"{'='*70}")
    print(f"STAGE DIAGNOSTICS SUMMARY")
    if results:
        print(f"  gate_up_w dequant max:     {max(r['dequant_vs_ref_gate_up_w'] for r in results):.2e}")
        print(f"  inter dequant-vs-ref max:  {max(r['inter_dequant_vs_inter_ref'] for r in results):.2e}")
        print(f"  down_w dequant max:        {max(r['dequant_vs_ref_down_w'] for r in results):.2e}")
        print(f"  down_out dequant-vs-ref:   {max(r['down_dequant_vs_down_ref'] for r in results):.2e}")
        print(f"  full dequant→stored:       {max(r['full_dequant_vs_stored_ref'] for r in results):.2e}")
        print(f"  slotorder ref→stored:      {max(r['full_slotorder_ref_vs_stored'] for r in results):.2e}")
        if results[0]["native_vs_dequant_python"] is not None:
            print(f"  native vs dequant Python:  {max(r['native_vs_dequant_python'] for r in results):.2e}")
    print(f"{'='*70}")

    out_path = args.out or str(packed_dir / "p141_stage_diagnostics.json")
    report = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[p141] Report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
