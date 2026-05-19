#!/usr/bin/env python3
"""Convert official Qwen3.6 MTP FP8 safetensors to Lynn fused layout.

Reads the official `mtp.safetensors` (with per-expert FP8 E4M3 layout and
scale_inv tensors) and produces a Lynn-compatible fused safetensors that
`engine/mtp_sidecar.py:load_mtp_sidecar` can directly load.

Key transformations:
  1. Per-expert gate_proj + up_proj → fused `mtp.layers.0.mlp.experts.gate_up_proj`
     Shape: [num_experts, 2*intermediate, hidden]
  2. Per-expert down_proj → stacked `mtp.layers.0.mlp.experts.down_proj`
     Shape: [num_experts, hidden, intermediate]
  3. FP8 E4M3 tensors → dequantized to BF16 (with scale_inv applied)
     Metadata records: source_dtype=fp8_e4m3fn, conversion=fp8_to_bf16_fused
  4. Passthrough: mtp.fc, mtp.pre_fc_norm_*, mtp.norm, attention, shared expert,
     router gate, layernorms (already BF16/FP32)

CPU-only. No GPU required for conversion.

Usage:
  python scripts/qwen36_convert_official_mtp_to_lynn_fused.py \\
    --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \\
    --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \\
    --out /path/to/output/mtp_lynn_fused.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _read_header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _load_tensor_cpu(path: Path, key: str) -> torch.Tensor:
    """Load a single tensor from safetensors on CPU."""
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _dequant_fp8_with_scale(tensor: torch.Tensor, scale_inv: torch.Tensor | None) -> torch.Tensor:
    """Dequantize FP8 E4M3 tensor to BF16 using scale_inv.

    Official Qwen FP8 layout: value_bf16 = fp8_value * scale_inv
    scale_inv is per-channel (row) or scalar.
    """
    # Convert FP8 bytes to float via view
    if tensor.dtype == torch.uint8:
        fp8_vals = tensor.view(torch.float8_e4m3fn).float()
    elif tensor.dtype == torch.float8_e4m3fn:
        fp8_vals = tensor.float()
    else:
        # Already float — may be BF16/FP32, just return as BF16
        return tensor.to(torch.bfloat16)

    if scale_inv is not None:
        scale = scale_inv.float()
        if scale.dim() == 0:
            fp8_vals = fp8_vals * scale
        elif scale.dim() == 1 and scale.shape[0] == fp8_vals.shape[0]:
            fp8_vals = fp8_vals * scale.unsqueeze(-1)
        else:
            # Try broadcast
            fp8_vals = fp8_vals * scale
    return fp8_vals.to(torch.bfloat16)


def _find_scale_inv_key(key: str, header: dict[str, Any]) -> str | None:
    """Find the matching scale_inv key for a weight tensor."""
    candidates = [
        key + "_scale_inv",
        key.replace(".weight", ".weight_scale_inv"),
        key.replace(".weight", "_scale_inv"),
    ]
    for c in candidates:
        if c in header:
            return c
    # Fuzzy: look for scale_inv with same prefix
    base = key.rsplit(".", 1)[0]
    for k in header:
        if k.startswith(base) and "scale_inv" in k:
            return k
    return None


def _parse_expert_id(key: str) -> int | None:
    """Extract expert index from key like experts.42.gate_proj.weight."""
    m = re.search(r"experts?[._](\d+)", key)
    return int(m.group(1)) if m else None


def _lynn_layer_key(official_key: str) -> str | None:
    """Map official MTP key to Lynn layer key under mtp.layers.0.* prefix.

    Returns None for keys that don't map to the layer (fc, norms handled separately).
    """
    # Patterns to remap
    patterns = [
        (r"^.*\.input_layernorm\.weight$", "mtp.layers.0.input_layernorm.weight"),
        (r"^.*\.post_attention_layernorm\.weight$", "mtp.layers.0.post_attention_layernorm.weight"),
        (r"^.*self_attn\.q_proj\.weight$", "mtp.layers.0.self_attn.q_proj.weight"),
        (r"^.*self_attn\.k_proj\.weight$", "mtp.layers.0.self_attn.k_proj.weight"),
        (r"^.*self_attn\.v_proj\.weight$", "mtp.layers.0.self_attn.v_proj.weight"),
        (r"^.*self_attn\.o_proj\.weight$", "mtp.layers.0.self_attn.o_proj.weight"),
        (r"^.*self_attn\.q_norm\.weight$", "mtp.layers.0.self_attn.q_norm.weight"),
        (r"^.*self_attn\.k_norm\.weight$", "mtp.layers.0.self_attn.k_norm.weight"),
        (r"^.*mlp\.gate\.weight$", "mtp.layers.0.mlp.gate.weight"),
        (r"^.*shared_expert\.gate_proj\.weight$", "mtp.layers.0.mlp.shared_expert.gate_proj.weight"),
        (r"^.*shared_expert\.up_proj\.weight$", "mtp.layers.0.mlp.shared_expert.up_proj.weight"),
        (r"^.*shared_expert\.down_proj\.weight$", "mtp.layers.0.mlp.shared_expert.down_proj.weight"),
        (r"^.*shared_expert_gate\.weight$", "mtp.layers.0.mlp.shared_expert_gate.weight"),
    ]
    for pat, target in patterns:
        if re.match(pat, official_key):
            return target
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert official Qwen3.6 MTP FP8 → Lynn fused")
    ap.add_argument("--mtp", required=True, help="Path to official mtp.safetensors")
    ap.add_argument("--config", default=None, help="config.json for hidden/intermediate dims")
    ap.add_argument("--out", required=True, help="Output Lynn fused safetensors path")
    ap.add_argument("--dry-run", action="store_true", help="Print key mapping without writing")
    args = ap.parse_args()

    mtp_path = Path(args.mtp)
    if not mtp_path.exists():
        print(f"ERROR: {mtp_path} not found", file=sys.stderr)
        return 1

    # Read config
    hidden_size = 2048  # Qwen3.6-35B default
    intermediate_size = 1408  # Qwen3.6-35B default
    num_experts = 64  # Qwen3.6-35B default
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            hidden_size = cfg.get("hidden_size", hidden_size)
            intermediate_size = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", intermediate_size))
            num_experts = cfg.get("num_experts", cfg.get("num_local_experts", num_experts))

    print(f"[converter] source: {mtp_path}", file=sys.stderr)
    print(f"[converter] hidden={hidden_size} intermediate={intermediate_size} experts={num_experts}", file=sys.stderr)

    # Read header
    header = _read_header(mtp_path)
    print(f"[converter] source keys: {len(header)}", file=sys.stderr)

    # Classify keys
    expert_gate_keys: dict[int, str] = {}  # expert_id → gate_proj key
    expert_up_keys: dict[int, str] = {}    # expert_id → up_proj key
    expert_down_keys: dict[int, str] = {}  # expert_id → down_proj key
    passthrough_keys: list[tuple[str, str]] = []  # (official_key, lynn_key)
    fc_key: str | None = None
    pre_fc_norm_embed_key: str | None = None
    pre_fc_norm_hidden_key: str | None = None
    output_norm_key: str | None = None
    scale_inv_keys: set[str] = set()

    for key in header:
        kl = key.lower()
        # Skip scale_inv (handled during dequant)
        if "scale_inv" in kl or "scale" in kl:
            scale_inv_keys.add(key)
            continue

        # Top-level MTP tensors
        if "pre_fc_norm_embedding" in kl:
            pre_fc_norm_embed_key = key
        elif "pre_fc_norm_hidden" in kl:
            pre_fc_norm_hidden_key = key
        elif re.search(r"\bfc\b.*weight", kl) and "norm" not in kl and "expert" not in kl:
            fc_key = key
        elif re.search(r"^.*\.norm\.weight$|^.*mtp_norm.*weight$", kl) and "layer" not in kl and "fc" not in kl:
            output_norm_key = key

        # Per-expert MLP
        eid = _parse_expert_id(key)
        if eid is not None and "shared" not in kl:
            if "gate_proj" in kl and "scale" not in kl:
                expert_gate_keys[eid] = key
            elif "up_proj" in kl and "scale" not in kl:
                expert_up_keys[eid] = key
            elif "down_proj" in kl and "scale" not in kl:
                expert_down_keys[eid] = key
            continue

        # Layer-level keys (attention, norms, shared expert, router)
        lynn_key = _lynn_layer_key(key)
        if lynn_key:
            passthrough_keys.append((key, lynn_key))

    # Validate expert counts
    detected_experts = max(
        len(expert_gate_keys), len(expert_up_keys), len(expert_down_keys)
    )
    if detected_experts == 0:
        print(f"[converter] WARNING: no per-expert keys found. May be pre-fused.", file=sys.stderr)
    else:
        print(f"[converter] detected {detected_experts} experts", file=sys.stderr)
        num_experts = detected_experts

    # Print mapping
    print(f"\n[converter] Key mapping:", file=sys.stderr)
    if fc_key:
        print(f"  {fc_key} → mtp.fc.weight", file=sys.stderr)
    if pre_fc_norm_embed_key:
        print(f"  {pre_fc_norm_embed_key} → mtp.pre_fc_norm_embedding.weight", file=sys.stderr)
    if pre_fc_norm_hidden_key:
        print(f"  {pre_fc_norm_hidden_key} → mtp.pre_fc_norm_hidden.weight", file=sys.stderr)
    if output_norm_key:
        print(f"  {output_norm_key} → mtp.norm.weight", file=sys.stderr)
    for off_key, lynn_key in passthrough_keys:
        print(f"  {off_key} → {lynn_key}", file=sys.stderr)
    if detected_experts:
        print(f"  experts.*.gate_proj + up_proj → mtp.layers.0.mlp.experts.gate_up_proj [{num_experts}, {2*intermediate_size}, {hidden_size}]", file=sys.stderr)
        print(f"  experts.*.down_proj → mtp.layers.0.mlp.experts.down_proj [{num_experts}, {hidden_size}, {intermediate_size}]", file=sys.stderr)
    print(f"  scale_inv keys: {len(scale_inv_keys)} (applied during dequant)", file=sys.stderr)

    if args.dry_run:
        print(f"\n[converter] DRY_RUN — not writing output.", file=sys.stderr)
        return 0

    # ─────────────────────────────────────────────────────────────
    # Load and convert
    # ─────────────────────────────────────────────────────────────
    print(f"\n[converter] Loading tensors...", file=sys.stderr)
    output_tensors: dict[str, torch.Tensor] = {}
    conversion_log: list[dict[str, Any]] = []
    fp8_count = 0

    def _load_and_dequant(key: str) -> torch.Tensor:
        nonlocal fp8_count
        t = _load_tensor_cpu(mtp_path, key)
        if t.dtype in (torch.float8_e4m3fn, torch.uint8) and t.element_size() == 1:
            scale_key = _find_scale_inv_key(key, header)
            scale = _load_tensor_cpu(mtp_path, scale_key) if scale_key else None
            result = _dequant_fp8_with_scale(t, scale)
            fp8_count += 1
            conversion_log.append({"key": key, "source_dtype": "fp8_e4m3fn", "scale_key": scale_key})
            return result
        return t.to(torch.bfloat16) if t.is_floating_point() else t

    # Top-level tensors
    if fc_key:
        output_tensors["mtp.fc.weight"] = _load_and_dequant(fc_key)
    if pre_fc_norm_embed_key:
        output_tensors["mtp.pre_fc_norm_embedding.weight"] = _load_and_dequant(pre_fc_norm_embed_key)
    if pre_fc_norm_hidden_key:
        output_tensors["mtp.pre_fc_norm_hidden.weight"] = _load_and_dequant(pre_fc_norm_hidden_key)
    if output_norm_key:
        output_tensors["mtp.norm.weight"] = _load_and_dequant(output_norm_key)

    # Passthrough layer keys
    for off_key, lynn_key in passthrough_keys:
        output_tensors[lynn_key] = _load_and_dequant(off_key)

    # Fuse per-expert gate+up → gate_up_proj
    if detected_experts > 0:
        print(f"[converter] Fusing {num_experts} experts gate+up...", file=sys.stderr)
        gate_up_list: list[torch.Tensor] = []
        down_list: list[torch.Tensor] = []

        for eid in range(num_experts):
            gate_key = expert_gate_keys.get(eid)
            up_key = expert_up_keys.get(eid)
            down_key = expert_down_keys.get(eid)

            if gate_key and up_key:
                gate_t = _load_and_dequant(gate_key)
                up_t = _load_and_dequant(up_key)
                fused = torch.cat([gate_t, up_t], dim=0)  # [2*intermediate, hidden]
                gate_up_list.append(fused)
            else:
                print(f"  [WARN] expert {eid}: missing gate_proj or up_proj", file=sys.stderr)
                gate_up_list.append(torch.zeros(2 * intermediate_size, hidden_size, dtype=torch.bfloat16))

            if down_key:
                down_t = _load_and_dequant(down_key)
                down_list.append(down_t)
            else:
                print(f"  [WARN] expert {eid}: missing down_proj", file=sys.stderr)
                down_list.append(torch.zeros(hidden_size, intermediate_size, dtype=torch.bfloat16))

        # Stack: [num_experts, 2*intermediate, hidden]
        gate_up_fused = torch.stack(gate_up_list, dim=0)
        output_tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = gate_up_fused
        print(f"  gate_up_proj shape: {list(gate_up_fused.shape)}", file=sys.stderr)

        # Stack: [num_experts, hidden, intermediate]
        down_fused = torch.stack(down_list, dim=0)
        output_tensors["mtp.layers.0.mlp.experts.down_proj"] = down_fused
        print(f"  down_proj shape: {list(down_fused.shape)}", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────
    # Write output
    # ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute sha256 prefix of the serialized data
    metadata = {
        "source_path": str(mtp_path),
        "source_key_count": str(len(header)),
        "expert_count": str(num_experts),
        "hidden_size": str(hidden_size),
        "intermediate_size": str(intermediate_size),
        "conversion_mode": "fp8_to_bf16_fused" if fp8_count > 0 else "passthrough_fused",
        "fp8_tensors_dequanted": str(fp8_count),
        "output_key_count": str(len(output_tensors)),
        "converter": "qwen36_convert_official_mtp_to_lynn_fused.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    save_file(output_tensors, str(out_path), metadata=metadata)

    # Compute sha256 of output file
    h = hashlib.sha256()
    with out_path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    sha256_prefix = h.hexdigest()[:32]
    print(f"\n[converter] Output: {out_path}", file=sys.stderr)
    print(f"[converter] Keys: {len(output_tensors)}", file=sys.stderr)
    print(f"[converter] Size: {out_path.stat().st_size / (1024*1024):.1f} MiB", file=sys.stderr)
    print(f"[converter] SHA256 prefix: {sha256_prefix}", file=sys.stderr)
    print(f"[converter] FP8 tensors dequanted: {fp8_count}", file=sys.stderr)
    print(f"[converter] Metadata: {json.dumps(metadata, indent=2)}", file=sys.stderr)

    # Print output key list
    print(f"\n[converter] Output keys:", file=sys.stderr)
    for key in sorted(output_tensors.keys()):
        t = output_tensors[key]
        print(f"  {key:60s} {str(t.dtype):12s} {list(t.shape)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
