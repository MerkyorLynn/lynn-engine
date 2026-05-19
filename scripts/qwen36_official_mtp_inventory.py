#!/usr/bin/env python3
"""Inventory the official Qwen3.6 MTP FP8 safetensors.

Reads the official `mtp.safetensors` (or sharded equivalent) and produces
a structured inventory: key_count, dtype_counts, shape_groups, expert layout,
pre_fc_norm presence, attention shapes, etc.

CPU-only — no GPU required. Does not load full tensors into memory for
large models; reads safetensors headers + selective tensor access.

Usage:
  python scripts/qwen36_official_mtp_inventory.py \\
    --mtp /home/merkyor/models/Qwen3.6-35B-A3B-FP8/mtp.safetensors \\
    --config /home/merkyor/models/Qwen3.6-35B-A3B-FP8/config.json \\
    --out reports/mtp/qwen36_official_mtp_inventory.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read safetensors header without loading tensor data."""
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def _classify_key(key: str) -> str:
    """Classify a tensor key into a functional group."""
    kl = key.lower()
    if "pre_fc_norm_embedding" in kl:
        return "pre_fc_norm_embedding"
    if "pre_fc_norm_hidden" in kl:
        return "pre_fc_norm_hidden"
    if ".fc." in kl or kl.endswith(".fc.weight"):
        return "fc_projection"
    if "norm" in kl and "layer" not in kl:
        return "output_norm"
    if "input_layernorm" in kl:
        return "input_layernorm"
    if "post_attention_layernorm" in kl:
        return "post_attention_layernorm"
    if "self_attn" in kl:
        if "q_proj" in kl:
            return "self_attn.q_proj"
        if "k_proj" in kl:
            return "self_attn.k_proj"
        if "v_proj" in kl:
            return "self_attn.v_proj"
        if "o_proj" in kl:
            return "self_attn.o_proj"
        if "q_norm" in kl:
            return "self_attn.q_norm"
        if "k_norm" in kl:
            return "self_attn.k_norm"
        return "self_attn.other"
    if "shared_expert_gate" in kl:
        return "shared_expert_gate"
    if "shared_expert" in kl:
        if "gate_proj" in kl:
            return "shared_expert.gate_proj"
        if "up_proj" in kl:
            return "shared_expert.up_proj"
        if "down_proj" in kl:
            return "shared_expert.down_proj"
        return "shared_expert.other"
    if "gate" in kl and "proj" not in kl and "up" not in kl:
        return "router_gate"
    if "expert" in kl or "mlp" in kl:
        if "gate_proj" in kl:
            return "expert.gate_proj"
        if "up_proj" in kl:
            return "expert.up_proj"
        if "down_proj" in kl:
            return "expert.down_proj"
        if "scale_inv" in kl or "scale" in kl:
            return "expert.scale"
        return "expert.other"
    if "scale_inv" in kl or "scale" in kl:
        return "scale"
    return "other"


def _detect_expert_count(header: dict[str, dict[str, Any]]) -> int:
    """Detect number of experts from key patterns like experts.42.gate_proj."""
    import re
    expert_ids: set[int] = set()
    for key in header:
        m = re.search(r"experts?[._](\d+)", key)
        if m:
            expert_ids.add(int(m.group(1)))
    return len(expert_ids) if expert_ids else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3.6 Official MTP Inventory")
    ap.add_argument("--mtp", required=True, help="Path to official mtp.safetensors")
    ap.add_argument("--config", default=None, help="Path to config.json (optional)")
    ap.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    args = ap.parse_args()

    mtp_path = Path(args.mtp)
    if not mtp_path.exists():
        print(f"ERROR: mtp file not found: {mtp_path}", file=sys.stderr)
        return 1

    # Read config
    config_info: dict[str, Any] = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            config_info = {
                "model_type": cfg.get("model_type", "unknown"),
                "hidden_size": cfg.get("hidden_size"),
                "intermediate_size": cfg.get("intermediate_size"),
                "num_experts": cfg.get("num_experts", cfg.get("num_local_experts")),
                "mtp_num_hidden_layers": cfg.get("mtp_num_hidden_layers", cfg.get("num_nextn_predict_layers")),
                "vocab_size": cfg.get("vocab_size"),
            }

    # Read header
    print(f"[inventory] reading: {mtp_path}", file=sys.stderr)
    header = _read_safetensors_header(mtp_path)
    key_count = len(header)
    print(f"[inventory] keys: {key_count}", file=sys.stderr)

    # Classify
    dtype_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    shape_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_keys: list[dict[str, Any]] = []

    for key, info in sorted(header.items()):
        dtype = info.get("dtype", "unknown")
        shape = info.get("shape", [])
        offsets = info.get("data_offsets", [0, 0])
        byte_size = offsets[1] - offsets[0] if len(offsets) == 2 else 0
        group = _classify_key(key)

        dtype_counts[dtype] += 1
        group_counts[group] += 1
        entry = {"key": key, "dtype": dtype, "shape": shape, "byte_size": byte_size, "group": group}
        all_keys.append(entry)
        shape_groups[group].append(entry)

    # Detect expert count
    expert_count = _detect_expert_count(header)

    # Check for pre_fc_norm
    has_pre_fc_norm_embedding = any("pre_fc_norm_embedding" in k.lower() for k in header)
    has_pre_fc_norm_hidden = any("pre_fc_norm_hidden" in k.lower() for k in header)

    # Self-attn summary
    attn_keys = [e for e in all_keys if "self_attn" in e["group"]]
    attn_summary = {}
    if attn_keys:
        attn_summary = {
            "count": len(attn_keys),
            "dtypes": list(set(e["dtype"] for e in attn_keys)),
            "shapes": list(set(str(e["shape"]) for e in attn_keys)),
        }

    # Shared expert summary
    shared_keys = [e for e in all_keys if "shared_expert" in e["group"]]
    shared_summary = {}
    if shared_keys:
        shared_summary = {
            "count": len(shared_keys),
            "dtypes": list(set(e["dtype"] for e in shared_keys)),
            "shapes": list(set(str(e["shape"]) for e in shared_keys)),
        }

    # FC/norm summary
    fc_keys = [e for e in all_keys if e["group"] == "fc_projection"]
    norm_keys = [e for e in all_keys if "norm" in e["group"]]

    report = {
        "schema": "lynn-qwen36-official-mtp-inventory-v1",
        "mtp_path": str(mtp_path),
        "config": config_info,
        "key_count": key_count,
        "dtype_counts": dict(dtype_counts),
        "group_counts": dict(group_counts),
        "expert_count": expert_count,
        "has_pre_fc_norm_embedding": has_pre_fc_norm_embedding,
        "has_pre_fc_norm_hidden": has_pre_fc_norm_hidden,
        "self_attn_summary": attn_summary,
        "shared_expert_summary": shared_summary,
        "fc_shapes": [{"key": e["key"], "shape": e["shape"], "dtype": e["dtype"]} for e in fc_keys],
        "norm_shapes": [{"key": e["key"], "shape": e["shape"], "dtype": e["dtype"]} for e in norm_keys[:20]],
        "total_bytes": sum(e["byte_size"] for e in all_keys),
        "total_mib": sum(e["byte_size"] for e in all_keys) / (1024 * 1024),
    }

    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"[inventory] wrote: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(output)

    # Summary to stderr
    print(f"[inventory] dtypes: {dict(dtype_counts)}", file=sys.stderr)
    print(f"[inventory] groups: {dict(group_counts)}", file=sys.stderr)
    print(f"[inventory] experts: {expert_count}", file=sys.stderr)
    print(f"[inventory] pre_fc_norm_embedding: {has_pre_fc_norm_embedding}", file=sys.stderr)
    print(f"[inventory] pre_fc_norm_hidden: {has_pre_fc_norm_hidden}", file=sys.stderr)
    print(f"[inventory] total: {report['total_mib']:.1f} MiB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
