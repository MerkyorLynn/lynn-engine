#!/usr/bin/env python3
"""Qwen MTP Public Head Inventory.

Scans a Qwen model directory (safetensors) and inventories all MTP-related
tensors (keys matching `mtp.*` or `model.mtp*`). Reports key, shape, dtype,
and SHA256 prefix for each tensor.

Does NOT require GPU or torch.cuda. Reads safetensors metadata + raw bytes
on CPU only.

Supports:
  - Qwen3.5-9B dense (official MTP heads, if present)
  - Qwen3.6-35B-A3B MoE (official NextN MTP heads)
  - Any sharded safetensors with index.json

Usage:
  python scripts/qwen_mtp_public_head_inventory.py --model-dir /path/to/model
  python scripts/qwen_mtp_public_head_inventory.py --model-dir /path/to/model --out report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


def _read_safetensors_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read safetensors header (tensor metadata) without loading tensor data.

    Returns {tensor_name: {"shape": [...], "dtype": "...", "data_offsets": [start, end]}}
    """
    with path.open("rb") as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            return {}
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size > 100_000_000:  # sanity: >100MB header is suspicious
            return {}
        header_bytes = f.read(header_size)
    header = json.loads(header_bytes)
    # Remove __metadata__ key if present
    header.pop("__metadata__", None)
    return header


def _sha256_prefix(path: Path, offset: int, length: int, prefix_len: int = 16) -> str:
    """Compute SHA256 of tensor bytes and return hex prefix."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        # Skip past the header
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        data_start = 8 + header_size
        f.seek(data_start + offset)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(remaining, 1 << 20))  # 1MB chunks
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()[:prefix_len]


def _discover_shards(model_dir: Path) -> list[Path]:
    """Find all safetensors shards, respecting index if present."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        return [model_dir / name for name in shard_names if (model_dir / name).exists()]

    # Single file
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    # Glob
    shards = sorted(model_dir.glob("*.safetensors"))
    return shards


def _detect_model_config(model_dir: Path) -> dict[str, Any]:
    """Read config.json for model_type, hidden_size, mtp info."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {"model_type": "unknown", "hidden_size": None, "mtp_num_hidden_layers": None}
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "model_type": cfg.get("model_type", "unknown"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_experts": cfg.get("num_experts", cfg.get("num_local_experts")),
        "mtp_num_hidden_layers": cfg.get("mtp_num_hidden_layers", cfg.get("num_nextn_predict_layers")),
        "mtp_hidden_size": cfg.get("mtp_hidden_size"),
        "vocab_size": cfg.get("vocab_size"),
    }


def _is_mtp_key(key: str) -> bool:
    """Check if a tensor key belongs to the MTP head."""
    lower = key.lower()
    return (
        "mtp" in lower
        or "nextn" in lower
        or "next_n" in lower
        or "multi_token" in lower
        or "speculative" in lower
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen MTP Public Head Inventory")
    ap.add_argument("--model-dir", required=True, help="Path to model directory with safetensors")
    ap.add_argument("--out", default=None, help="Output JSON path (default: stdout)")
    ap.add_argument("--sha256-prefix-len", type=int, default=16, help="SHA256 hex prefix length")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        return 1

    # Detect model config
    config = _detect_model_config(model_dir)
    print(f"[mtp-inventory] model_dir: {model_dir}", file=sys.stderr)
    print(f"[mtp-inventory] model_type: {config['model_type']}", file=sys.stderr)
    print(f"[mtp-inventory] hidden_size: {config['hidden_size']}", file=sys.stderr)
    print(f"[mtp-inventory] mtp_num_hidden_layers: {config['mtp_num_hidden_layers']}", file=sys.stderr)

    # Discover shards
    shards = _discover_shards(model_dir)
    if not shards:
        print(f"ERROR: no safetensors files found in {model_dir}", file=sys.stderr)
        return 1
    print(f"[mtp-inventory] shards: {len(shards)}", file=sys.stderr)

    # Scan all tensors, collect MTP ones
    mtp_tensors: list[dict[str, Any]] = []
    all_tensor_count = 0

    for shard_path in shards:
        if not shard_path.exists():
            continue
        metadata = _read_safetensors_metadata(shard_path)
        for key, info in metadata.items():
            all_tensor_count += 1
            if not _is_mtp_key(key):
                continue
            shape = info.get("shape", [])
            dtype = info.get("dtype", "unknown")
            offsets = info.get("data_offsets", [0, 0])
            byte_size = offsets[1] - offsets[0] if len(offsets) == 2 else 0

            # Compute sha256 prefix
            sha_prefix = ""
            if byte_size > 0:
                sha_prefix = _sha256_prefix(
                    shard_path, offsets[0], byte_size,
                    prefix_len=args.sha256_prefix_len,
                )

            mtp_tensors.append({
                "key": key,
                "shape": shape,
                "dtype": dtype,
                "byte_size": byte_size,
                "sha256_prefix": sha_prefix,
                "shard": shard_path.name,
            })

    # Sort by key for stable output
    mtp_tensors.sort(key=lambda t: t["key"])

    # Build report
    report = {
        "schema": "lynn-qwen-mtp-public-head-inventory-v1",
        "model_dir": str(model_dir),
        "model_config": config,
        "total_tensors": all_tensor_count,
        "mtp_tensors_count": len(mtp_tensors),
        "mtp_tensors": mtp_tensors,
        "mtp_total_bytes": sum(t["byte_size"] for t in mtp_tensors),
        "mtp_total_mib": sum(t["byte_size"] for t in mtp_tensors) / (1024 * 1024),
    }

    # Output
    output_text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        print(f"[mtp-inventory] wrote: {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)

    # Summary to stderr
    print(f"[mtp-inventory] total tensors: {all_tensor_count}", file=sys.stderr)
    print(f"[mtp-inventory] MTP tensors: {len(mtp_tensors)}", file=sys.stderr)
    print(f"[mtp-inventory] MTP size: {report['mtp_total_mib']:.2f} MiB", file=sys.stderr)
    if mtp_tensors:
        print(f"[mtp-inventory] MTP keys:", file=sys.stderr)
        for t in mtp_tensors[:20]:
            print(f"  {t['key']:60s} {t['dtype']:8s} {t['shape']}  sha={t['sha256_prefix']}", file=sys.stderr)
        if len(mtp_tensors) > 20:
            print(f"  ... and {len(mtp_tensors) - 20} more", file=sys.stderr)
    else:
        print(f"[mtp-inventory] WARNING: No MTP tensors found. Model may not have MTP heads.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
