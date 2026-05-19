#!/usr/bin/env python3
"""P199 · Qwen3.5-9B NVFP4 artifact size audit + shrink plan.

Reads the Lynn-native NVFP4 model directory, categorizes every stored byte
into quantized packed weights / scales / kept BF16 / metadata / unknown,
and produces a JSON report with three shrink options.

Requires only Python stdlib + torch (for safetensors header reading).
Does NOT modify engine / server / resident_runner code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────────
Q4KM_REFERENCE_GIB = 5.3

# Category regexes for kept BF16 tensors
_KEPT_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("embed_tokens", re.compile(r"embed_tokens")),
    ("lm_head", re.compile(r"lm_head")),
    ("norms", re.compile(r"(layernorm|rmsnorm|norm)", re.IGNORECASE)),
    ("rope", re.compile(r"rotary|rope", re.IGNORECASE)),
    ("mlp_gate", re.compile(r"mlp\.gate")),
    ("visual", re.compile(r"visual")),
    ("mtp", re.compile(r"\.mtp\b|mtp\.", re.IGNORECASE)),
]

# Suffix patterns for quantized tensor components
_PACKED_SUFFIXES = (".packed", ".weight_packed")
_SCALE_SUFFIXES = (".scale", ".weight_scale")
_GLOBAL_SCALE_SUFFIXES = (".global_scale", ".weight_global_scale")


def _gib(n: int) -> float:
    return round(n / (1024**3), 4)


def _classify_tensor(key: str) -> str:
    """Classify a single safetensors key into a category."""
    for suffix in _PACKED_SUFFIXES:
        if key.endswith(suffix):
            return "quantized_packed"
    for suffix in _SCALE_SUFFIXES:
        if key.endswith(suffix):
            return "quantized_scale"
    for suffix in _GLOBAL_SCALE_SUFFIXES:
        if key.endswith(suffix):
            return "quantized_global_scale"
    # Not quantized — check kept BF16 categories
    for cat_name, pattern in _KEPT_CATEGORIES:
        if pattern.search(key):
            return f"kept_bf16_{cat_name}"
    # Config/tokenizer/metadata files are not tensors
    return "kept_bf16_other"


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read the JSON header from a .safetensors file (first 8 bytes = header len)."""
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            return {}
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_json = f.read(header_len)
        return json.loads(header_json)


def _scan_model_dir(model_dir: Path) -> dict[str, Any]:
    """Scan a model directory and return size breakdown."""
    # ── Collect file sizes ──────────────────────────────────────────────────
    file_sizes: dict[str, int] = {}
    for f in model_dir.iterdir():
        if f.is_file():
            file_sizes[f.name] = f.stat().st_size

    total_file_bytes = sum(file_sizes.values())

    # ── Read manifest if present ────────────────────────────────────────────
    manifest_path = model_dir / "lynn_quant_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ── Scan safetensors headers ────────────────────────────────────────────
    st_files = sorted(model_dir.glob("*.safetensors"))
    tensor_meta: dict[str, dict[str, Any]] = {}
    for st_path in st_files:
        try:
            header = _read_safetensors_header(st_path)
        except Exception:
            continue
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            # safetensors header: {"dtype": "F16", "shape": [4096, 4096], "data_offsets": [start, end]}
            offsets = meta.get("data_offsets", [0, 0])
            nbytes = offsets[1] - offsets[0] if len(offsets) == 2 else 0
            tensor_meta[key] = {
                "file": st_path.name,
                "dtype": meta.get("dtype", "?"),
                "shape": meta.get("shape", []),
                "nbytes": nbytes,
            }

    # ── Categorize each tensor ──────────────────────────────────────────────
    category_bytes: dict[str, int] = defaultdict(int)
    category_count: dict[str, int] = defaultdict(int)
    kept_bf16_tensors: list[dict[str, Any]] = []

    for key, meta in tensor_meta.items():
        cat = _classify_tensor(key)
        nb = meta["nbytes"]
        category_bytes[cat] += nb
        category_count[cat] += 1
        if cat.startswith("kept_bf16_"):
            kept_bf16_tensors.append({
                "key": key,
                "category": cat.replace("kept_bf16_", ""),
                "shape": meta["shape"],
                "dtype": meta["dtype"],
                "nbytes": nb,
                "gib": _gib(nb),
            })

    # Sort kept BF16 by size descending
    kept_bf16_tensors.sort(key=lambda x: x["nbytes"], reverse=True)

    # ── Aggregate categories ───────────────────────────────────────────────
    # Merge all quantized_packed/scale/global_scale into "quantized_total"
    quantized_total = (
        category_bytes.get("quantized_packed", 0)
        + category_bytes.get("quantized_scale", 0)
        + category_bytes.get("quantized_global_scale", 0)
    )
    kept_total = sum(v for k, v in category_bytes.items() if k.startswith("kept_bf16_"))

    # Non-tensor files (tokenizer, config, manifest, etc.)
    tensor_file_bytes = sum(
        meta["nbytes"] for meta in tensor_meta.values()
    )
    non_tensor_bytes = total_file_bytes - tensor_file_bytes
    if non_tensor_bytes < 0:
        non_tensor_bytes = 0

    # ── Category breakdown ──────────────────────────────────────────────────
    category_breakdown = {
        "quantized_packed": {"gib": _gib(category_bytes.get("quantized_packed", 0)),
                             "count": category_count.get("quantized_packed", 0)},
        "quantized_scale": {"gib": _gib(category_bytes.get("quantized_scale", 0)),
                            "count": category_count.get("quantized_scale", 0)},
        "quantized_global_scale": {"gib": _gib(category_bytes.get("quantized_global_scale", 0)),
                                   "count": category_count.get("quantized_global_scale", 0)},
        "quantized_total": {"gib": _gib(quantized_total)},
        "kept_bf16_total": {"gib": _gib(kept_total)},
        "non_tensor_metadata": {"gib": _gib(non_tensor_bytes)},
    }

    # ── Kept BF16 breakdown by category ────────────────────────────────────
    kept_by_cat: dict[str, dict[str, Any]] = defaultdict(lambda: {"gib": 0.0, "count": 0})
    for t in kept_bf16_tensors:
        kept_by_cat[t["category"]]["gib"] = round(kept_by_cat[t["category"]]["gib"] + t["gib"], 4)
        kept_by_cat[t["category"]]["count"] += 1

    # ── Compute shrinkable amounts ──────────────────────────────────────────
    embed_gib = kept_by_cat.get("embed_tokens", {}).get("gib", 0.0)
    lmhead_gib = kept_by_cat.get("lm_head", {}).get("gib", 0.0)
    visual_gib = (
        kept_by_cat.get("visual", {}).get("gib", 0.0)
        + category_breakdown.get("quantized_packed", {}).get("gib", 0.0) * 0.06  # rough visual fraction
    )
    mtp_gib = kept_by_cat.get("mtp", {}).get("gib", 0.0)

    total_gib = _gib(total_file_bytes)

    # ── Shrink options ──────────────────────────────────────────────────────
    shrink_options = [
        {
            "tier": "SAFE_NO_CHANGE",
            "description": "Ship current artifact as-is.  No quality risk.",
            "expected_size_gib": total_gib,
            "quality_risk": "none",
            "changes": [],
        },
        {
            "tier": "MODERATE_QUANTIZE_EMBED_LMHEAD",
            "description": (
                "Quantize embed_tokens + lm_head to NVFP4.  Saves ~3.5 GiB "
                "but lm_head FP4 exact gate currently fails on 9B.  Needs "
                "MMLU/GPQA spot-check before promotion."
            ),
            "expected_size_gib": round(total_gib - embed_gib - lmhead_gib + 0.8, 2),
            "quality_risk": "high",
            "changes": [
                f"quantize embed_tokens.weight (currently {embed_gib:.2f} GiB BF16)",
                f"quantize lm_head.weight (currently {lmhead_gib:.2f} GiB BF16)",
            ],
        },
        {
            "tier": "AGGRESSIVE_QUANTIZE_MORE_KEEPERS",
            "description": (
                "Quantize embed_tokens + lm_head + prune visual/MTP for text-only. "
                "Saves ~4+ GiB.  Highest quality risk.  Only after MODERATE tier passes."
            ),
            "expected_size_gib": round(total_gib - embed_gib - lmhead_gib - visual_gib - mtp_gib + 0.5, 2),
            "quality_risk": "very high",
            "changes": [
                f"quantize embed_tokens.weight ({embed_gib:.2f} GiB)",
                f"quantize lm_head.weight ({lmhead_gib:.2f} GiB)",
                f"prune visual tensors (~{visual_gib:.2f} GiB)",
                f"prune MTP tensors (~{mtp_gib:.2f} GiB)",
            ],
        },
    ]

    # ── Final report ────────────────────────────────────────────────────────
    return {
        "schema": "lynn-nvfp4-size-audit-v1",
        "model_dir": str(model_dir),
        "total_gib": total_gib,
        "total_bytes": total_file_bytes,
        "n_files": len(file_sizes),
        "n_tensors": len(tensor_meta),
        "q4km_reference_gib": Q4KM_REFERENCE_GIB,
        "delta_gib": round(total_gib - Q4KM_REFERENCE_GIB, 4),
        "manifest_present": manifest is not None,
        "manifest_schema": manifest.get("schema_version") if manifest else None,
        "keep_regex": (manifest or {}).get("quantization", {}).get("keep_regex", "N/A"),
        "category_breakdown": dict(category_breakdown),
        "kept_bf16_breakdown": dict(kept_by_cat),
        "kept_bf16_top30": kept_bf16_tensors[:30],
        "shrink_options": shrink_options,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P199 NVFP4 artifact size audit.")
    ap.add_argument("--model-dir", required=True, help="Path to NVFP4 model directory.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"[p199] model dir not found: {model_dir}")
        print(f"[p199] writing PENDING status report")
        result = {
            "schema": "lynn-nvfp4-size-audit-v1",
            "model_dir": str(model_dir),
            "decision": "PENDING",
            "reason": "model dir not found on this host",
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[p198] report: {out_path}")
        return 0

    print(f"[p199] scanning {model_dir} ...")
    result = _scan_model_dir(model_dir)

    # Pretty summary
    print(f"\n[p199] === Size Audit ===")
    print(f"[p199] total:          {result['total_gib']:.3f} GiB ({result['n_files']} files, {result['n_tensors']} tensors)")
    print(f"[p199] Q4_K_M ref:     {Q4KM_REFERENCE_GIB} GiB")
    print(f"[p199] delta:          +{result['delta_gib']:.3f} GiB")
    print(f"[p199] quantized:      {result['category_breakdown']['quantized_total']['gib']:.3f} GiB")
    print(f"[p199] kept BF16:      {result['category_breakdown']['kept_bf16_total']['gib']:.3f} GiB")
    print(f"[p199] metadata:       {result['category_breakdown']['non_tensor_metadata']['gib']:.3f} GiB")
    print()
    print(f"[p199] === Kept BF16 Breakdown ===")
    for cat, info in sorted(result["kept_bf16_breakdown"].items(), key=lambda x: -x[1]["gib"]):
        print(f"[p199]   {cat:20s}  {info['gib']:.3f} GiB  ({info['count']} tensors)")
    print()
    print(f"[p199] === Shrink Options ===")
    for opt in result["shrink_options"]:
        print(f"[p199]   {opt['tier']:45s}  → {opt['expected_size_gib']:.2f} GiB  risk={opt['quality_risk']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[p199] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
