#!/usr/bin/env python3
"""Build a repack inventory for Lynn-native Qwen3.6 W4A16 NVFP4 artifacts.

The script is intentionally read-only.  It inspects `lynn_quant_manifest.json`
and `model.safetensors.index.json` and emits a compact JSON/Markdown summary of
which tensors should be grouped first for an offline serving-layout repack.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _prod(shape: list[int] | tuple[int, ...] | None) -> int:
    if not shape:
        return 0
    return int(math.prod(int(x) for x in shape))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_id(key: str) -> int | None:
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _bucket(key: str) -> str:
    if ".mlp.experts." in key:
        return "active_moe"
    if ".mlp.shared_expert" in key or ".mlp.shared_expert_gate" in key:
        return "shared_moe"
    if ".linear_attn." in key:
        return "linear_attn"
    if ".self_attn." in key:
        return "full_attn"
    if ".mlp.gate" in key or ".input_layernorm" in key or ".post_attention_layernorm" in key:
        return "routing_norm"
    if key.startswith("model.visual."):
        return "visual"
    if key.startswith("mtp."):
        return "mtp"
    return "other"


def _module_name(key: str) -> str:
    layer = _layer_id(key)
    if layer is not None:
        return key.split(f".layers.{layer}.", 1)[1].rsplit(".weight", 1)[0]
    return key.rsplit(".weight", 1)[0]


def _shards_for_record(rec: dict[str, Any], weight_map: dict[str, str]) -> set[str]:
    shards: set[str] = set()
    for field in ("packed_key", "scale_key", "global_scale_key"):
        mapped = weight_map.get(str(rec.get(field, "")))
        if mapped:
            shards.add(mapped)
    return shards


def build_inventory(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "lynn_quant_manifest.json"
    index_path = model_dir / "model.safetensors.index.json"
    manifest = _read_json(manifest_path)
    index = _read_json(index_path)
    weight_map = index.get("weight_map", {})
    records: dict[str, dict[str, Any]] = manifest.get("quantized_tensors", {})
    if not isinstance(records, dict):
        raise TypeError("manifest quantized_tensors must be a dict")

    by_bucket: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "packed_bytes": 0,
            "scale_elements": 0,
            "global_scale_count": 0,
            "shards": set(),
            "modules": Counter(),
        }
    )
    by_layer: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "packed_bytes": 0,
            "scale_elements": 0,
            "buckets": Counter(),
            "modules": Counter(),
            "shards": set(),
        }
    )
    missing_index: list[str] = []

    for key, rec in records.items():
        bucket = _bucket(key)
        layer = _layer_id(key)
        packed_bytes = _prod(rec.get("packed_shape"))
        scale_elements = _prod(rec.get("scale_shape"))
        shards = _shards_for_record(rec, weight_map)
        for field in ("packed_key", "scale_key", "global_scale_key"):
            tensor_key = str(rec.get(field, ""))
            if tensor_key and tensor_key not in weight_map:
                missing_index.append(tensor_key)

        b = by_bucket[bucket]
        b["records"] += 1
        b["packed_bytes"] += packed_bytes
        b["scale_elements"] += scale_elements
        b["global_scale_count"] += 1 if rec.get("global_scale_key") else 0
        b["shards"].update(shards)
        b["modules"][_module_name(key)] += 1

        if layer is not None:
            l = by_layer[layer]
            l["records"] += 1
            l["packed_bytes"] += packed_bytes
            l["scale_elements"] += scale_elements
            l["buckets"][bucket] += 1
            l["modules"][_module_name(key)] += 1
            l["shards"].update(shards)

    def clean_group(group: dict[str, Any]) -> dict[str, Any]:
        return {
            "records": group["records"],
            "packed_bytes": group["packed_bytes"],
            "packed_gib": round(group["packed_bytes"] / (1024**3), 4),
            "scale_elements": group["scale_elements"],
            "global_scale_count": group["global_scale_count"],
            "shard_count": len(group["shards"]),
            "shards": sorted(group["shards"]),
            "top_modules": group["modules"].most_common(12),
        }

    layer_rows = []
    for layer, group in sorted(by_layer.items()):
        layer_rows.append(
            {
                "layer": layer,
                "records": group["records"],
                "packed_bytes": group["packed_bytes"],
                "packed_mib": round(group["packed_bytes"] / (1024**2), 2),
                "scale_elements": group["scale_elements"],
                "buckets": dict(group["buckets"]),
                "shard_count": len(group["shards"]),
                "shards": sorted(group["shards"]),
                "modules": group["modules"].most_common(16),
            }
        )

    language_rows = [r for r in layer_rows if 0 <= r["layer"] < 40]
    full_attn_layers = [r["layer"] for r in language_rows if r["buckets"].get("full_attn")]
    linear_attn_layers = [r["layer"] for r in language_rows if r["buckets"].get("linear_attn")]

    return {
        "schema_version": "qwen36-w4a16-repack-inventory-v1",
        "model_dir": str(model_dir),
        "manifest_schema": manifest.get("schema_version"),
        "quantized_count": len(records),
        "kept_count": len(manifest.get("kept_tensors", {})),
        "missing_index_keys": sorted(set(missing_index)),
        "buckets": {name: clean_group(group) for name, group in sorted(by_bucket.items())},
        "language_layers": {
            "count": len(language_rows),
            "linear_attn_layers": linear_attn_layers,
            "full_attn_layers": full_attn_layers,
            "linear_attn_count": len(linear_attn_layers),
            "full_attn_count": len(full_attn_layers),
        },
        "layers": layer_rows,
        "recommended_repack_order": [
            {
                "name": "active_moe_gateup_down",
                "reason": "largest repeated decode boundary; Q4_K_M llama.cpp suggests serving layout matters more than another scalar tile sweep",
                "bucket": "active_moe",
            },
            {
                "name": "shared_moe_gateup_down_gate",
                "reason": "hot per-token shared expert path; keep BF16 activation semantics",
                "bucket": "shared_moe",
            },
            {
                "name": "linear_attn_projection_pack",
                "reason": "30 hybrid SSM layers carry multiple projection launches per token",
                "bucket": "linear_attn",
            },
            {
                "name": "full_attn_qkvo_pack",
                "reason": "11 full-attention layers still need exact RoPE/cache order",
                "bucket": "full_attn",
            },
        ],
    }


def write_markdown(inv: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Qwen3.6 35B W4A16 Repack Inventory")
    lines.append("")
    lines.append("This is a read-only inventory for the offline serving-layout repack route.")
    lines.append("")
    lines.append(f"- model: `{inv['model_dir']}`")
    lines.append(f"- quantized tensors: `{inv['quantized_count']}`")
    lines.append(f"- missing index keys: `{len(inv['missing_index_keys'])}`")
    ll = inv["language_layers"]
    lines.append(
        f"- language layers: `{ll['count']}`; linear-attn `{ll['linear_attn_count']}`, full-attn `{ll['full_attn_count']}`"
    )
    lines.append("")
    lines.append("## Bucket Summary")
    lines.append("")
    lines.append("| Bucket | Records | Packed GiB | Scale Elements | Shards | Top Modules |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, group in inv["buckets"].items():
        top = ", ".join(f"{m} ({c})" for m, c in group["top_modules"][:4])
        lines.append(
            f"| `{name}` | {group['records']} | {group['packed_gib']:.4f} | {group['scale_elements']} | {group['shard_count']} | {top} |"
        )
    lines.append("")
    lines.append("## Repack Order")
    lines.append("")
    for i, item in enumerate(inv["recommended_repack_order"], 1):
        lines.append(f"{i}. `{item['name']}` (`{item['bucket']}`): {item['reason']}.")
    lines.append("")
    lines.append("## Language Layer Summary")
    lines.append("")
    lines.append("| Layer | Packed MiB | Buckets | Shards |")
    lines.append("|---:|---:|---|---:|")
    for row in inv["layers"]:
        if not (0 <= row["layer"] < 40):
            continue
        buckets = ", ".join(f"{k}:{v}" for k, v in sorted(row["buckets"].items()))
        lines.append(f"| {row['layer']} | {row['packed_mib']:.2f} | {buckets} | {row['shard_count']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, help="write JSON inventory")
    ap.add_argument("--markdown", type=Path, help="write Markdown summary")
    args = ap.parse_args()

    inv = build_inventory(args.model_dir)
    text = json.dumps(inv, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(inv, args.markdown)


if __name__ == "__main__":
    main()
