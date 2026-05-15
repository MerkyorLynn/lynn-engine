#!/usr/bin/env python3
"""P9-B: packed-resident coverage and memory ownership plan.

The current Lynn-native NVFP4 runner slow-dequants quantized tensors into BF16
resident tensors at load time. That is correct and already fast enough for the
P7/P8 serving path, but it leaves a large memory advantage unused.

This probe is intentionally static: it reads `lynn_quant_manifest.json` and
answers three questions before we change loader ownership:

1. Which hot-path tensor groups dominate BF16 resident memory?
2. How much GPU memory could packed-resident ownership release?
3. Which groups should be converted first without disturbing the stable BF16
   prefill path?
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any


LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.")


@dataclass
class Bucket:
    count: int = 0
    current_bf16_bytes: int = 0
    packed_payload_bytes: int = 0
    native_scale_bytes: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, key: str, bf16: int, packed: int, native_scale: int) -> None:
        self.count += 1
        self.current_bf16_bytes += int(bf16)
        self.packed_payload_bytes += int(packed)
        self.native_scale_bytes += int(native_scale)
        if len(self.examples) < 5:
            self.examples.append(key)


def _gib(n: int) -> float:
    return n / (1024**3)


def _rows_k(shape: list[int]) -> tuple[int, int]:
    if len(shape) < 2:
        return 0, 0
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    return rows, int(shape[-1])


def _packed_bytes(shape: list[int]) -> int:
    """Approximate packed GPU bytes for Lynn variable NVFP4 row/group layout."""
    rows, k = _rows_k(shape)
    if rows <= 0 or k <= 0:
        return 0
    packed = rows * ((k + 1) // 2)  # uint8, two fp4 values per byte
    scale = rows * ((k + 15) // 16) * 2  # fp16 scale per 16 columns
    global_scale = 4  # float32 scalar
    return packed + scale + global_scale


def _native_scale_bytes(shape: list[int]) -> int:
    """Extra swizzled scale_b bytes if torch._scaled_mm native path prewarms.

    P9-A showed native scale preparation can reduce first-token overhead but is
    still slower than the BF16 resident baseline. We track this separately so
    packed-resident memory accounting can choose whether to pay it.
    """
    rows, k = _rows_k(shape)
    if rows <= 0 or k <= 0:
        return 0
    padded_rows = max(rows, 128)
    padded_groups = max((k + 15) // 16, 4)
    return padded_rows * padded_groups  # fp8 scale_b, one byte each


def _classify(key: str) -> str:
    if key in {"lm_head.weight", "model.language_model.embed_tokens.weight"}:
        return "outside.embed_or_lm_head"
    if ".input_layernorm." in key or ".post_attention_layernorm." in key or key.endswith(".norm.weight"):
        return "norms"
    if ".linear_attn." in key:
        if ".in_proj_qkv." in key:
            return "linear_attn.in_proj_qkv"
        if ".in_proj_z." in key:
            return "linear_attn.in_proj_z"
        if ".in_proj_b." in key or ".in_proj_a." in key:
            return "linear_attn.in_proj_ba"
        if ".out_proj." in key:
            return "linear_attn.out_proj"
        return "linear_attn.other"
    if ".self_attn." in key:
        if ".q_proj." in key or ".k_proj." in key or ".v_proj." in key:
            return "full_attn.qkv_proj"
        if ".o_proj." in key:
            return "full_attn.o_proj"
        if ".q_norm." in key or ".k_norm." in key:
            return "full_attn.norms"
        return "full_attn.other"
    if ".mlp.experts.gate_up_proj" in key:
        return "moe.experts.gate_up"
    if ".mlp.experts.down_proj" in key:
        return "moe.experts.down"
    if ".mlp.shared_expert." in key:
        return "moe.shared_expert"
    if ".mlp.gate." in key:
        return "moe.router"
    if ".mtp." in key or "mtp" in key:
        return "mtp"
    return "other"


def _layer_type(config: dict[str, Any], key: str) -> str:
    m = LAYER_RE.search(key)
    if not m:
        return "outside"
    layer_idx = int(m.group(1))
    try:
        return str(config["text_config"]["layer_types"][layer_idx])
    except Exception:
        return "unknown"


def _priority(bucket: str) -> str:
    if bucket.startswith("linear_attn."):
        return "P9-C/P10: high memory share; needs fused block kernel before default enable"
    if bucket.startswith("full_attn."):
        return "P9-A/P9-B: bridge already validated; keep opt-in until fused path wins"
    if bucket.startswith("moe.experts."):
        return "P10: large memory share; convert after packed MoE fused expert path"
    if bucket == "moe.shared_expert":
        return "P10: lower count but always active; good fused-kernel candidate"
    if bucket in {"outside.embed_or_lm_head", "moe.router", "norms"}:
        return "keep BF16 for now; small or numerically sensitive"
    return "defer"


def build_report(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / "lynn_quant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((model_dir / "config.json").read_text())

    buckets: dict[str, Bucket] = defaultdict(Bucket)
    layer_types: dict[str, Bucket] = defaultdict(Bucket)
    by_layer: dict[str, Bucket] = defaultdict(Bucket)

    for key, rec in manifest.get("quantized_tensors", {}).items():
        shape = [int(x) for x in rec["original_shape"]]
        bf16 = int(rec.get("input_bytes") or (2 * _rows_k(shape)[0] * _rows_k(shape)[1]))
        packed = _packed_bytes(shape)
        native_scale = _native_scale_bytes(shape)
        bucket = _classify(key)
        ltype = _layer_type(config, key)
        layer_match = LAYER_RE.search(key)
        layer_key = f"layer_{int(layer_match.group(1)):02d}" if layer_match else "outside"
        buckets[bucket].add(key, bf16, packed, native_scale)
        layer_types[ltype].add(key, bf16, packed, native_scale)
        by_layer[layer_key].add(key, bf16, packed, native_scale)

    kept_bytes = sum(int(x.get("bytes", 0)) for x in manifest.get("kept_tensors", {}).values())
    quant_bf16_bytes = sum(b.current_bf16_bytes for b in buckets.values())
    quant_packed_bytes = sum(b.packed_payload_bytes for b in buckets.values())
    quant_native_scale_bytes = sum(b.native_scale_bytes for b in buckets.values())

    def render_bucket(name: str, b: Bucket) -> dict[str, Any]:
        release = b.current_bf16_bytes - b.packed_payload_bytes
        release_native = b.current_bf16_bytes - b.packed_payload_bytes - b.native_scale_bytes
        return {
            "name": name,
            "count": b.count,
            "current_bf16_gib": round(_gib(b.current_bf16_bytes), 4),
            "packed_payload_gib": round(_gib(b.packed_payload_bytes), 4),
            "native_scale_extra_gib": round(_gib(b.native_scale_bytes), 4),
            "releasable_gib_without_native_scale": round(_gib(release), 4),
            "releasable_gib_with_native_scale": round(_gib(release_native), 4),
            "priority": _priority(name),
            "examples": b.examples,
        }

    bucket_rows = [render_bucket(k, v) for k, v in buckets.items()]
    bucket_rows.sort(key=lambda x: x["current_bf16_gib"], reverse=True)

    layer_type_rows = [render_bucket(k, v) for k, v in layer_types.items()]
    layer_type_rows.sort(key=lambda x: x["current_bf16_gib"], reverse=True)

    layer_rows = [render_bucket(k, v) for k, v in by_layer.items()]
    layer_rows.sort(key=lambda x: x["name"])

    return {
        "schema_version": "lynn-engine-p9b-packed-resident-coverage-v1",
        "model_dir": str(model_dir),
        "manifest_schema": manifest.get("schema_version"),
        "counts": {
            "quantized_tensors": len(manifest.get("quantized_tensors", {})),
            "kept_tensors": len(manifest.get("kept_tensors", {})),
        },
        "summary_gib": {
            "current_quantized_bf16_resident": round(_gib(quant_bf16_bytes), 4),
            "packed_payload": round(_gib(quant_packed_bytes), 4),
            "native_scale_extra_if_prepared": round(_gib(quant_native_scale_bytes), 4),
            "kept_bf16_resident": round(_gib(kept_bytes), 4),
            "total_current_bf16_plus_kept": round(_gib(quant_bf16_bytes + kept_bytes), 4),
            "total_packed_plus_kept": round(_gib(quant_packed_bytes + kept_bytes), 4),
            "releasable_without_native_scale": round(_gib(quant_bf16_bytes - quant_packed_bytes), 4),
            "releasable_with_native_scale": round(
                _gib(quant_bf16_bytes - quant_packed_bytes - quant_native_scale_bytes), 4
            ),
        },
        "by_bucket": bucket_rows,
        "by_layer_type": layer_type_rows,
        "by_layer": layer_rows,
        "recommended_plan": [
            "Keep current BF16-resident P7/P8 path as default until packed path beats 68 TPS.",
            "P9-B: add loader memory accounting and packed ownership toggles, not global default switches.",
            "P9-C: attack linear-attn as fused block, because it owns the largest memory share and previous per-projection swaps lost speed.",
            "P10: move MoE expert tensors to packed-resident only with a fused active-expert kernel; per-expert scalar/native calls are not enough.",
            "Keep embeddings, lm_head, routers, norms in BF16 until quality and memory accounting say otherwise.",
        ],
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P9-B Packed-Resident Coverage",
        "",
        f"Model: `{report['model_dir']}`",
        "",
        "## Summary",
        "",
    ]
    for k, v in report["summary_gib"].items():
        lines.append(f"- `{k}`: **{v} GiB**")
    lines += [
        "",
        "## Hot-Path Buckets",
        "",
        "| Bucket | Count | Current BF16 GiB | Packed GiB | Native scale GiB | Releasable GiB | Priority |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["by_bucket"]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {row['current_bf16_gib']} | "
            f"{row['packed_payload_gib']} | {row['native_scale_extra_gib']} | "
            f"{row['releasable_gib_with_native_scale']} | {row['priority']} |"
        )
    lines += [
        "",
        "## Layer Type Split",
        "",
        "| Layer type | Count | Current BF16 GiB | Packed GiB | Releasable GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["by_layer_type"]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {row['current_bf16_gib']} | "
            f"{row['packed_payload_gib']} | {row['releasable_gib_with_native_scale']} |"
        )
    lines += [
        "",
        "## Recommended Plan",
        "",
    ]
    for item in report["recommended_plan"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md-out")
    args = ap.parse_args()

    report = build_report(Path(args.model))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        md_out = Path(args.md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, md_out)
    print(json.dumps(report["summary_gib"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
