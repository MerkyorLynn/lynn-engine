#!/usr/bin/env python3
"""P127: verify MoE repack sidecar bitwise matches the current manifest path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from engine.moe_repack_sidecar import load_moe_repack_layer, load_moe_repack_manifest
from engine.nvfp4_runtime import load_grouped_nvfp4_weight


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing weight_map entry for {key}")
    with safe_open(model_dir / rel, framework="pt", device="cpu") as st:
        return st.get_tensor(key).contiguous()


def _rec(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    records = manifest.get("quantized_tensors", {})
    if key not in records:
        raise KeyError(f"missing quantized tensor record: {key}")
    return records[key]


def _load_quant_triplet(
    model_dir: Path,
    weight_map: dict[str, str],
    manifest: dict[str, Any],
    key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = _rec(manifest, key)
    return (
        _load_tensor(model_dir, weight_map, rec["packed_key"]),
        _load_tensor(model_dir, weight_map, rec["scale_key"]),
        _load_tensor(model_dir, weight_map, rec["global_scale_key"]).float().reshape(()),
    )


def _eq(a: torch.Tensor, b: torch.Tensor, *, compare_float: bool = False) -> dict[str, Any]:
    same_shape = tuple(a.shape) == tuple(b.shape)
    same_dtype = a.dtype == b.dtype
    if compare_float:
        same = same_shape and torch.equal(a.float().cpu(), b.float().cpu())
    else:
        same = same_shape and same_dtype and torch.equal(a.cpu(), b.cpu())
    return {
        "ok": bool(same),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype).replace("torch.", ""),
        "dtype_b": str(b.dtype).replace("torch.", ""),
        "compare_float": compare_float,
    }


def check_layer(model_dir: Path, sidecar_dir: Path, layer: int) -> dict[str, Any]:
    manifest = _read_json(model_dir / "lynn_quant_manifest.json")
    weight_map = _read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    prefix = f"model.language_model.layers.{layer}"
    t0 = time.time()
    side = load_moe_repack_layer(sidecar_dir, layer, device="cpu")
    side_load_seconds = time.time() - t0

    gateup_packed, gateup_scale, gateup_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.mlp.experts.gate_up_proj", device="cpu"
    )
    down_packed, down_scale, down_global = load_grouped_nvfp4_weight(
        model_dir, f"{prefix}.mlp.experts.down_proj", device="cpu"
    )
    router = _load_tensor(model_dir, weight_map, f"{prefix}.mlp.gate.weight")

    checks = {
        "active_gate_up.packed": _eq(side.tensors["active_gate_up.packed"], gateup_packed),
        "active_gate_up.scale": _eq(side.tensors["active_gate_up.scale"], gateup_scale, compare_float=True),
        "active_gate_up.global_scale": _eq(side.tensors["active_gate_up.global_scale"].float().reshape(()), gateup_global.float().reshape(())),
        "active_down.packed": _eq(side.tensors["active_down.packed"], down_packed),
        "active_down.scale": _eq(side.tensors["active_down.scale"], down_scale, compare_float=True),
        "active_down.global_scale": _eq(side.tensors["active_down.global_scale"].float().reshape(()), down_global.float().reshape(())),
        "router.weight": _eq(side.tensors["router.weight"], router),
    }

    for short, source in {
        "shared_gate": f"{prefix}.mlp.shared_expert.gate_proj.weight",
        "shared_up": f"{prefix}.mlp.shared_expert.up_proj.weight",
        "shared_down": f"{prefix}.mlp.shared_expert.down_proj.weight",
        "shared_scalar_gate": f"{prefix}.mlp.shared_expert_gate.weight",
    }.items():
        packed, scale, global_scale = _load_quant_triplet(model_dir, weight_map, manifest, source)
        checks[f"{short}.packed"] = _eq(side.tensors[f"{short}.packed"], packed)
        checks[f"{short}.scale"] = _eq(side.tensors[f"{short}.scale"], scale)
        checks[f"{short}.global_scale"] = _eq(side.tensors[f"{short}.global_scale"].float().reshape(()), global_scale)

    failed = [name for name, rec in checks.items() if not rec["ok"]]
    return {
        "layer": layer,
        "ok": not failed,
        "failed": failed,
        "sidecar_file": side.metadata["file"],
        "sidecar_mib": side.metadata["mib"],
        "side_load_seconds": side_load_seconds,
        "checks": checks,
    }


def _parse_layers(raw: str) -> list[int]:
    layers: set[int] = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            layers.update(range(int(lo), int(hi) + 1))
        else:
            layers.add(int(item))
    return sorted(layers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--sidecar-dir", required=True, type=Path)
    ap.add_argument("--layers", default="0,20,39")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    sidecar_manifest = load_moe_repack_manifest(args.sidecar_dir)
    rows = [check_layer(args.model_dir, args.sidecar_dir, layer) for layer in _parse_layers(args.layers)]
    report = {
        "schema_version": "p127-moe-repack-sidecar-contract-v1",
        "model_dir": str(args.model_dir),
        "sidecar_dir": str(args.sidecar_dir),
        "sidecar_schema": sidecar_manifest.get("schema_version"),
        "sidecar_total_gib": sidecar_manifest.get("total_gib"),
        "layers": rows,
        "ok": all(row["ok"] for row in rows),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
