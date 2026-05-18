#!/usr/bin/env python3
"""Create a MoE-first serving-layout sidecar for Qwen3.6 W4A16 NVFP4.

This is the first offline repack step toward a Lynn-native CUDA MoE boundary.
It does not change model math.  It co-locates per-layer router, active experts,
and shared expert tensors in a stable expert-major layout so a later kernel can
consume one layer-local record instead of chasing generic manifest keys.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


SCHEMA_VERSION = "qwen36-w4a16-moe-repack-sidecar-v1"


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


def _reshape_grouped(tensor: torch.Tensor, original_shape: list[int], divisor: int) -> torch.Tensor:
    if len(original_shape) != 3:
        raise ValueError(f"expected grouped [experts,out,in] original shape, got {original_shape}")
    experts, out_features, in_features = map(int, original_shape)
    return tensor.reshape(experts, out_features, in_features // divisor).contiguous()


def _reshape_grouped_scale(tensor: torch.Tensor, original_shape: list[int]) -> torch.Tensor:
    return _reshape_grouped(tensor, original_shape, 16)


def _reshape_grouped_packed(tensor: torch.Tensor, original_shape: list[int]) -> torch.Tensor:
    return _reshape_grouped(tensor, original_shape, 2)


def _load_quantized_triplet(
    model_dir: Path,
    weight_map: dict[str, str],
    rec: dict[str, Any],
    *,
    grouped: bool,
    fold_global_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    packed = _load_tensor(model_dir, weight_map, rec["packed_key"])
    scale = _load_tensor(model_dir, weight_map, rec["scale_key"])
    global_scale = _load_tensor(model_dir, weight_map, rec["global_scale_key"]).float().reshape(())
    original_shape = [int(x) for x in rec["original_shape"]]
    if grouped:
        packed = _reshape_grouped_packed(packed, original_shape)
        scale = _reshape_grouped_scale(scale, original_shape)
    if fold_global_scale:
        scale = (scale.float() / global_scale.float()).contiguous()
        global_scale = torch.ones_like(global_scale.float())
    info = {
        "original_shape": original_shape,
        "packed_shape": list(packed.shape),
        "scale_shape": list(scale.shape),
        "original_dtype": rec.get("original_dtype"),
        "packed_key": rec["packed_key"],
        "scale_key": rec["scale_key"],
        "global_scale_key": rec["global_scale_key"],
        "global_scale_folded": fold_global_scale,
    }
    return packed.contiguous(), scale.contiguous(), global_scale.contiguous(), info


def _layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


def _layer_tensors(
    model_dir: Path,
    weight_map: dict[str, str],
    manifest: dict[str, Any],
    layer: int,
    *,
    include_shared: bool,
    include_router: bool,
    fold_active_global_scale: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prefix = _layer_prefix(layer)
    active_gateup_key = f"{prefix}.mlp.experts.gate_up_proj"
    active_down_key = f"{prefix}.mlp.experts.down_proj"

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, Any] = {
        "layer": layer,
        "layout": {
            "active_moe": "expert_major:[experts,out_features,packed_input_or_scale_group]",
            "shared_moe": "row_major:[out_features,packed_input_or_scale_group]",
            "router": "bf16/fp32 row_major:[experts,hidden]",
        },
        "source_keys": {},
        "tensors": {},
    }

    for short, source_key in (
        ("active_gate_up", active_gateup_key),
        ("active_down", active_down_key),
    ):
        packed, scale, global_scale, info = _load_quantized_triplet(
            model_dir,
            weight_map,
            _rec(manifest, source_key),
            grouped=True,
            fold_global_scale=fold_active_global_scale,
        )
        tensors[f"{short}.packed"] = packed
        tensors[f"{short}.scale"] = scale
        tensors[f"{short}.global_scale"] = global_scale
        meta["source_keys"][short] = source_key
        meta["tensors"][short] = info

    if include_router:
        router_key = f"{prefix}.mlp.gate.weight"
        tensors["router.weight"] = _load_tensor(model_dir, weight_map, router_key)
        meta["source_keys"]["router"] = router_key
        meta["tensors"]["router"] = {
            "shape": list(tensors["router.weight"].shape),
            "dtype": str(tensors["router.weight"].dtype).replace("torch.", ""),
        }

    if include_shared:
        shared_map = {
            "shared_gate": f"{prefix}.mlp.shared_expert.gate_proj.weight",
            "shared_up": f"{prefix}.mlp.shared_expert.up_proj.weight",
            "shared_down": f"{prefix}.mlp.shared_expert.down_proj.weight",
            "shared_scalar_gate": f"{prefix}.mlp.shared_expert_gate.weight",
        }
        for short, source_key in shared_map.items():
            rec = _rec(manifest, source_key)
            packed, scale, global_scale, info = _load_quantized_triplet(
                model_dir,
                weight_map,
                rec,
                grouped=False,
            )
            tensors[f"{short}.packed"] = packed
            tensors[f"{short}.scale"] = scale
            tensors[f"{short}.global_scale"] = global_scale
            meta["source_keys"][short] = source_key
            meta["tensors"][short] = info

    return tensors, meta


def _validate_layer(path: Path, expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    loaded = load_file(path, device="cpu")
    missing = sorted(set(expected) - set(loaded))
    extra = sorted(set(loaded) - set(expected))
    mismatched: list[str] = []
    for key, tensor in expected.items():
        got = loaded.get(key)
        if got is None:
            continue
        if tuple(got.shape) != tuple(tensor.shape) or got.dtype != tensor.dtype:
            mismatched.append(key)
            continue
        if tensor.numel() and not torch.equal(got.reshape(-1)[:16], tensor.reshape(-1)[:16]):
            mismatched.append(key)
    return {
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "ok": not missing and not extra and not mismatched,
    }


def _parse_layers(raw: str | None) -> list[int]:
    if not raw:
        return list(range(40))
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
    bad = [x for x in layers if x < 0 or x >= 40]
    if bad:
        raise ValueError(f"language layer ids must be in [0,39], got {bad}")
    return sorted(layers)


def build_sidecar(
    model_dir: Path,
    out_dir: Path,
    *,
    layers: list[int],
    include_shared: bool,
    include_router: bool,
    fold_active_global_scale: bool,
    overwrite: bool,
    validate: bool,
) -> dict[str, Any]:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(model_dir / "lynn_quant_manifest.json")
    weight_map = _read_json(model_dir / "model.safetensors.index.json")["weight_map"]

    start = time.time()
    sidecar: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_model": str(model_dir),
        "out_dir": str(out_dir),
        "layers": {},
        "layer_count": len(layers),
        "include_shared": include_shared,
        "include_router": include_router,
        "layout_contract": {
            "activation_contract": "W4A16_weight_only",
            "active_moe_layout": "expert_major_3d",
            "shared_moe_layout": "row_major_2d",
            "active_scale_contract": (
                "effective_scale_global_one"
                if fold_active_global_scale
                else "scale_div_global"
            ),
            "math_order": "unchanged: router -> topk -> gate/up -> bf16 inter -> down -> weighted sum -> shared add",
        },
    }

    for layer in layers:
        layer_start = time.time()
        tensors, meta = _layer_tensors(
            model_dir,
            weight_map,
            manifest,
            layer,
            include_shared=include_shared,
            include_router=include_router,
            fold_active_global_scale=fold_active_global_scale,
        )
        rel = f"moe_layer_{layer:02d}.safetensors"
        path = out_dir / rel
        save_file(
            tensors,
            path,
            metadata={
                "format": "pt",
                "schema_version": SCHEMA_VERSION,
                "layer": str(layer),
            },
        )
        validation = _validate_layer(path, tensors) if validate else {"ok": None}
        if validation.get("ok") is False:
            raise RuntimeError(f"validation failed for layer {layer}: {validation}")
        layer_bytes = path.stat().st_size
        sidecar["layers"][str(layer)] = {
            "file": rel,
            "bytes": layer_bytes,
            "mib": round(layer_bytes / (1024**2), 3),
            "elapsed_seconds": round(time.time() - layer_start, 3),
            "validation": validation,
            **meta,
        }
        print(
            f"[moe-repack] layer={layer:02d} file={rel} size={layer_bytes/(1024**2):.1f} MiB "
            f"elapsed={time.time()-layer_start:.1f}s",
            flush=True,
        )

    total_bytes = sum(int(x["bytes"]) for x in sidecar["layers"].values())
    sidecar["total_bytes"] = total_bytes
    sidecar["total_gib"] = round(total_bytes / (1024**3), 4)
    sidecar["elapsed_seconds"] = round(time.time() - start, 3)
    (out_dir / "moe_repack_manifest.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--layers", help="comma/range layer selector, default all 0-39")
    ap.add_argument("--no-shared", action="store_true", help="exclude shared expert tensors")
    ap.add_argument("--no-router", action="store_true", help="exclude router gate weight")
    ap.add_argument(
        "--fold-active-global-scale",
        action="store_true",
        help="store active MoE scales as scale/global_scale and write global_scale=1 for native kernels",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    sidecar = build_sidecar(
        args.model_dir,
        args.out_dir,
        layers=_parse_layers(args.layers),
        include_shared=not args.no_shared,
        include_router=not args.no_router,
        fold_active_global_scale=args.fold_active_global_scale,
        overwrite=args.overwrite,
        validate=not args.no_validate,
    )
    print(
        json.dumps(
            {
                "out_dir": sidecar["out_dir"],
                "layer_count": sidecar["layer_count"],
                "total_gib": sidecar["total_gib"],
                "elapsed_seconds": sidecar["elapsed_seconds"],
                "manifest": str(Path(sidecar["out_dir"]) / "moe_repack_manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
