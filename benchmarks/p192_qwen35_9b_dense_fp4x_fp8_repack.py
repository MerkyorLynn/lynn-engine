#!/usr/bin/env python3
"""P192 · Qwen3.5-9B dense FP4xFP8 offline repack sidecar.

Reads the native W4A16 NVFP4 model, extracts gate/up/down for each dense layer,
and writes kernel-ready pretransposed E2M1 packed layouts plus scale metadata
into a sidecar directory.

Output layout per layer (safetensors):
  gate_proj.weight_packed      uint8   [N, K/2]
  gate_proj.weight_t_packed    uint8   [K/2, N]  (pretransposed contiguous)
  gate_proj.weight_scale       float32 [N, K/16]
  gate_proj.weight_global_scale float32 scalar
  gate_proj.scale_b_native     uint8   [rows*groups] (float8_e4m3fn viewed as uint8)
  ... same for up_proj, down_proj

Manifest (JSON):
  schema_version: qwen35-9b-dense-fp4x-fp8-repack-v1
  maps layer_id -> file, sha256, tensor metadata (shape, stride, dtype, storage_dtype)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = "qwen35-9b-dense-fp4x-fp8-repack-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text_cfg = cfg.get("text_config")
    return text_cfg if isinstance(text_cfg, dict) else cfg


def _num_hidden_layers(cfg: dict[str, Any]) -> int:
    text_cfg = _text_config(cfg)
    for key in ("num_hidden_layers", "n_layers", "num_layers"):
        value = text_cfg.get(key)
        if value is not None:
            return int(value)
    raise ValueError("could not determine num_hidden_layers from config.json/text_config")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing weight_map entry for {key}")
    with safe_open(model_dir / rel, framework="pt", device="cpu") as st:
        return st.get_tensor(key)


def _pretranspose_packed(w_packed: torch.Tensor) -> torch.Tensor:
    """Return pretransposed E2M1 packed layout as contiguous uint8 [K/2, N].

    Each uint8 element stores an x2 FP4 pair along K.  The kernel-side
    pretranspose is therefore a byte-element transpose from [N, K/2] to
    [K/2, N]; using torch.float4_e2m1fn_x2 for the view would describe the
    same element layout, but PyTorch cannot CPU-copy that dtype today.
    """
    if w_packed.dtype is not torch.uint8 or w_packed.ndim != 2:
        raise ValueError(f"expected uint8 [N, K/2] packed weight, got {w_packed.shape} {w_packed.dtype}")
    return w_packed.t().contiguous()


def _native_scale_b(weight_scale: torch.Tensor, outer_dim: int, k: int) -> torch.Tensor:
    """Compute swizzled FP8 scale_b for torch._scaled_mm FP4 path.

    Returns uint8 view of float8_e4m3fn tensor so safetensors can store it.
    """
    from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8
    scale_native = _compact_scale_to_swizzled_fp8(
        weight_scale.float(), outer_dim=outer_dim, k=k
    )
    return scale_native.view(torch.uint8).contiguous()


def _layer_prefix(layer: int, cfg: dict[str, Any]) -> str:
    arch = cfg.get("architectures", [""])[0]
    if "Qwen2" in arch or "Qwen3" in arch:
        return f"model.language_model.layers.{layer}"
    return f"model.layers.{layer}"


def _find_key(weight_map: dict[str, str], *candidates: str) -> str:
    for c in candidates:
        if c in weight_map:
            return c
    raise KeyError(f"none of {candidates!r} found in weight_map")


def _load_keys_from_lynn_manifest(
    model_dir: Path, base: str
) -> tuple[str, str, str]:
    """Use lynn_quant_manifest.json to resolve packed/scale/global_scale keys."""
    manifest = _read_json(model_dir / "lynn_quant_manifest.json")
    rec = manifest.get("quantized_tensors", {}).get(f"{base}.weight")
    if rec is None:
        raise KeyError(f"{base}.weight not found in lynn_quant_manifest.json")
    return rec["packed_key"], rec["scale_key"], rec["global_scale_key"]


def repack_layer(
    model_dir: Path,
    weight_map: dict[str, str],
    layer: int,
    prefix: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Repack one dense layer and return manifest record."""
    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: dict[str, Any] = {}

    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.mlp.{proj}"
        alt_base = base.replace("model.language_model.", "")
        try:
            # Lynn-native dot naming first, then v8-RTN underscore naming
            packed_key = _find_key(
                weight_map,
                f"{base}.weight.packed",
                f"{alt_base}.weight.packed",
                f"{base}.weight_packed",
                f"{alt_base}.weight_packed",
            )
            scale_key = _find_key(
                weight_map,
                f"{base}.weight.scale",
                f"{alt_base}.weight.scale",
                f"{base}.weight_scale",
                f"{alt_base}.weight_scale",
            )
            global_key = _find_key(
                weight_map,
                f"{base}.weight.global_scale",
                f"{alt_base}.weight.global_scale",
                f"{base}.weight_global_scale",
                f"{alt_base}.weight_global_scale",
            )
        except KeyError:
            # Fallback to lynn_quant_manifest.json
            packed_key, scale_key, global_key = _load_keys_from_lynn_manifest(
                model_dir, base
            )

        w_packed = _load_tensor(model_dir, weight_map, packed_key)
        w_scale = _load_tensor(model_dir, weight_map, scale_key).float()
        w_global = _load_tensor(model_dir, weight_map, global_key).float().reshape(())

        n, k_half = w_packed.shape
        k = k_half * 2

        # Original packed layout
        tensors[f"{proj}.weight_packed"] = w_packed.contiguous()
        tensor_meta[f"{proj}.weight_packed"] = {
            "shape": list(w_packed.shape),
            "stride": list(w_packed.stride()),
            "dtype": str(w_packed.dtype).replace("torch.", ""),
            "storage_dtype": "uint8",
        }

        # Pretransposed layout
        w_t = _pretranspose_packed(w_packed)
        if w_t is not None:
            tensors[f"{proj}.weight_t_packed"] = w_t
            tensor_meta[f"{proj}.weight_t_packed"] = {
                "shape": list(w_t.shape),
                "stride": list(w_t.stride()),
                "dtype": "uint8",
                "storage_dtype": "uint8",
                "note": "contiguous byte transpose of [N, K/2] packed E2M1 x2 elements",
            }

        # Scale metadata
        tensors[f"{proj}.weight_scale"] = w_scale.contiguous()
        tensor_meta[f"{proj}.weight_scale"] = {
            "shape": list(w_scale.shape),
            "stride": list(w_scale.stride()),
            "dtype": str(w_scale.dtype).replace("torch.", ""),
        }

        tensors[f"{proj}.weight_global_scale"] = w_global.contiguous()
        tensor_meta[f"{proj}.weight_global_scale"] = {
            "shape": list(w_global.shape),
            "stride": list(w_global.stride()),
            "dtype": str(w_global.dtype).replace("torch.", ""),
        }

        # Native swizzled scale for _scaled_mm
        scale_native = _native_scale_b(w_scale, outer_dim=n, k=k)
        if scale_native is not None:
            tensors[f"{proj}.scale_b_native"] = scale_native
            tensor_meta[f"{proj}.scale_b_native"] = {
                "shape": list(scale_native.shape),
                "stride": list(scale_native.stride()),
                "dtype": "float8_e4m3fn",
                "storage_dtype": "uint8",
                "note": "view as float8_e4m3fn before passing to torch._scaled_mm",
            }

    file_name = f"layer_{layer:02d}.safetensors"
    file_path = out_dir / file_name
    save_file(tensors, str(file_path))
    sha256 = _sha256_file(file_path)

    return {
        "file": file_name,
        "sha256": sha256,
        "tensors": tensor_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P192: dense FP4xFP8 offline repack sidecar")
    ap.add_argument("--model-dir", required=True, help="Qwen3.5-9B NVFP4 model directory")
    ap.add_argument("--out-dir", required=True, help="Sidecar output directory")
    ap.add_argument("--layers", default="", help="Comma-separated layer ids (default: all)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model metadata
    config = _read_json(model_dir / "config.json")
    n_layers = _num_hidden_layers(config)

    weight_map = _read_json(model_dir / "model.safetensors.index.json")["weight_map"]

    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers = list(range(n_layers))

    # Source model integrity hashes
    index_path = model_dir / "model.safetensors.index.json"
    lynn_manifest_path = model_dir / "lynn_quant_manifest.json"
    source_hashes = {
        "config_sha256": _sha256_file(model_dir / "config.json"),
        "index_sha256": _sha256_file(index_path) if index_path.exists() else None,
    }
    if lynn_manifest_path.exists():
        source_hashes["lynn_manifest_sha256"] = _sha256_file(lynn_manifest_path)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_model": str(model_dir),
        "source_hashes": source_hashes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_layers": n_layers,
        "layers": {},
    }

    t0 = time.time()
    total_bytes = 0
    for layer in layers:
        prefix = _layer_prefix(layer, config)
        try:
            rec = repack_layer(model_dir, weight_map, layer, prefix, out_dir)
            manifest["layers"][str(layer)] = rec
            file_path = out_dir / rec["file"]
            total_bytes += file_path.stat().st_size
            print(f"  L{layer:02d}: OK -> {rec['file']} ({len(rec['tensors'])} tensors)")
        except Exception as exc:
            print(f"  L{layer:02d}: FAIL -> {exc}")
            manifest["layers"][str(layer)] = {"error": str(exc)}

    manifest["elapsed_seconds"] = time.time() - t0
    manifest["sidecar_total_bytes"] = total_bytes
    manifest["sidecar_total_mib"] = round(total_bytes / (1024 * 1024), 2)
    manifest_path = out_dir / "repack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Layers: {len([l for l in manifest['layers'].values() if 'error' not in l])}/{len(layers)} OK")
    print(f"Sidecar size: {manifest['sidecar_total_mib']:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
