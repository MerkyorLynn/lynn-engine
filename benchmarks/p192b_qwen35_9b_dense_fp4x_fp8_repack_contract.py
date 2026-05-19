#!/usr/bin/env python3
"""P192-B · Dense FP4xFP8 repack sidecar contract.

Validates that the P192 sidecar is a bitwise-faithful repack of the original
model weights, with correct pretransposed layouts and scale metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.nvfp4_runtime import _compact_scale_to_swizzled_fp8  # noqa: E402


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


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing weight_map entry for {key}")
    with safe_open(model_dir / rel, framework="pt", device="cpu") as st:
        return st.get_tensor(key)


def _find_key(weight_map: dict[str, str], *candidates: str) -> str:
    for c in candidates:
        if c in weight_map:
            return c
    raise KeyError(f"none of {candidates!r} found in weight_map")


def _load_keys_from_lynn_manifest(
    model_dir: Path, base: str
) -> tuple[str, str, str]:
    manifest = _read_json(model_dir / "lynn_quant_manifest.json")
    rec = manifest.get("quantized_tensors", {}).get(f"{base}.weight")
    if rec is None:
        raise KeyError(f"{base}.weight not found in lynn_quant_manifest.json")
    return rec["packed_key"], rec["scale_key"], rec["global_scale_key"]


def _eq(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    same_shape = tuple(a.shape) == tuple(b.shape)
    same_dtype = a.dtype == b.dtype
    same = same_shape and same_dtype and torch.equal(a.cpu(), b.cpu())
    return {
        "ok": bool(same),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype).replace("torch.", ""),
        "dtype_b": str(b.dtype).replace("torch.", ""),
    }


def _pretranspose_ref(w_packed: torch.Tensor) -> torch.Tensor:
    """Reference pretransposed layout.

    Byte-level transpose is sufficient because each uint8 stores an x2 FP4
    pair along K.  PyTorch cannot CPU-copy float4_e2m1fn_x2, so we avoid
    the dtype view entirely for the reference.
    """
    if w_packed.dtype is not torch.uint8 or w_packed.ndim != 2:
        raise ValueError(f"expected uint8 [N, K/2] packed weight, got {w_packed.shape} {w_packed.dtype}")
    return w_packed.t().contiguous()


def check_layer(model_dir: Path, sidecar_dir: Path, layer: int, prefix: str) -> dict[str, Any]:
    weight_map = _read_json(model_dir / "model.safetensors.index.json")["weight_map"]
    manifest = _read_json(sidecar_dir / "repack_manifest.json")
    layer_rec = manifest.get("layers", {}).get(str(layer))
    if layer_rec is None or "error" in layer_rec:
        return {"layer": layer, "ok": False, "error": f"missing or failed layer record: {layer_rec}"}

    side = load_file(str(sidecar_dir / layer_rec["file"]), device="cpu")
    checks: dict[str, Any] = {}

    for proj in ("gate_proj", "up_proj", "down_proj"):
        base = f"{prefix}.mlp.{proj}"
        alt_base = base.replace("model.language_model.", "")
        try:
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
            packed_key, scale_key, global_key = _load_keys_from_lynn_manifest(
                model_dir, base
            )

        orig_packed = _load_tensor(model_dir, weight_map, packed_key)
        orig_scale = _load_tensor(model_dir, weight_map, scale_key).float()
        orig_global = _load_tensor(model_dir, weight_map, global_key).float().reshape(())

        n, k_half = orig_packed.shape
        k = k_half * 2

        # 1) packed identical
        checks[f"{proj}.weight_packed"] = _eq(side[f"{proj}.weight_packed"], orig_packed)

        # 2) pretransposed layout matches reference
        ref_t = _pretranspose_ref(orig_packed)
        checks[f"{proj}.weight_t_packed"] = _eq(side[f"{proj}.weight_t_packed"], ref_t)
        #   and is contiguous
        w_t = side[f"{proj}.weight_t_packed"]
        checks[f"{proj}.weight_t_packed.contiguous"] = {
            "ok": w_t.is_contiguous(),
            "shape": list(w_t.shape),
            "stride": list(w_t.stride()),
        }

        # 3) scale identical
        checks[f"{proj}.weight_scale"] = _eq(side[f"{proj}.weight_scale"], orig_scale)
        checks[f"{proj}.weight_global_scale"] = _eq(
            side[f"{proj}.weight_global_scale"].float().reshape(()),
            orig_global,
        )

        # 4) native swizzled scale matches reference
        if f"{proj}.scale_b_native" in side:
            ref_native = _compact_scale_to_swizzled_fp8(orig_scale, outer_dim=n, k=k)
            checks[f"{proj}.scale_b_native"] = _eq(
                side[f"{proj}.scale_b_native"].view(torch.float8_e4m3fn),
                ref_native,
            )

    failed = [name for name, rec in checks.items() if not rec["ok"]]
    return {
        "layer": layer,
        "ok": not failed,
        "failed": failed,
        "checks": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P192-B: dense FP4xFP8 repack contract")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--layers", default="0,8,16,-1", help="Layers to validate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    sidecar_dir = Path(args.sidecar_dir)
    config = _read_json(model_dir / "config.json")
    n_layers = _num_hidden_layers(config)

    # Parse layer spec
    layers: list[int] = []
    for raw in args.layers.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx = n_layers + idx
        if 0 <= idx < n_layers and idx not in layers:
            layers.append(idx)

    results = []
    t0 = time.time()
    for layer in layers:
        arch = config.get("architectures", [""])[0]
        if "Qwen2" in arch or "Qwen3" in arch:
            prefix = f"model.language_model.layers.{layer}"
        else:
            prefix = f"model.layers.{layer}"
        try:
            rec = check_layer(model_dir, sidecar_dir, layer, prefix)
            results.append(rec)
            status = "GREEN" if rec["ok"] else "RED"
            print(f"  L{layer:02d}: {status} ({len(rec.get('failed', []))} failed)")
        except Exception as exc:
            results.append({"layer": layer, "ok": False, "error": str(exc)})
            print(f"  L{layer:02d}: ERROR -> {exc}")

    report = {
        "schema": "lynn-qwen35-9b-dense-fp4x-fp8-repack-contract-v1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_dir": str(model_dir),
        "sidecar_dir": str(sidecar_dir),
        "layers_checked": layers,
        "elapsed_seconds": time.time() - t0,
        "results": results,
        "overall": "GREEN" if all(r.get("ok") for r in results) else "RED",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport: {out}")
    print(f"Overall: {report['overall']}")
    return 0 if report["overall"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
