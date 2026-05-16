#!/usr/bin/env python3
"""Build a copy-on-write BF16 artifact with W4A8 alpha folded into down_proj.

The P106 alpha overlay is learned as:

    inter_q_corrected = inter_q * alpha[layer, expert, channel]

For runtime, this can be folded into MoE down weights:

    down_weight[expert, hidden, channel] *= alpha[layer, expert, channel]

This script creates a new artifact directory that symlinks unchanged files to
the source model and writes modified safetensors files only for the target
`mlp.experts.down_proj` shards. It never mutates the source model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file


LAYER_RE = re.compile(r"layer_(\d+)_.*_alpha\.pt$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=dst.parent)
    os.symlink(rel, dst)


def _copy_or_link_tree(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        dst = out / child.name
        if child.name == "tensors":
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if child.is_dir():
            if dst.exists():
                continue
            _safe_symlink(child, dst)
        else:
            # Metadata is tiny; copy it so the overlay remains understandable if
            # the source directory is moved later. Tensor shards stay symlinked
            # unless explicitly rewritten below.
            if not dst.exists() and not dst.is_symlink():
                shutil.copy2(child, dst)


def _load_alpha_files(alpha_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(alpha_dir.glob("layer_*_alpha.pt")):
        match = LAYER_RE.match(path.name)
        if not match:
            continue
        out[int(match.group(1))] = path
    if not out:
        raise FileNotFoundError(f"no layer_*_alpha.pt files found under {alpha_dir}")
    return out


def _link_all_tensor_shards(src: Path, out: Path, weight_map: dict[str, str]) -> None:
    for rel in sorted(set(weight_map.values())):
        _safe_symlink(src / rel, out / rel)


def _fold_one_layer(
    *,
    src: Path,
    out: Path,
    weight_map: dict[str, str],
    layer: int,
    alpha_path: Path,
) -> dict[str, Any]:
    key = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing down_proj key for layer {layer}: {key}")
    src_file = src / rel
    out_file = out / rel
    payload = torch.load(alpha_path, map_location="cpu")
    alpha = payload["alpha"].float()
    if alpha.ndim != 2:
        raise ValueError(f"expected expert alpha [E, I] for folding, got {tuple(alpha.shape)} at {alpha_path}")
    tensors = load_file(src_file, device="cpu")
    if key not in tensors:
        raise KeyError(f"{key} not found inside {src_file}")
    weight = tensors[key]
    if weight.ndim != 3:
        raise ValueError(f"expected down_proj [E, hidden, I], got {tuple(weight.shape)}")
    if tuple(alpha.shape) != (int(weight.shape[0]), int(weight.shape[2])):
        raise ValueError(
            f"alpha shape {tuple(alpha.shape)} does not match weight [E,I] "
            f"{(int(weight.shape[0]), int(weight.shape[2]))} for layer {layer}"
        )
    folded = (weight.float() * alpha[:, None, :]).to(weight.dtype).contiguous()
    rewritten = dict(tensors)
    rewritten[key] = folded
    if out_file.exists() or out_file.is_symlink():
        out_file.unlink()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(rewritten, out_file, metadata={"lynn_w4a8_alpha_folded": "true", "source": str(src_file)})
    return {
        "layer": layer,
        "key": key,
        "relative_file": rel,
        "alpha_path": str(alpha_path),
        "weight_shape": list(weight.shape),
        "alpha_shape": list(alpha.shape),
        "alpha_min": float(alpha.min().item()),
        "alpha_max": float(alpha.max().item()),
        "alpha_mean": float(alpha.mean().item()),
        "alpha_std": float(alpha.std().item()),
        "written_bytes": out_file.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model", required=True)
    parser.add_argument("--alpha-dir", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src_model).resolve()
    alpha_dir = Path(args.alpha_dir).resolve()
    out = Path(args.out_model).resolve()
    if out.exists() and not args.overwrite:
        raise SystemExit(f"output already exists: {out} (pass --overwrite to update modified shards)")
    if out.exists() and args.overwrite:
        # Only remove prior symlink/copy-on-write overlay, never the source.
        if src == out:
            raise SystemExit("refusing to overwrite source model")
        shutil.rmtree(out)

    index = _read_json(src / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    alpha_files = _load_alpha_files(alpha_dir)
    _copy_or_link_tree(src, out)
    _link_all_tensor_shards(src, out, weight_map)

    folded = [
        _fold_one_layer(src=src, out=out, weight_map=weight_map, layer=layer, alpha_path=alpha_path)
        for layer, alpha_path in sorted(alpha_files.items())
    ]
    manifest = {
        "schema_version": "lynn-a100-w4a8-alpha-folded-artifact-v1",
        "source_model": str(src),
        "alpha_dir": str(alpha_dir),
        "out_model": str(out),
        "folding_rule": "down_proj[expert, :, channel] *= alpha[layer, expert, channel]",
        "folded_layers": [x["layer"] for x in folded],
        "folded": folded,
    }
    (out / "lynn_w4a8_alpha_fold_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
