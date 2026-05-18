#!/usr/bin/env python3
"""Build a safetensors index overlay for Lynn layer-split artifacts.

Some internal conversion jobs produce one shard per layer, named like
`layers-0.safetensors`, without a Hugging Face `model.safetensors.index.json`.
This helper creates a lightweight overlay directory with relative symlinks to
those shards and a generated index, so generic pack/probe tools can consume the
artifact while preserving a clear non-official provenance marker.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from safetensors import safe_open


def _copy_or_link_metadata(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name.endswith(".safetensors") or child.name == "model.safetensors.index.json":
            continue
        dst = out / child.name
        if dst.exists() or dst.is_symlink():
            continue
        if child.is_dir():
            shutil.copytree(child, dst, symlinks=True)
        else:
            shutil.copy2(child, dst)


def _relative_symlink(src_file: Path, dst_file: Path) -> None:
    if dst_file.exists() or dst_file.is_symlink():
        dst_file.unlink()
    rel = os.path.relpath(src_file, start=dst_file.parent)
    dst_file.symlink_to(rel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--glob", default="layers-*.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src_model).resolve()
    out = Path(args.out_model).resolve()
    if not src.exists():
        raise SystemExit(f"missing source model: {src}")
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {out}")
        if out == src:
            raise SystemExit("refusing to overwrite source model")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.glob))
    if not files:
        raise SystemExit(f"no files matched {args.glob!r} in {src}")

    _copy_or_link_metadata(src, out)

    weight_map: dict[str, str] = {}
    source_files: dict[str, dict[str, Any]] = {}
    for src_file in files:
        dst_file = out / src_file.name
        _relative_symlink(src_file, dst_file)
        with safe_open(src_file, framework="pt", device="cpu") as st:
            keys = list(st.keys())
        for key in keys:
            if key in weight_map:
                raise SystemExit(f"duplicate tensor key {key!r} in {src_file}")
            weight_map[key] = src_file.name
        source_files[src_file.name] = {
            "tensor_count": len(keys),
            "bytes": src_file.stat().st_size,
        }

    total_size = sum(rec["bytes"] for rec in source_files.values())
    index = {
        "metadata": {
            "total_size": total_size,
            "source_layout": "lynn_layer_split",
            "source_model": str(src),
            "official_hf_layout": "false",
        },
        "weight_map": weight_map,
    }
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "schema_version": "lynn-layer-split-index-overlay-v1",
        "src_model": str(src),
        "out_model": str(out),
        "file_count": len(files),
        "tensor_count": len(weight_map),
        "total_size": total_size,
        "source_files": source_files,
        "warning": "derived layer-split artifact; do not treat as canonical official BF16",
    }
    (out / "lynn_layer_split_index_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
