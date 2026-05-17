#!/usr/bin/env python3
"""Create a damped W4A8 alpha overlay from an existing alpha directory.

Structured variants v6+ are cheap layer-selection experiments over the
`structured_v3_minimal` alpha tensors. This utility adds another conservative
knob without retraining:

    alpha_damped = 1 + damping * (alpha - 1)

`damping=1` preserves the source overlay; smaller values pull the folded down
weights closer to the BF16 source model.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch


LAYER_RE = re.compile(r"layer_(\d+)_.*_alpha\.pt$")


def _layer_from_path(path: Path) -> int:
    match = LAYER_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse layer from {path.name}")
    return int(match.group(1))


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.float()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
    }


def _load_source_files(source_alpha_dir: Path, layers: list[int] | None) -> list[Path]:
    by_layer: dict[int, Path] = {}
    for path in sorted(source_alpha_dir.glob("layer_*_alpha.pt")):
        by_layer[_layer_from_path(path)] = path
    if not by_layer:
        raise FileNotFoundError(f"no layer_*_alpha.pt files found under {source_alpha_dir}")
    if layers is None:
        return [by_layer[layer] for layer in sorted(by_layer)]
    missing = [layer for layer in layers if layer not in by_layer]
    if missing:
        raise FileNotFoundError(f"missing requested layers in {source_alpha_dir}: {missing}")
    return [by_layer[layer] for layer in layers]


def build_overlay(
    *,
    source_alpha_dir: Path,
    out_alpha_dir: Path,
    damping: float,
    layers: list[int] | None,
    overwrite: bool,
) -> dict[str, Any]:
    if damping < 0:
        raise ValueError("damping must be non-negative")
    source_files = _load_source_files(source_alpha_dir, layers)
    if out_alpha_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output alpha dir exists: {out_alpha_dir}")
        shutil.rmtree(out_alpha_dir)
    out_alpha_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source_path in source_files:
        payload = torch.load(source_path, map_location="cpu")
        alpha = payload["alpha"].float()
        damped = (1.0 + damping * (alpha - 1.0)).to(alpha.dtype).contiguous()
        out_path = out_alpha_dir / source_path.name
        new_payload = dict(payload)
        new_payload["alpha"] = damped.cpu()
        new_payload["source_alpha_path"] = str(source_path)
        new_payload["damping"] = damping
        new_payload["alpha_min"] = float(damped.min().item())
        new_payload["alpha_max"] = float(damped.max().item())
        torch.save(new_payload, out_path)
        rows.append(
            {
                "layer": _layer_from_path(source_path),
                "source_alpha_path": str(source_path),
                "out_alpha_path": str(out_path),
                "source_stats": _stats(alpha),
                "damped_stats": _stats(damped),
            }
        )

    return {
        "schema_version": "lynn-a100-w4a8-alpha-damped-overlay-v1",
        "source_alpha_dir": str(source_alpha_dir),
        "out_alpha_dir": str(out_alpha_dir),
        "damping": damping,
        "layers": [row["layer"] for row in rows],
        "rows": rows,
        "decision": "GREEN: damped alpha overlay written; fold and run generation gate before promotion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-alpha-dir", required=True)
    parser.add_argument("--out-alpha-dir", required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = build_overlay(
        source_alpha_dir=Path(args.source_alpha_dir),
        out_alpha_dir=Path(args.out_alpha_dir),
        damping=args.damping,
        layers=args.layers,
        overwrite=args.overwrite,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
