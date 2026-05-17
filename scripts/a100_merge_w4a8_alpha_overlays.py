#!/usr/bin/env python3
"""Merge two foldable W4A8 alpha overlay directories.

This is for conservative Recovery composition. A general baseline alpha can be
combined with a more targeted correction without retraining:

    merged = base_alpha * (1 + correction_damping * (correction_alpha - 1))

Shared [I] alpha vectors and expert [E, I] alpha tensors are both supported.
If either input is expert-wise, the output is expert-wise.
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


def _layer(path: Path) -> int:
    match = LAYER_RE.match(path.name)
    if not match:
        raise ValueError(f"cannot parse layer from {path.name}")
    return int(match.group(1))


def _files(alpha_dir: Path) -> dict[int, Path]:
    out = {_layer(path): path for path in sorted(alpha_dir.glob("layer_*_alpha.pt"))}
    if not out:
        raise FileNotFoundError(f"no layer_*_alpha.pt files found under {alpha_dir}")
    return out


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.float()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
    }


def _broadcast_pair(base: torch.Tensor, correction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    base = base.float()
    correction = correction.float()
    if base.ndim == correction.ndim:
        if tuple(base.shape) != tuple(correction.shape):
            raise ValueError(f"alpha shapes do not match: {tuple(base.shape)} vs {tuple(correction.shape)}")
        mode = "expert" if base.ndim == 2 else "shared"
        return base, correction, mode
    if base.ndim == 1 and correction.ndim == 2:
        if int(base.shape[0]) != int(correction.shape[1]):
            raise ValueError(f"shared base {tuple(base.shape)} incompatible with expert correction {tuple(correction.shape)}")
        return base[None, :].expand_as(correction), correction, "expert"
    if base.ndim == 2 and correction.ndim == 1:
        if int(base.shape[1]) != int(correction.shape[0]):
            raise ValueError(f"expert base {tuple(base.shape)} incompatible with shared correction {tuple(correction.shape)}")
        return base, correction[None, :].expand_as(base), "expert"
    raise ValueError(f"expected shared [I] or expert [E, I] alpha, got {tuple(base.shape)} and {tuple(correction.shape)}")


def merge(
    *,
    base_alpha_dir: Path,
    correction_alpha_dir: Path,
    out_alpha_dir: Path,
    correction_damping: float,
    alpha_min: float,
    alpha_max: float,
    layers: list[int] | None,
    overwrite: bool,
) -> dict[str, Any]:
    if correction_damping < 0:
        raise ValueError("correction_damping must be non-negative")
    base_files = _files(base_alpha_dir)
    correction_files = _files(correction_alpha_dir)
    selected = sorted(set(base_files) & set(correction_files)) if layers is None else layers
    missing = [layer for layer in selected if layer not in base_files or layer not in correction_files]
    if missing:
        raise FileNotFoundError(f"missing layer alpha files: {missing}")
    if out_alpha_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output alpha dir exists: {out_alpha_dir}")
        shutil.rmtree(out_alpha_dir)
    out_alpha_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in selected:
        base_payload = torch.load(base_files[layer], map_location="cpu")
        correction_payload = torch.load(correction_files[layer], map_location="cpu")
        base = base_payload["alpha"].float()
        correction = correction_payload["alpha"].float()
        base_b, correction_b, alpha_mode = _broadcast_pair(base, correction)
        merged = (base_b * (1.0 + correction_damping * (correction_b - 1.0))).clamp(alpha_min, alpha_max)
        merged = merged.contiguous()
        out_path = out_alpha_dir / f"layer_{layer:02d}_{alpha_mode}_alpha.pt"
        payload = dict(base_payload)
        payload["schema_version"] = "lynn-a100-w4a8-alpha-merged-overlay-v1"
        payload["alpha_mode"] = alpha_mode
        payload["shape"] = list(merged.shape)
        payload["alpha"] = merged.cpu()
        payload["base_alpha_path"] = str(base_files[layer])
        payload["correction_alpha_path"] = str(correction_files[layer])
        payload["correction_damping"] = correction_damping
        payload["alpha_min"] = alpha_min
        payload["alpha_max"] = alpha_max
        torch.save(payload, out_path)
        rows.append(
            {
                "layer": layer,
                "alpha_mode": alpha_mode,
                "base_alpha_path": str(base_files[layer]),
                "correction_alpha_path": str(correction_files[layer]),
                "out_alpha_path": str(out_path),
                "base_stats": _stats(base),
                "correction_stats": _stats(correction),
                "merged_stats": _stats(merged),
            }
        )
    return {
        "schema_version": "lynn-a100-w4a8-alpha-merged-overlay-v1",
        "base_alpha_dir": str(base_alpha_dir),
        "correction_alpha_dir": str(correction_alpha_dir),
        "out_alpha_dir": str(out_alpha_dir),
        "correction_damping": correction_damping,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "layers": selected,
        "rows": rows,
        "decision": "GREEN: merged alpha overlay written; fold and run local/generation gates before promotion.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-alpha-dir", required=True)
    parser.add_argument("--correction-alpha-dir", required=True)
    parser.add_argument("--out-alpha-dir", required=True)
    parser.add_argument("--correction-damping", type=float, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.75)
    parser.add_argument("--alpha-max", type=float, default=1.25)
    parser.add_argument("--layers", type=int, nargs="*")
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = merge(
        base_alpha_dir=Path(args.base_alpha_dir),
        correction_alpha_dir=Path(args.correction_alpha_dir),
        out_alpha_dir=Path(args.out_alpha_dir),
        correction_damping=args.correction_damping,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
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
