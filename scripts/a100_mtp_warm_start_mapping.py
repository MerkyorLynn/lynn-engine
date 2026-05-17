#!/usr/bin/env python3
"""Build a Lynn MTP warm-start mapping from a Qwen3.6 A3B sidecar.

This does not train or integrate MTP into decode. It answers the concrete
checkpoint question: which official `mtp.*` tensors can initialize Lynn's
2048-hidden MTP head without shape surgery, and which tensors must be rebuilt.

When tensors are compatible, `--out-sidecar-dir` creates a warm-start directory.
It symlinks the source sidecar for direct-copy cases, or writes an aligned
`mtp.safetensors` when a supported shape adaptation such as expert-dimension
slicing is required.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MTP_BASE_SUFFIXES = {
    "mtp.layers.0.input_layernorm.weight": "input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight": "self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate.weight": "mlp.gate.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj": "mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj": "mlp.experts.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": "mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": "mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": "mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight": "mlp.shared_expert_gate.weight",
}

ROOT_TARGETS = {
    "mtp.fc.weight": lambda cfg: [int(cfg["hidden_size"]), int(cfg["hidden_size"]) * 2],
    "mtp.norm.weight": lambda cfg: [int(cfg["hidden_size"])],
    "mtp.pre_fc_norm_embedding.weight": lambda cfg: [int(cfg["hidden_size"])],
    "mtp.pre_fc_norm_hidden.weight": lambda cfg: [int(cfg["hidden_size"])],
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("text_config") or cfg


def _tensor_inventory(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            rows[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": tensor.numel(),
            }
    return rows


def _tensor_meta(model_dir: Path, weight_map: dict[str, str], key: str) -> dict[str, Any]:
    rel = weight_map.get(key)
    if rel is None:
        raise KeyError(f"missing base tensor key: {key}")
    with safe_open(model_dir / rel, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)
        return {
            "key": key,
            "relative_file": rel,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": tensor.numel(),
        }


def _find_first_attention_layer(weight_map: dict[str, str]) -> int:
    pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    layers = sorted({int(match.group(1)) for key in weight_map for match in [pattern.match(key)] if match})
    if not layers:
        raise KeyError("no full-attention self_attn.q_proj.weight layer found in base model")
    return layers[0]


def _target_shapes(
    *,
    base_model: Path,
    weight_map: dict[str, str],
    cfg: dict[str, Any],
    attention_layer: int,
    moe_layer: int,
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for key, make_shape in ROOT_TARGETS.items():
        shape = make_shape(cfg)
        targets[key] = {
            "source": "derived_from_config",
            "shape": shape,
            "dtype": "bfloat16",
            "base_key": None,
        }

    for mtp_key, suffix in MTP_BASE_SUFFIXES.items():
        layer = attention_layer if suffix.startswith("self_attn.") else moe_layer
        base_key = f"model.language_model.layers.{layer}.{suffix}"
        meta = _tensor_meta(base_model, weight_map, base_key)
        targets[mtp_key] = {
            "source": "derived_from_base_layer",
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "base_key": base_key,
            "base_relative_file": meta["relative_file"],
        }
    return targets


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=dst.parent)
    os.symlink(rel, dst)


def _can_slice_first_dim(src: dict[str, Any], target: dict[str, Any]) -> bool:
    src_shape = src["shape"]
    target_shape = target["shape"]
    return (
        len(src_shape) == len(target_shape)
        and len(src_shape) >= 2
        and src_shape[0] > target_shape[0]
        and src_shape[1:] == target_shape[1:]
    )


def _aligned_status(src: dict[str, Any] | None, target: dict[str, Any]) -> str:
    if src is None:
        return "missing_source"
    if src["dtype"] != target["dtype"]:
        return "dtype_mismatch"
    if src["shape"] == target["shape"]:
        return "direct_copy"
    if _can_slice_first_dim(src, target):
        return "slice_first_dim"
    return "shape_mismatch"


def _write_aligned_sidecar(
    *,
    sidecar_file: Path,
    out_file: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {}
    slice_rows = [row for row in rows if row["status"] == "slice_first_dim"]
    with safe_open(sidecar_file, framework="pt", device="cpu") as f:
        for row in rows:
            key = row["key"]
            tensor = f.get_tensor(key)
            if row["status"] == "slice_first_dim":
                target_first_dim = int(row["target_shape"][0])
                tensor = tensor[:target_first_dim].contiguous()
            tensors[key] = tensor
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(out_file),
        metadata={
            "lynn_mtp_warm_start": "true",
            "source_sidecar": str(sidecar_file),
            "alignment": "slice_first_dim_for_expert_count_mismatch",
        },
    )
    return {
        "aligned_sidecar_file": str(out_file),
        "aligned_sidecar_bytes": out_file.stat().st_size,
        "slice_first_dim_count": len(slice_rows),
        "slice_first_dim_keys": [row["key"] for row in slice_rows],
    }


def build_mapping(
    *,
    sidecar_file: Path,
    base_model: Path,
    attention_layer: int | None,
    moe_layer: int,
    out_sidecar_dir: Path | None,
    write_aligned_sidecar: bool,
) -> dict[str, Any]:
    cfg = _text_config(_read_json(base_model / "config.json"))
    weight_map = _read_json(base_model / "model.safetensors.index.json")["weight_map"]
    if attention_layer is None:
        attention_layer = _find_first_attention_layer(weight_map)
    source = _tensor_inventory(sidecar_file)
    targets = _target_shapes(
        base_model=base_model,
        weight_map=weight_map,
        cfg=cfg,
        attention_layer=attention_layer,
        moe_layer=moe_layer,
    )

    rows: list[dict[str, Any]] = []
    for key in sorted(targets):
        target = targets[key]
        src = source.get(key)
        status = _aligned_status(src, target)
        rows.append(
            {
                "key": key,
                "status": status,
                "source_shape": None if src is None else src["shape"],
                "target_shape": target["shape"],
                "source_dtype": None if src is None else src["dtype"],
                "target_dtype": target["dtype"],
                "target_source": target["source"],
                "base_key": target.get("base_key"),
            }
        )

    extra_source = sorted(set(source) - set(targets))
    counts = {
        "direct_copy": sum(row["status"] == "direct_copy" for row in rows),
        "slice_first_dim": sum(row["status"] == "slice_first_dim" for row in rows),
        "missing_source": sum(row["status"] == "missing_source" for row in rows),
        "shape_mismatch": sum(row["status"] == "shape_mismatch" for row in rows),
        "dtype_mismatch": sum(row["status"] == "dtype_mismatch" for row in rows),
        "extra_source": len(extra_source),
    }
    compatible = (
        counts["direct_copy"] + counts["slice_first_dim"] == len(rows)
        and counts["missing_source"] == 0
        and counts["shape_mismatch"] == 0
        and counts["dtype_mismatch"] == 0
        and counts["extra_source"] == 0
    )
    decision = (
        "GREEN: official MTP sidecar is direct-copy compatible with the Lynn 2048-hidden target contract."
        if compatible and counts["slice_first_dim"] == 0
        else "GREEN: official MTP sidecar is shape-alignable for Lynn by slicing the expert dimension."
        if compatible
        else "AMBER: MTP sidecar needs shape/dtype handling before warm-start use."
    )
    result = {
        "schema_version": "lynn-a100-mtp-warm-start-mapping-v1",
        "decision": decision,
        "sidecar_file": str(sidecar_file),
        "sidecar_bytes": sidecar_file.stat().st_size,
        "base_model": str(base_model),
        "base_hidden_size": cfg.get("hidden_size"),
        "reference_attention_layer": attention_layer,
        "reference_moe_layer": moe_layer,
        "counts": counts,
        "extra_source_keys": extra_source,
        "rows": rows,
    }
    if out_sidecar_dir is not None:
        out_sidecar_dir.mkdir(parents=True, exist_ok=True)
        if counts["slice_first_dim"] == 0 and compatible:
            _safe_symlink(sidecar_file.resolve(), out_sidecar_dir / "mtp.safetensors")
            result["warm_start_sidecar_mode"] = "symlink_direct"
        elif compatible and write_aligned_sidecar:
            aligned = _write_aligned_sidecar(
                sidecar_file=sidecar_file.resolve(),
                out_file=out_sidecar_dir / "mtp.safetensors",
                rows=rows,
            )
            result["warm_start_sidecar_mode"] = "aligned_safetensors"
            result["alignment"] = aligned
        else:
            result["warm_start_sidecar_mode"] = "manifest_only"
        (out_sidecar_dir / "mtp_warm_start_mapping.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result["warm_start_dir"] = str(out_sidecar_dir)
        if (out_sidecar_dir / "mtp.safetensors").exists() or (out_sidecar_dir / "mtp.safetensors").is_symlink():
            result["warm_start_sidecar"] = str(out_sidecar_dir / "mtp.safetensors")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-file", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--attention-layer", type=int)
    parser.add_argument("--moe-layer", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-sidecar-dir")
    parser.add_argument("--write-aligned-sidecar", action="store_true")
    args = parser.parse_args()

    result = build_mapping(
        sidecar_file=Path(args.sidecar_file),
        base_model=Path(args.base_model),
        attention_layer=args.attention_layer,
        moe_layer=args.moe_layer,
        out_sidecar_dir=Path(args.out_sidecar_dir) if args.out_sidecar_dir else None,
        write_aligned_sidecar=args.write_aligned_sidecar,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["decision"].startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
