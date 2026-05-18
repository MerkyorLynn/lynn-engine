"""Loader helpers for Lynn MoE serving-layout sidecars."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


SCHEMA_VERSION = "qwen36-w4a16-moe-repack-sidecar-v1"


@dataclass(slots=True)
class MoeRepackLayer:
    """One layer from a MoE repack sidecar.

    Tensor names intentionally mirror the sidecar contract, not the original
    safetensors manifest.  `active_aliases()` exposes the current runtime keys
    so probes can compare the sidecar against existing Triton paths.
    """

    layer_idx: int
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    def active_aliases(self) -> dict[str, torch.Tensor]:
        return {
            "mlp.gate.weight": self.tensors["router.weight"],
            "mlp.experts._gate_up_packed": self.tensors["active_gate_up.packed"],
            "mlp.experts._gate_up_scale": self.tensors["active_gate_up.scale"].float(),
            "mlp.experts._gate_up_global_scale": self.tensors["active_gate_up.global_scale"].float(),
            "mlp.experts._down_packed": self.tensors["active_down.packed"],
            "mlp.experts._down_scale": self.tensors["active_down.scale"].float(),
            "mlp.experts._down_global_scale": self.tensors["active_down.global_scale"].float(),
        }

    def shared_aliases(self) -> dict[str, torch.Tensor]:
        aliases: dict[str, torch.Tensor] = {}
        mapping = {
            "mlp.shared_expert.gate_proj.weight": "shared_gate",
            "mlp.shared_expert.up_proj.weight": "shared_up",
            "mlp.shared_expert.down_proj.weight": "shared_down",
            "mlp.shared_expert_gate.weight": "shared_scalar_gate",
        }
        for runtime_key, short in mapping.items():
            packed_key = f"{short}.packed"
            scale_key = f"{short}.scale"
            global_key = f"{short}.global_scale"
            if packed_key in self.tensors:
                aliases[f"{runtime_key}.packed"] = self.tensors[packed_key]
                aliases[f"{runtime_key}.scale"] = self.tensors[scale_key].float()
                aliases[f"{runtime_key}.global_scale"] = self.tensors[global_key].float()
        return aliases


def load_moe_repack_manifest(sidecar_dir: str | Path) -> dict[str, Any]:
    sidecar = Path(sidecar_dir)
    manifest_path = sidecar / "moe_repack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported MoE repack schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    return manifest


def load_moe_repack_layer(
    sidecar_dir: str | Path,
    layer_idx: int,
    *,
    device: str | torch.device = "cuda",
) -> MoeRepackLayer:
    sidecar = Path(sidecar_dir)
    manifest = load_moe_repack_manifest(sidecar)
    rec = manifest.get("layers", {}).get(str(layer_idx))
    if rec is None:
        raise KeyError(f"layer {layer_idx} missing from {sidecar / 'moe_repack_manifest.json'}")
    tensors = load_file(sidecar / rec["file"], device=str(device))
    required = {
        "router.weight",
        "active_gate_up.packed",
        "active_gate_up.scale",
        "active_gate_up.global_scale",
        "active_down.packed",
        "active_down.scale",
        "active_down.global_scale",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise KeyError(f"MoE sidecar layer {layer_idx} missing required tensors: {missing}")
    return MoeRepackLayer(layer_idx=layer_idx, tensors=tensors, metadata=rec)
