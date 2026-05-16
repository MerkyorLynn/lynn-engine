"""Fail-loud NVFP4 artifact layout detection.

Lynn now intentionally supports two NVFP4 artifact families:

* Lynn-native per-16 variable-expert artifacts, consumed by Lynn engine.
* Vendor-friendly ModelOpt / compressed-tensors artifacts, consumed by the
  ecosystem and optionally by future Lynn vendor backends.

This module is deliberately metadata-only. It does not import torch or touch
tensor payloads, so it is safe to run before selecting a loader/backend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


NVFP4LayoutKind = Literal[
    "lynn_native_per16_variable",
    "compressed_tensors_nvfp4",
    "modelopt_nvfp4",
    "packed_fp4_unknown",
    "bf16_or_unquantized",
    "unsupported",
]


@dataclass(slots=True)
class NVFP4LayoutReport:
    schema_version: str
    model_dir: str
    layout_kind: NVFP4LayoutKind
    recommended_loader: str
    recommended_backend_family: str
    quant_method: str | None
    quant_format: str | None
    has_lynn_manifest: bool
    has_variable_expert_spec: bool
    hf_vanilla_compatible: bool | None
    weight_key_count: int
    suffix_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - fail-loud metadata path
        raise ValueError(f"Failed to parse {path}: {exc}") from exc


def _read_weight_keys(model_dir: Path) -> list[str]:
    index = _load_json(model_dir / "model.safetensors.index.json")
    if index:
        return sorted(index.get("weight_map", {}).keys())
    single = model_dir / "model.safetensors"
    if not single.exists():
        return []
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as st:
        return sorted(st.keys())


def _suffix_counts(keys: list[str]) -> dict[str, int]:
    suffixes = (
        ".weight",
        ".weight_packed",
        ".weight_scale",
        ".weight_global_scale",
        ".input_global_scale",
        ".weight_scale_inv",
        ".bias",
    )
    counts = {suffix: 0 for suffix in suffixes}
    counts["other"] = 0
    for key in keys:
        for suffix in suffixes:
            if key.endswith(suffix):
                counts[suffix] += 1
                break
        else:
            counts["other"] += 1
    return counts


def detect_nvfp4_layout(model_dir: str | Path) -> NVFP4LayoutReport:
    """Classify a checkpoint before any tensor loader is selected."""

    model_dir = Path(model_dir)
    config = _load_json(model_dir / "config.json")
    quant_cfg = config.get("quantization_config", {}) if config else {}
    quant_method = quant_cfg.get("quant_method")
    quant_format = quant_cfg.get("format")
    keys = _read_weight_keys(model_dir)
    counts = _suffix_counts(keys)

    lynn_manifest = _load_json(model_dir / "lynn_quant_manifest.json")
    variable_spec = _load_json(model_dir / "lynn_engine_variable_expert_spec.json")
    has_lynn_manifest = bool(lynn_manifest)
    has_variable_spec = bool(variable_spec)
    hf_vanilla_compatible = variable_spec.get("hf_vanilla_compatible") if has_variable_spec else None

    warnings: list[str] = []
    blockers: list[str] = []

    if lynn_manifest.get("schema_version") == "lynn-variable-nvfp4-pack-v1":
        layout = "lynn_native_per16_variable"
        loader = "engine.loader:_load_qwen36_layer_lynn_variable_nvfp4"
        backend = "lynn_native_per16"
        if hf_vanilla_compatible is not False:
            warnings.append("lynn manifest present but variable-expert spec is missing or not explicitly non-vanilla")
    elif quant_method == "compressed-tensors" and quant_format and "nvfp4" in str(quant_format).lower():
        layout = "compressed_tensors_nvfp4"
        loader = "engine.loader:_load_qwen36_layer_nvfp4_v8_rtn"
        backend = "vendor_compressed_tensors"
        if has_variable_spec and hf_vanilla_compatible is False:
            blockers.append("compressed-tensors NVFP4 plus physical variable-expert spec requires explicit adapter/padding support")
    elif quant_method in ("modelopt", "modelopt_fp4") or (
        quant_format and "modelopt" in str(quant_format).lower()
    ):
        layout = "modelopt_nvfp4"
        loader = "vendor_modelopt_or_compressed_tensors_converter"
        backend = "vendor_modelopt"
        blockers.append("Lynn engine does not yet implement direct ModelOpt NVFP4 tensor loading")
    elif counts.get(".weight_packed", 0):
        layout = "packed_fp4_unknown"
        loader = "none"
        backend = "unsupported_packed_fp4"
        blockers.append("packed FP4 keys present but no recognized Lynn or vendor NVFP4 manifest")
    elif quant_method is None:
        layout = "bf16_or_unquantized"
        loader = "engine.loader:load_qwen36_layer"
        backend = "bf16_reference"
    else:
        layout = "unsupported"
        loader = "none"
        backend = "unsupported"
        blockers.append(f"unknown quantization_config quant_method={quant_method!r} format={quant_format!r}")

    return NVFP4LayoutReport(
        schema_version="lynn-engine-nvfp4-layout-report-v1",
        model_dir=str(model_dir),
        layout_kind=layout,  # type: ignore[arg-type]
        recommended_loader=loader,
        recommended_backend_family=backend,
        quant_method=quant_method,
        quant_format=quant_format,
        has_lynn_manifest=has_lynn_manifest,
        has_variable_expert_spec=has_variable_spec,
        hf_vanilla_compatible=hf_vanilla_compatible,
        weight_key_count=len(keys),
        suffix_counts=counts,
        warnings=warnings,
        blockers=blockers,
    )
