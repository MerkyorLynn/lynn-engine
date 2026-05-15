"""
Lynn Engine Phase 4 P1 quantization manifest scanner.

This module does not load tensors into GPU memory. It inspects config/index and
safetensors metadata to produce a canonical, fail-loud description of a
checkpoint layout before any forward/dequant code touches the weights.

Goal: replace "silent garbage" with either a known QuantManifest or a clear
unsupported-format report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


SCHEMA_VERSION = "lynn-quant-manifest-v1"


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list[int]
    file: str = ""


@dataclass
class QuantManifest:
    schema_version: str
    model_dir: str
    model_type: str | None
    architectures: list[str]
    quant_method: str | None
    quant_format: str | None
    quant_format_normalized: str
    total_tensors: int
    total_size_bytes: int | None
    file_count: int
    suffix_counts: dict[str, int]
    dtype_counts: dict[str, int]
    sample_tensors: list[TensorInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    p1_status: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["sample_tensors"] = [asdict(x) for x in self.sample_tensors]
        return out

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


SUFFIXES = (
    ".weight",
    ".weight_scale_inv",
    ".weight_packed",
    ".weight_scale",
    ".weight_global_scale",
    ".input_global_scale",
    ".bias",
)


def _load_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config.json under {model_dir}")
    return json.loads(path.read_text())


def _normalize_quant_format(quant_cfg: dict[str, Any], keys: list[str]) -> str:
    method = quant_cfg.get("quant_method")
    fmt = quant_cfg.get("format")

    if method == "compressed-tensors" and fmt and "nvfp4" in str(fmt).lower():
        return "compressed_tensors_nvfp4_v8_rtn"
    if method in ("modelopt", "modelopt_fp4"):
        return "modelopt_fp4"
    if method in ("fp8", "float8"):
        return "fp8_block_scaled"
    if any(k.endswith(".weight_packed") for k in keys):
        return "packed_fp4_unknown"
    if any(k.endswith(".weight_scale_inv") for k in keys):
        return "fp8_block_scaled"
    if method is None:
        return "bf16_or_unquantized"
    return f"unknown:{method}:{fmt}"


def _read_weight_map_and_meta(model_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return index["weight_map"], index.get("metadata", {})
    if single_path.exists():
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as st:
            return {k: single_path.name for k in st.keys()}, {}
    raise FileNotFoundError(
        f"Expected model.safetensors.index.json or model.safetensors under {model_dir}"
    )


def _tensor_metadata(model_dir: Path, weight_map: dict[str, str], max_samples: int) -> tuple[dict[str, int], list[TensorInfo]]:
    from safetensors import safe_open

    dtype_counts: dict[str, int] = {}
    samples: list[TensorInfo] = []
    by_file: dict[str, list[str]] = {}
    for name, file_name in weight_map.items():
        by_file.setdefault(file_name, []).append(name)

    for file_name, names in by_file.items():
        with safe_open(model_dir / file_name, framework="pt", device="cpu") as st:
            for name in names:
                sl = st.get_slice(name)
                dtype = str(sl.get_dtype())
                shape = list(sl.get_shape())
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                if len(samples) < max_samples:
                    samples.append(TensorInfo(name=name, dtype=dtype, shape=shape, file=file_name))
    return dtype_counts, samples


def _suffix_counts(keys: list[str]) -> dict[str, int]:
    counts = {s: 0 for s in SUFFIXES}
    counts["other"] = 0
    for key in keys:
        for suffix in SUFFIXES:
            if key.endswith(suffix):
                counts[suffix] += 1
                break
        else:
            counts["other"] += 1
    return counts


def scan_checkpoint(model_dir: str | Path, max_samples: int = 24) -> QuantManifest:
    model_dir = Path(model_dir)
    config = _load_config(model_dir)
    text_config = config.get("text_config", config)
    quant_cfg = config.get("quantization_config", {})
    weight_map, index_meta = _read_weight_map_and_meta(model_dir)
    keys = sorted(weight_map)
    suffix_counts = _suffix_counts(keys)
    dtype_counts, samples = _tensor_metadata(model_dir, weight_map, max_samples)
    normalized = _normalize_quant_format(quant_cfg, keys)

    warnings: list[str] = []
    if normalized == "compressed_tensors_nvfp4_v8_rtn":
        required = [".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"]
        missing = [s for s in required if suffix_counts.get(s, 0) == 0]
        if missing:
            warnings.append(f"compressed-tensors NVFP4 missing expected suffixes: {missing}")
    if normalized == "bf16_or_unquantized" and suffix_counts.get(".weight_packed", 0):
        warnings.append("packed FP4 keys present without quantization_config")

    p1_status = "SUPPORTED_BF16_METADATA" if normalized == "bf16_or_unquantized" else "SUPPORTED_MANIFEST_ONLY"
    if normalized.startswith("unknown"):
        p1_status = "UNSUPPORTED_UNKNOWN_FORMAT"

    return QuantManifest(
        schema_version=SCHEMA_VERSION,
        model_dir=str(model_dir),
        model_type=text_config.get("model_type") or config.get("model_type"),
        architectures=config.get("architectures") or [],
        quant_method=quant_cfg.get("quant_method"),
        quant_format=quant_cfg.get("format"),
        quant_format_normalized=normalized,
        total_tensors=len(weight_map),
        total_size_bytes=int(index_meta["total_size"]) if "total_size" in index_meta else None,
        file_count=len(set(weight_map.values())),
        suffix_counts=suffix_counts,
        dtype_counts=dtype_counts,
        sample_tensors=samples,
        warnings=warnings,
        p1_status=p1_status,
    )


def save_manifest(model_dir: str | Path, out_path: str | Path, max_samples: int = 24) -> QuantManifest:
    manifest = scan_checkpoint(model_dir, max_samples=max_samples)
    Path(out_path).write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest
