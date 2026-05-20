#!/usr/bin/env python3
"""Offline NVFP4 → FP8 E4M3 repack for Spark sm_121 FP8 MMA path.

Spark sm_121 (GB10) has FP8 E4M3/E5M2 MMA at 162 TFLOPS peak (1.64× BF16)
but lacks FP4 MMA. The naive runtime path (dequant NVFP4→BF16→matmul) only
hits ~21 TPS at decode; a naive `_scaled_mm` swap of NVFP4 weight to FP8
inline measured 14 TPS due to activation cast / layout / scale / launch
overhead (memory ``reference_spark_fp8_w4a8_design_strategy_20260519``).

The strategy is: do the FP4→FP8 cast **once offline** so the inference
kernel only handles FP8 × FP8 matmul + activation cast. This script is
the offline repack tool (Phase 2 task #1).

V0 scope:
  * Function-level NVFP4 → FP8 conversion with per-row scale.
  * Verify dequant cos > 0.999 vs original NVFP4.
  * Self-test on synthetic NVFP4 data (no external model files needed).
  * CLI to repack a single safetensors weight key for an end-to-end smoke.

V1 scope (this file, ``full-dir`` subcommand):
  * Full Lynn-native model dir repack: read ``lynn_quant_manifest.json``
    + ``model.safetensors.index.json`` from the input dir, repack every
    2D ``quantized_tensors`` entry, copy every ``kept_tensors`` entry,
    write a parallel output dir with new shards + new manifest (schema
    ``lynn-variable-w4a8-fp8-v1``).
  * 3D MoE expert weights (``mlp.experts.gate_up_proj``,
    ``mlp.experts.down_proj``) are intentionally deferred: the FP8 layout
    for the flattened-experts storage is still being designed, so the
    tool emits an explicit ``deferred_tensors`` map in the output
    manifest instead of guessing.

V2 scope (later, separate commit):
  * Col-major storage layout for cuBLASLt FP8 GEMM sweet spot.
  * Fused gate+up concatenated weight packing.
  * MoE expert FP8 layout once Triton expert kernel design lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# E2M1 magnitude table (compressed-tensors / Lynn-native NVFP4 share this)
E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32,
)

# FP8 E4M3 max representable absolute value (= 448).
FP8_E4M3_MAX = 448.0


@dataclass(slots=True)
class RepackResult:
    """Output of one NVFP4 → FP8 repack call."""
    fp8_weight: torch.Tensor          # [out_features, in_features] in float8_e4m3fn
    fp8_scale: torch.Tensor           # [out_features] (per-row) or scalar (per-tensor) — float32
    scale_granularity: str            # "per_row" | "per_tensor"
    bf16_intermediate_norm: float     # ||W_bf16||_F (for diagnostic)
    diff_max_abs_vs_bf16: float       # FP8-roundtrip vs BF16 max abs diff
    cosine_vs_bf16: float             # cos between FP8-roundtrip and BF16 dequant


def unpack_fp4_e2m1_from_uint8(
    packed: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack NVFP4 E2M1 values from uint8 storage.

    Two FP4 values per byte: low nibble first, high nibble second.
    Bit 3 = sign, bits 0-2 = magnitude index into E2M1_TO_FLOAT.

    Mirrors ``engine/dequant.py::unpack_fp4_e2m1_from_uint8``.
    """
    if packed.dtype is not torch.uint8:
        raise TypeError(f"expected uint8 packed tensor, got {packed.dtype}")
    flat = packed.flatten()
    low = flat & 0x0F
    high = (flat & 0xF0) >> 4
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    magnitudes = (combined & 0x07).to(torch.long)
    table = E2M1_TO_FLOAT.to(device=packed.device)
    values = table[magnitudes] * torch.where(
        signs,
        torch.tensor(-1.0, device=packed.device),
        torch.tensor(1.0, device=packed.device),
    )
    unpacked_shape = list(packed.shape)
    unpacked_shape[-1] *= 2
    return values.reshape(unpacked_shape).to(dtype=dtype)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weight to BF16/FP32.

    Mirrors ``engine/dequant.py::dequantize_nvfp4_v8_rtn_weight`` —
    group_size = 16, per-group BF16 scale, optional global_scale
    (compressed-tensors v8-RTN). Lynn-native ``per16_variable`` uses
    ``weight_global_scale=None``.
    """
    unpacked = unpack_fp4_e2m1_from_uint8(weight_packed, dtype=torch.float32)
    scale = weight_scale.to(torch.float32)
    if unpacked.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"expected 2D weight + 2D scale; got weight={tuple(unpacked.shape)} "
            f"scale={tuple(scale.shape)}"
        )
    group_size = unpacked.shape[1] // scale.shape[1]
    if group_size != 16:
        raise ValueError(f"expected group_size=16, got {group_size}")
    if weight_global_scale is not None:
        scale = scale / weight_global_scale.to(torch.float32)
    scale_full = scale.repeat_interleave(group_size, dim=1)
    if scale_full.shape[0] == 1 and unpacked.shape[0] != 1:
        scale_full = scale_full.expand(unpacked.shape[0], -1)
    scale_full = scale_full[: unpacked.shape[0], : unpacked.shape[1]]
    return (unpacked * scale_full).to(output_dtype)


def repack_nvfp4_to_fp8(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None = None,
    *,
    scale_granularity: str = "per_row",
) -> RepackResult:
    """Offline NVFP4 → FP8 E4M3 repack.

    Args:
        weight_packed: uint8 [out_features, in_features // 2] NVFP4 E2M1 packed.
        weight_scale: BF16/FP32 [out_features, in_features // 16] per-group scale.
        weight_global_scale: optional FP32 scalar (compressed-tensors v8-RTN).
            For Lynn-native ``per16_variable`` pass None.
        scale_granularity: ``per_row`` (per-output-channel) or ``per_tensor``.
            cuBLASLt _scaled_mm supports per-row scale_b; ``per_tensor`` is
            simpler but sacrifices ~3-5% quality on heavy-tailed weights.

    Returns:
        :class:`RepackResult` with fp8_weight, fp8_scale, and verification
        diff vs the BF16 dequant.
    """
    if scale_granularity not in {"per_row", "per_tensor"}:
        raise ValueError(f"unknown scale_granularity={scale_granularity!r}")

    # 1. Reference BF16 dequant.
    bf16 = dequantize_nvfp4(
        weight_packed, weight_scale, weight_global_scale, output_dtype=torch.bfloat16,
    )
    bf16_norm = float(bf16.float().flatten().norm().item())

    # 2. Derive FP8 scale.
    bf16_f = bf16.float()
    if scale_granularity == "per_row":
        per_row_max = bf16_f.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-12)
        fp8_scale_2d = per_row_max / FP8_E4M3_MAX  # [N, 1] f32
        fp8_scale_out = fp8_scale_2d.squeeze(-1).contiguous()  # [N] f32
    else:
        per_tensor_max = bf16_f.abs().amax().clamp_min(1.0e-12)
        fp8_scale_2d = (per_tensor_max / FP8_E4M3_MAX).view(1, 1)
        fp8_scale_out = fp8_scale_2d.flatten().contiguous()  # [1] f32

    # 3. Quantize BF16 → FP8 E4M3 with derived scale.
    fp8_weight = (bf16_f / fp8_scale_2d).to(torch.float8_e4m3fn).contiguous()

    # 4. Verify round-trip dequant. Use FP64 for cosine accumulation —
    # large tensors (e.g. lm_head 311M elements) overflow FP32 accumulator
    # precision and produce nonsense cos > 1.0.
    fp8_roundtrip = fp8_weight.to(torch.float32) * fp8_scale_2d
    diff = (fp8_roundtrip - bf16_f).flatten()
    max_abs = float(diff.abs().max().item())
    af = fp8_roundtrip.flatten().double()
    bf = bf16_f.flatten().double()
    dot = float((af * bf).sum().item())
    na = float(af.norm().item())
    nb = float(bf.norm().item())
    cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")

    return RepackResult(
        fp8_weight=fp8_weight,
        fp8_scale=fp8_scale_out,
        scale_granularity=scale_granularity,
        bf16_intermediate_norm=bf16_norm,
        diff_max_abs_vs_bf16=max_abs,
        cosine_vs_bf16=cos,
    )


def synthetic_nvfp4(
    out_features: int,
    in_features: int,
    *,
    seed: int = 1234,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic NVFP4 (packed uint8 + BF16 per-16 scale) tensor.

    Used by the self-test. Produces a deterministic packed pattern that
    decodes to a realistic-ish weight distribution.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # Packed weight: random uint8 of shape [N, K/2]
    if in_features % 2 != 0:
        raise ValueError("in_features must be even for NVFP4 packing")
    if in_features % 16 != 0:
        raise ValueError("in_features must be divisible by group_size 16")
    packed = torch.randint(
        0, 256, (out_features, in_features // 2), generator=g, dtype=torch.uint8, device=device,
    )
    # Scale: per-row per-16-group, BF16, small positive values
    scale = (
        0.01 + 0.05 * torch.rand(
            (out_features, in_features // 16), generator=g, dtype=torch.float32, device=device,
        )
    ).to(torch.bfloat16)
    return packed, scale


def self_test() -> int:
    """Self-test: synthetic NVFP4 → FP8 repack, verify cos > 0.999."""
    torch.manual_seed(0)
    print("[spark_pack_w4a8_fp8] running self-test on synthetic NVFP4...")
    shapes = [
        (256, 2048),     # MoE expert gate/up size
        (2048, 6144),    # shared expert fan-in
        (2048, 2048),    # q/k/v_proj-style square
        (151936, 2048),  # lm_head-style large
    ]
    overall_ok = True
    for out_features, in_features in shapes:
        for granularity in ("per_row", "per_tensor"):
            packed, scale = synthetic_nvfp4(out_features, in_features)
            result = repack_nvfp4_to_fp8(packed, scale, scale_granularity=granularity)
            cos_ok = result.cosine_vs_bf16 > 0.999
            print(
                f"  shape=({out_features}, {in_features}) granularity={granularity}: "
                f"cos={result.cosine_vs_bf16:.6f} max_abs={result.diff_max_abs_vs_bf16:.4e} "
                f"fp8_scale_shape={tuple(result.fp8_scale.shape)} "
                f"bf16_norm={result.bf16_intermediate_norm:.2e} "
                f"{'OK' if cos_ok else 'FAIL'}"
            )
            if not cos_ok:
                overall_ok = False

    print(f"[spark_pack_w4a8_fp8] self-test {'PASSED' if overall_ok else 'FAILED'}")
    return 0 if overall_ok else 1


def repack_safetensors_weight(
    input_path: Path,
    output_path: Path,
    weight_key: str,
    *,
    scale_granularity: str = "per_row",
) -> dict[str, Any]:
    """Repack a single weight from a safetensors file.

    Looks for ``{weight_key}.weight_packed`` and ``{weight_key}.weight_scale``
    (Lynn-native naming) in the input file, repacks, and writes FP8 result
    + scale + verification report.

    Returns a JSON-serializable manifest dict.
    """
    from safetensors.torch import load_file, save_file

    src = load_file(str(input_path))
    packed_key = f"{weight_key}.weight_packed"
    scale_key = f"{weight_key}.weight_scale"
    global_scale_key = f"{weight_key}.weight_global_scale"
    if packed_key not in src or scale_key not in src:
        raise KeyError(
            f"input file missing required keys {packed_key!r} / {scale_key!r}; "
            f"available keys: {sorted(src.keys())[:10]}..."
        )
    packed = src[packed_key]
    scale = src[scale_key]
    global_scale = src.get(global_scale_key)

    result = repack_nvfp4_to_fp8(
        packed, scale, global_scale, scale_granularity=scale_granularity,
    )

    out_tensors = {
        f"{weight_key}.weight_fp8": result.fp8_weight,
        f"{weight_key}.weight_fp8_scale": result.fp8_scale,
    }
    save_file(out_tensors, str(output_path))

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "weight_key": weight_key,
        "input_packed_shape": list(packed.shape),
        "input_packed_dtype": str(packed.dtype),
        "input_scale_shape": list(scale.shape),
        "input_scale_dtype": str(scale.dtype),
        "output_fp8_shape": list(result.fp8_weight.shape),
        "output_fp8_dtype": str(result.fp8_weight.dtype),
        "output_scale_shape": list(result.fp8_scale.shape),
        "scale_granularity": result.scale_granularity,
        "cosine_vs_bf16": result.cosine_vs_bf16,
        "max_abs_vs_bf16": result.diff_max_abs_vs_bf16,
        "bf16_intermediate_norm": result.bf16_intermediate_norm,
    }
    return manifest


def main() -> int:
    pass  # placeholder, replaced below

# ---------------------------------------------------------------------
# V1: full Lynn-native model dir repack
# ---------------------------------------------------------------------


# Output schema constant. Bumping this means the V1-compatible loader
# must be updated in lockstep.
LYNN_W4A8_FP8_MANIFEST_SCHEMA = "lynn-variable-w4a8-fp8-v1"

# File names inside a Lynn-native model dir.
MANIFEST_FILE = "lynn_quant_manifest.json"
INDEX_FILE = "model.safetensors.index.json"

# Non-tensor sidecar files we copy verbatim from input -> output so the
# output dir is independently usable by HF-compatible loaders.
SIDECAR_FILES_DEFAULT = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "LICENSE",
    "README.md",
)


def _build_shard_index(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Group keys in ``model.safetensors.index.json`` by shard filename."""
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    return by_shard


def _open_shard_for_key(
    input_dir: Path,
    weight_map: dict[str, str],
    key: str,
    *,
    cache: dict[str, Any] | None = None,
):
    """Return a ``safe_open`` handle for the shard that contains ``key``.

    The caller is responsible for closing the handle. ``cache`` is an
    optional shard-name -> handle dict for batch reuse.
    """
    from safetensors import safe_open

    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"key {key!r} not in weight_map")
    if cache is not None and shard in cache:
        return cache[shard], shard
    handle = safe_open(input_dir / shard, framework="pt", device="cpu")
    if cache is not None:
        cache[shard] = handle
    return handle, shard


def _safetensors_metadata_total_size(tensors: dict[str, torch.Tensor]) -> int:
    """Sum of element_size * numel for every tensor — used for the index
    metadata.total_size field that HF tooling expects."""
    total = 0
    for t in tensors.values():
        total += t.element_size() * t.numel()
    return total


def _is_repackable(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Decide whether a manifest entry is a 2D weight we can FP8-repack.

    Returns (True, None) if repackable; (False, reason) otherwise.
    """
    shape = list(record.get("original_shape") or [])
    if len(shape) == 2:
        return True, None
    if len(shape) == 0:
        return False, "no original_shape recorded"
    if len(shape) == 1:
        return False, "1D weight — unexpected for matrix MMA path"
    if len(shape) >= 3:
        return False, f"non-2D original_shape {shape} (MoE expert / vision tile — FP8 layout TBD)"
    return False, f"unexpected original_shape rank {len(shape)}"


def repack_full_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    scale_granularity: str = "per_row",
    verify_cos_threshold: float = 0.999,
    copy_sidecars: bool = True,
    sidecar_files: tuple[str, ...] = SIDECAR_FILES_DEFAULT,
    progress: bool = True,
    limit_quantized: int | None = None,
) -> dict[str, Any]:
    """Repack a full Lynn-native model dir to a parallel FP8 V1 dir.

    The output dir is shard-parallel with the input: every input shard
    ``model-XXXXX-of-NNNNN.safetensors`` produces a same-named output
    shard with the FP8 weight + scale tensors that replace its
    NVFP4 ``.weight.packed`` / ``.weight.scale`` / ``.weight.global_scale``
    triples, plus all kept tensors that originally lived in that shard.
    Cross-shard triples (where packed and scale live in different
    shards) are loaded transparently via the index.

    Args:
      input_dir: a Lynn-native ``lynn-variable-nvfp4-pack-v1`` dir.
      output_dir: destination dir; created if missing. Must NOT be the
                  same as ``input_dir``.
      scale_granularity: per-row or per-tensor FP8 scale.
      verify_cos_threshold: per-tensor cos threshold; tensors below
                            this go into the failure list.
      copy_sidecars: copy ``config.json`` etc. from input -> output.
      sidecar_files: which non-tensor files to copy.
      progress: print one line per tensor (every Nth, capped).
      limit_quantized: stop after the first N quantized tensors
                       (debug / smoke; ``None`` = process all).

    Returns the summary dict (also written to ``<output_dir>/repack_summary.json``).
    """
    import shutil
    import time

    from safetensors import safe_open
    from safetensors.torch import save_file

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("input and output dirs must differ")
    if not (input_dir / MANIFEST_FILE).is_file():
        raise FileNotFoundError(f"missing {MANIFEST_FILE} in {input_dir}")
    if not (input_dir / INDEX_FILE).is_file():
        raise FileNotFoundError(f"missing {INDEX_FILE} in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    src_manifest = json.loads((input_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    src_index_full = json.loads((input_dir / INDEX_FILE).read_text(encoding="utf-8"))
    src_weight_map: dict[str, str] = src_index_full.get("weight_map") or {}

    quantized_tensors = src_manifest.get("quantized_tensors") or {}
    kept_tensors = src_manifest.get("kept_tensors") or {}

    # Pre-classify every quantized entry into repackable / deferred so we
    # can decide which output shard each FP8 tensor lives in (output
    # sharding mirrors the original ``packed_key`` shard).
    repackable: list[tuple[str, dict[str, Any]]] = []
    deferred: list[dict[str, Any]] = []
    for logical_key, record in quantized_tensors.items():
        ok, reason = _is_repackable(record)
        if ok:
            repackable.append((logical_key, record))
        else:
            deferred.append({
                "logical_key": logical_key,
                "reason": reason,
                "original_shape": record.get("original_shape"),
                "packed_key": record.get("packed_key"),
            })
    if limit_quantized is not None:
        repackable = repackable[:int(limit_quantized)]

    # Group output writes by shard. The output-shard for an FP8 weight is
    # the shard that originally held its ``.weight.packed`` key. Kept
    # tensors stay in their original shard.
    out_by_shard: dict[str, dict[str, torch.Tensor]] = {}

    # Per-tensor stats for the summary.
    per_tensor: list[dict[str, Any]] = []
    cos_min = float("inf")
    cos_max = -float("inf")
    cos_sum = 0.0
    cos_count = 0
    cos_failures: list[dict[str, Any]] = []
    total_fp8_bytes = 0
    total_scale_bytes = 0
    kept_count = 0

    new_manifest_quantized: dict[str, dict[str, Any]] = {}

    # Open shard cache so we don't re-open the same file once per key.
    handle_cache: dict[str, Any] = {}

    t_repack_start = time.time()
    try:
        for idx, (logical_key, record) in enumerate(repackable):
            packed_key = record["packed_key"]
            scale_key = record["scale_key"]
            global_scale_key = record["global_scale_key"]
            original_shape = list(record["original_shape"])

            packed_handle, packed_shard = _open_shard_for_key(
                input_dir, src_weight_map, packed_key, cache=handle_cache,
            )
            scale_handle, _ = _open_shard_for_key(
                input_dir, src_weight_map, scale_key, cache=handle_cache,
            )
            global_handle, _ = _open_shard_for_key(
                input_dir, src_weight_map, global_scale_key, cache=handle_cache,
            )
            packed = packed_handle.get_tensor(packed_key)
            scale = scale_handle.get_tensor(scale_key)
            global_scale = global_handle.get_tensor(global_scale_key)

            result = repack_nvfp4_to_fp8(
                packed, scale, global_scale, scale_granularity=scale_granularity,
            )
            cos = result.cosine_vs_bf16
            cos_min = min(cos_min, cos)
            cos_max = max(cos_max, cos)
            cos_sum += cos
            cos_count += 1
            if cos < verify_cos_threshold:
                cos_failures.append({
                    "logical_key": logical_key,
                    "cos": cos,
                    "max_abs": result.diff_max_abs_vs_bf16,
                    "shape": list(result.fp8_weight.shape),
                })

            fp8_key = f"{logical_key}_fp8"
            scale_out_key = f"{logical_key}_fp8_scale"
            # Output shard mirrors the original packed-shard placement so
            # large dirs stay cleanly sharded by layer.
            shard_bucket = out_by_shard.setdefault(packed_shard, {})
            shard_bucket[fp8_key] = result.fp8_weight
            shard_bucket[scale_out_key] = result.fp8_scale

            total_fp8_bytes += result.fp8_weight.element_size() * result.fp8_weight.numel()
            total_scale_bytes += result.fp8_scale.element_size() * result.fp8_scale.numel()

            new_manifest_quantized[logical_key] = {
                "weight_fp8_key": fp8_key,
                "weight_fp8_scale_key": scale_out_key,
                "original_shape": original_shape,
                "original_dtype": record.get("original_dtype"),
                "fp8_dtype": "float8_e4m3fn",
                "fp8_scale_dtype": "float32",
                "fp8_scale_shape": list(result.fp8_scale.shape),
                "fp8_scale_granularity": result.scale_granularity,
                "verify_cos_vs_bf16": cos,
                "verify_max_abs_vs_bf16": result.diff_max_abs_vs_bf16,
                "source_packed_key": packed_key,
                "source_scale_key": scale_key,
                "source_global_scale_key": global_scale_key,
            }
            per_tensor.append({
                "logical_key": logical_key,
                "fp8_key": fp8_key,
                "fp8_scale_key": scale_out_key,
                "shape": list(result.fp8_weight.shape),
                "scale_shape": list(result.fp8_scale.shape),
                "cos_vs_bf16": cos,
                "max_abs_vs_bf16": result.diff_max_abs_vs_bf16,
                "shard": packed_shard,
            })
            if progress and (idx % 25 == 0 or idx == len(repackable) - 1):
                print(
                    f"  [{idx+1}/{len(repackable)}] {logical_key} "
                    f"cos={cos:.6f} max_abs={result.diff_max_abs_vs_bf16:.3e} "
                    f"shard={packed_shard}",
                    flush=True,
                )

        # Kept tensors: copy verbatim into the same shard they came from.
        for kept_key, _kept_rec in kept_tensors.items():
            handle, shard = _open_shard_for_key(
                input_dir, src_weight_map, kept_key, cache=handle_cache,
            )
            tensor = handle.get_tensor(kept_key)
            out_by_shard.setdefault(shard, {})[kept_key] = tensor
            kept_count += 1
    finally:
        for handle in handle_cache.values():
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass

    # Write each output shard.
    out_weight_map: dict[str, str] = {}
    out_total_size = 0
    for shard_name, shard_tensors in out_by_shard.items():
        out_path = output_dir / shard_name
        save_file(shard_tensors, str(out_path))
        out_total_size += _safetensors_metadata_total_size(shard_tensors)
        for k in shard_tensors:
            out_weight_map[k] = shard_name
        if progress:
            print(
                f"  wrote {shard_name}: {len(shard_tensors)} keys, "
                f"{sum(t.element_size()*t.numel() for t in shard_tensors.values())/1e9:.2f} GB",
                flush=True,
            )

    # New manifest.
    new_manifest = {
        "schema_version": LYNN_W4A8_FP8_MANIFEST_SCHEMA,
        "produced_by": "scripts/spark_pack_w4a8_fp8.py full-dir",
        "source_model_dir": str(input_dir),
        "source_manifest_schema": src_manifest.get("schema_version"),
        "quantization": {
            "format": "fp8_e4m3_per_row" if scale_granularity == "per_row" else "fp8_e4m3_per_tensor",
            "weight_dtype": "float8_e4m3fn",
            "scale_dtype": "float32",
            "scale_granularity": scale_granularity,
            "weight_activation_contract": "W4A8_weight_fp8_act_fp8",
            "fp8_max": FP8_E4M3_MAX,
            "verify_cos_threshold": verify_cos_threshold,
            "source_quantization": src_manifest.get("quantization"),
        },
        "runtime_contract": {
            "inference_path_required": "fp8_scaled_mm",
            "fallback_path_allowed": True,
        },
        "kept_tensors": kept_tensors,
        "quantized_tensors": new_manifest_quantized,
        "deferred_tensors": deferred,
        "quantized_count": len(new_manifest_quantized),
        "deferred_count": len(deferred),
        "kept_count": kept_count,
        "output_shards": sorted(out_by_shard.keys()),
    }
    (output_dir / MANIFEST_FILE).write_text(
        json.dumps(new_manifest, indent=2) + "\n", encoding="utf-8",
    )

    # New index. Total size matches the safetensors writer's metadata key.
    new_index = {
        "metadata": {
            "total_size": out_total_size,
            "schema_version": LYNN_W4A8_FP8_MANIFEST_SCHEMA,
        },
        "weight_map": out_weight_map,
    }
    (output_dir / INDEX_FILE).write_text(
        json.dumps(new_index, indent=2) + "\n", encoding="utf-8",
    )

    # Sidecar copy.
    copied_sidecars: list[str] = []
    if copy_sidecars:
        for fname in sidecar_files:
            src_file = input_dir / fname
            if src_file.is_file():
                shutil.copy2(src_file, output_dir / fname)
                copied_sidecars.append(fname)

    elapsed = time.time() - t_repack_start
    cos_mean = (cos_sum / cos_count) if cos_count else float("nan")

    summary = {
        "schema_version": LYNN_W4A8_FP8_MANIFEST_SCHEMA,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "scale_granularity": scale_granularity,
        "verify_cos_threshold": verify_cos_threshold,
        "elapsed_seconds": elapsed,
        "totals": {
            "quantized_in_input": len(quantized_tensors),
            "repacked": len(per_tensor),
            "deferred": len(deferred),
            "kept": kept_count,
            "fp8_bytes": total_fp8_bytes,
            "fp8_scale_bytes": total_scale_bytes,
            "fp8_total_gib": (total_fp8_bytes + total_scale_bytes) / (1024 ** 3),
            "input_total_size_index": src_index_full.get("metadata", {}).get("total_size"),
            "output_total_size_index": out_total_size,
        },
        "cos_vs_bf16": {
            "min": (cos_min if cos_count else None),
            "max": (cos_max if cos_count else None),
            "mean": cos_mean,
            "count": cos_count,
            "failures_below_threshold": cos_failures,
            "failures_count": len(cos_failures),
        },
        "deferred_tensors": deferred,
        "copied_sidecars": copied_sidecars,
        "output_shards": sorted(out_by_shard.keys()),
        "per_tensor": per_tensor,
    }
    (output_dir / "repack_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------
# CLI plumbing (V0 commands kept; V1 ``full-dir`` added)
# ---------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline NVFP4 → FP8 E4M3 repack")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_test = sub.add_parser("self-test", help="Run self-test on synthetic NVFP4")  # noqa: F841

    sp_one = sub.add_parser("one", help="Repack a single weight from a safetensors file")
    sp_one.add_argument("--input", required=True, type=Path)
    sp_one.add_argument("--output", required=True, type=Path)
    sp_one.add_argument(
        "--weight-key", required=True,
        help='e.g. "mlp.experts.0.gate_proj" or "self_attn.q_proj"',
    )
    sp_one.add_argument(
        "--scale-granularity", default="per_row", choices=["per_row", "per_tensor"],
    )
    sp_one.add_argument(
        "--manifest-out", default=None, type=Path,
        help="optional JSON manifest write path",
    )

    sp_full = sub.add_parser(
        "full-dir",
        help="Repack a full Lynn-native model dir (manifest-driven)",
    )
    sp_full.add_argument("--input", required=True, type=Path)
    sp_full.add_argument("--output", required=True, type=Path)
    sp_full.add_argument(
        "--scale-granularity", default="per_row", choices=["per_row", "per_tensor"],
    )
    sp_full.add_argument(
        "--verify-cos-threshold", default=0.999, type=float,
        help="per-tensor cosine vs BF16 threshold; below this is a failure",
    )
    sp_full.add_argument(
        "--no-copy-sidecars", action="store_true",
        help="don't copy config.json / tokenizer files / chat template",
    )
    sp_full.add_argument(
        "--limit-quantized", default=None, type=int,
        help="stop after the first N quantized tensors (debug/smoke)",
    )
    sp_full.add_argument(
        "--no-progress", action="store_true",
        help="suppress per-tensor progress prints",
    )

    args = ap.parse_args()

    if args.cmd == "self-test":
        return self_test()
    if args.cmd == "one":
        m = repack_safetensors_weight(
            args.input, args.output, args.weight_key,
            scale_granularity=args.scale_granularity,
        )
        print(json.dumps(m, indent=2))
        if args.manifest_out is not None:
            args.manifest_out.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.cmd == "full-dir":
        summary = repack_full_dir(
            args.input,
            args.output,
            scale_granularity=args.scale_granularity,
            verify_cos_threshold=args.verify_cos_threshold,
            copy_sidecars=not args.no_copy_sidecars,
            limit_quantized=args.limit_quantized,
            progress=not args.no_progress,
        )
        # Print a compact head-of-summary; the full summary is in the file.
        compact = {
            "schema_version": summary["schema_version"],
            "output_dir": summary["output_dir"],
            "totals": summary["totals"],
            "cos_vs_bf16": {
                k: v for k, v in summary["cos_vs_bf16"].items()
                if k != "failures_below_threshold"
            },
            "cos_failures_count": summary["cos_vs_bf16"]["failures_count"],
            "deferred_count": len(summary["deferred_tensors"]),
            "elapsed_seconds": summary["elapsed_seconds"],
        }
        print(json.dumps(compact, indent=2))
        return 0 if summary["cos_vs_bf16"]["failures_count"] == 0 else 1
    print(f"unknown cmd {args.cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
