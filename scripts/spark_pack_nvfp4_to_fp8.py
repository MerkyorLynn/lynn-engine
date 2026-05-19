"""Offline NVFP4 → FP8 E4M3 repack tool (Spark sm_121).

Spark has no native FP4 MMA but real native FP8 MMA at 162 TFLOPS. The naive
inference-time NVFP4 → FP8 cast lost (32-76% TPS) because the per-call
activation cast + layout transform fought the M=1 GEMM. This tool moves the
weight cast offline:

* Read Lynn NVFP4 packed weights via the existing `lynn_quant_manifest.json`.
* Dequantize FP4 → BF16 with the canonical slow path.
* Quantize BF16 → FP8 E4M3 with per-row scale (per-output-channel).
* Store the FP8 packed tensor row-major. The runtime takes ``.t()`` to expose
  the column-major view ``torch._scaled_mm`` expects for ``B``.

Output files live next to the NVFP4 checkpoint as a sibling
``fp8e4m3/`` directory containing safetensors shards plus a fresh
``lynn_fp8_manifest.json``. The original NVFP4 artifact is untouched —
``LYNN_SPARK_FP8_FUSED=1`` opts into the new path; absence keeps the W4A16
path live.

CPU-validatable: ``--self-test`` runs a synthetic NVFP4 round-trip and
verifies cos > 0.999 with no GPU and no model on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.dequant import (  # noqa: E402  (sys.path mutation)
    E2M1_TO_FLOAT,
    dequantize_nvfp4_v8_rtn_weight,
    unpack_fp4_e2m1_from_uint8,
)

SCHEMA_VERSION = "lynn-fp8-e4m3-pack-v1"
FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Core quantization (BF16 → FP8 E4M3 per-row)
# ---------------------------------------------------------------------------

def quantize_bf16_to_fp8_per_row(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-row absmax scale + E4M3 cast.

    Returns:
        packed:  FP8 E4M3 view as uint8 storage, shape ``[O, I]``.
        scale:   FP32 ``[O]`` per-row scale s.t. ``bf16 ≈ fp8 * scale[:, None]``.

    Per-row (per-output-channel) keeps scale storage tiny (O floats) while
    capturing the bulk of the dynamic-range variation between output channels.
    Spark ``_scaled_mm`` accepts a per-row ``scale_b`` vector directly.
    """
    if weight_bf16.ndim != 2:
        raise ValueError(f"weight must be 2D [O, I], got {tuple(weight_bf16.shape)}")
    w32 = weight_bf16.float()
    # ``clamp_min`` prevents zero-scale rows (all-zero output channels) from
    # producing NaN on divide.
    absmax = w32.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    scale = absmax / FP8_E4M3_MAX
    normalized = (w32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    fp8 = normalized.to(FP8_DTYPE)
    return fp8, scale.squeeze(1).float().contiguous()


def dequantize_fp8_per_row(
    fp8_packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Inverse of :func:`quantize_bf16_to_fp8_per_row` for correctness checks."""
    if fp8_packed.dtype is not FP8_DTYPE:
        fp8_packed = fp8_packed.view(FP8_DTYPE)
    w32 = fp8_packed.float() * scale.float().unsqueeze(1)
    return w32.to(output_dtype)


# ---------------------------------------------------------------------------
# NVFP4 → FP8 repack
# ---------------------------------------------------------------------------

def nvfp4_to_fp8_per_row(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lynn NVFP4 (E2M1 + per-16 scale + global) → FP8 E4M3 per-row.

    Dequantization uses the canonical CPU path
    (``dequantize_nvfp4_v8_rtn_weight``) so the FP8 artifact matches the
    BF16 reference bit-for-bit on the input side. The lossy step is exclusively
    the per-row FP8 cast.
    """
    bf16 = dequantize_nvfp4_v8_rtn_weight(
        weight_packed,
        weight_scale,
        weight_global_scale=weight_global_scale,
        output_dtype=torch.bfloat16,
    )
    return quantize_bf16_to_fp8_per_row(bf16)


# ---------------------------------------------------------------------------
# Cosine similarity (small-batch friendly)
# ---------------------------------------------------------------------------

def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.flatten().float()
    bv = b.flatten().float()
    na = av.norm()
    nb = bv.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return float("nan")
    return float((av @ bv) / (na * nb))


# ---------------------------------------------------------------------------
# Self-test (no GPU, no model)
# ---------------------------------------------------------------------------

def _synthesize_nvfp4(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU NVFP4 packer for synthetic test data.

    Mirrors Lynn's per-16 quantizer used in :mod:`engine.nvfp4_runtime` /
    :mod:`engine.dequant`: per-16 absmax → quantize to E2M1 indices → pack
    two FP4 codes per uint8.
    """
    if weight_bf16.ndim != 2 or weight_bf16.shape[1] % 16 != 0:
        raise ValueError("expected 2D weight with I divisible by 16")
    w = weight_bf16.float()
    O, I = w.shape
    groups = I // 16
    table = E2M1_TO_FLOAT  # ``[0, .5, 1, 1.5, 2, 3, 4, 6]``
    grouped = w.reshape(O, groups, 16)

    # Per-group local scale and global scale.
    local_max = grouped.abs().amax(dim=-1)              # [O, groups]
    global_scale = local_max.amax().clamp_min(1.0e-6) / float(table[-1])
    weight_scale = (local_max / float(table[-1])).clamp_min(1.0e-8)

    normalized = grouped.abs() / weight_scale.unsqueeze(-1)
    # Nearest-magnitude lookup against the LUT.
    diffs = (normalized.unsqueeze(-1) - table.view(1, 1, 1, -1)).abs()
    mag = diffs.argmin(dim=-1).to(torch.uint8)
    sign = (grouped < 0).to(torch.uint8) * 8
    codes = (mag | sign).reshape(O, I)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).contiguous()

    # Scale storage matches compressed-tensors convention: scale baked with the
    # global scale, i.e., ``effective_scale = weight_scale * global_scale``.
    # The dequant path then divides by ``global_scale`` to recover ``weight_scale``.
    # We store ``weight_scale * global_scale`` and a separate global.
    stored_scale = (weight_scale * global_scale).contiguous()
    return packed, stored_scale, global_scale.reshape(1)


def run_self_test(seed: int = 0, *, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    O, I = 256, 1024
    weight = torch.randn(O, I, dtype=torch.bfloat16) * 0.02

    packed, scale, global_scale = _synthesize_nvfp4(weight)
    bf16_from_fp4 = dequantize_nvfp4_v8_rtn_weight(
        packed, scale, weight_global_scale=global_scale, output_dtype=torch.bfloat16
    )
    fp8_packed, fp8_scale = nvfp4_to_fp8_per_row(packed, scale, global_scale)
    bf16_from_fp8 = dequantize_fp8_per_row(fp8_packed, fp8_scale, output_dtype=torch.bfloat16)

    cos_fp4_vs_orig = _flat_cos(weight, bf16_from_fp4)
    cos_fp8_vs_fp4 = _flat_cos(bf16_from_fp4, bf16_from_fp8)
    cos_fp8_vs_orig = _flat_cos(weight, bf16_from_fp8)

    nvfp4_bytes = packed.numel() + scale.numel() * 4 + global_scale.numel() * 4
    fp8_bytes = fp8_packed.numel() + fp8_scale.numel() * 4
    size_ratio = fp8_bytes / nvfp4_bytes

    result = {
        "shape": [O, I],
        "cos_nvfp4_vs_original_bf16": cos_fp4_vs_orig,
        "cos_fp8_vs_nvfp4": cos_fp8_vs_fp4,
        "cos_fp8_vs_original_bf16": cos_fp8_vs_orig,
        "nvfp4_bytes": nvfp4_bytes,
        "fp8_bytes": fp8_bytes,
        "size_ratio_fp8_over_nvfp4": size_ratio,
        "passes_cos_gate": cos_fp8_vs_fp4 > 0.999,
        "passes_size_gate": size_ratio <= 2.10,
    }
    if verbose:
        print("[self-test] NVFP4 → FP8 repack on synthetic [256, 1024] BF16 weight")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"  {k:<32s} = {v:.6f}")
            else:
                print(f"  {k:<32s} = {v}")
        if not result["passes_cos_gate"]:
            print("  FAIL: cos_fp8_vs_nvfp4 below 0.999 gate", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Real-model repack
# ---------------------------------------------------------------------------

def _read_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as st:
            return {k: single.name for k in st.keys()}
    raise FileNotFoundError(
        f"Expected model.safetensors.index.json or model.safetensors under {model_dir}"
    )


def _load_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    file_name = weight_map[key]
    with safe_open(model_dir / file_name, framework="pt", device="cpu") as st:
        return st.get_tensor(key)


def _iter_nvfp4_records(manifest: dict) -> list[tuple[str, dict]]:
    quantized = manifest.get("quantized_tensors", {})
    return list(quantized.items())


def repack_model(
    model_dir: Path,
    output_dir: Path,
    *,
    shard_bytes: int = 4 * 1024 * 1024 * 1024,
    check_cos_threshold: float = 0.999,
    sample_check_count: int = 8,
) -> dict:
    """Repack every NVFP4 tensor in ``model_dir`` to FP8 under ``output_dir``.

    The output mirrors a stand-alone safetensors checkpoint:

    * shards named ``model-fp8e4m3-NNNNN-of-MMMMM.safetensors``
    * ``lynn_fp8_manifest.json`` describing per-tensor packed/scale keys.

    A small random sample is re-loaded after writing and dequantized to verify
    ``cos > check_cos_threshold`` against the NVFP4 dequant. This catches
    file-IO bugs without a full GPU smoke.
    """
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = model_dir / "lynn_quant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{model_dir} has no lynn_quant_manifest.json — Lynn-native NVFP4 only"
        )
    manifest = json.loads(manifest_path.read_text())
    records = _iter_nvfp4_records(manifest)
    if not records:
        raise ValueError(f"{manifest_path} has empty quantized_tensors")

    weight_map = _read_weight_map(model_dir)

    fp8_manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "source_nvfp4_manifest": str(manifest_path.name),
        "quantized_tensors": {},
        "shards": [],
    }

    shard_idx = 0
    shard_buffer: dict[str, torch.Tensor] = {}
    shard_bytes_acc = 0
    written_shards: list[str] = []

    def _current_shard_name() -> str:
        return f"model-fp8e4m3-{shard_idx:05d}.safetensors"

    def _flush_shard():
        nonlocal shard_idx, shard_buffer, shard_bytes_acc
        if not shard_buffer:
            return
        shard_name = _current_shard_name()
        save_file(shard_buffer, str(output_dir / shard_name))
        written_shards.append(shard_name)
        shard_idx += 1
        shard_buffer = {}
        shard_bytes_acc = 0

    skipped: list[str] = []
    n_tensors = 0
    total_in_bytes = 0
    total_out_bytes = 0

    for base_key, rec in records:
        packed_key = rec.get("packed_key")
        scale_key = rec.get("scale_key")
        global_key = rec.get("global_scale_key")
        original_shape = rec.get("original_shape")
        if not all((packed_key, scale_key, global_key, original_shape)):
            skipped.append(f"{base_key} (incomplete manifest record)")
            continue
        if any(k not in weight_map for k in (packed_key, scale_key, global_key)):
            skipped.append(f"{base_key} (missing tensors in weight_map)")
            continue

        weight_packed = _load_tensor(model_dir, weight_map, packed_key)
        weight_scale = _load_tensor(model_dir, weight_map, scale_key).float()
        weight_global_scale = _load_tensor(model_dir, weight_map, global_key).float()

        # Lynn variable-expert tensors flatten ``[experts, O, I]`` to 2D in the
        # shard. We repack per-expert (matching the runtime ``[E, O, I]`` view
        # used by ``_expert_ffn``).
        is_grouped = len(original_shape) == 3
        if is_grouped:
            experts, O, I = map(int, original_shape)
            packed_3d = weight_packed.reshape(experts, O, I // 2)
            scale_3d = weight_scale.reshape(experts, O, I // 16)
            fp8_groups = []
            scale_groups = []
            for e in range(experts):
                fp8_e, fp8_scale_e = nvfp4_to_fp8_per_row(
                    packed_3d[e].contiguous(),
                    scale_3d[e].contiguous(),
                    weight_global_scale,
                )
                fp8_groups.append(fp8_e)
                scale_groups.append(fp8_scale_e)
            fp8_packed = torch.stack(fp8_groups, dim=0).contiguous()
            fp8_scale = torch.stack(scale_groups, dim=0).contiguous()
        elif len(original_shape) == 2:
            fp8_packed, fp8_scale = nvfp4_to_fp8_per_row(
                weight_packed.contiguous(),
                weight_scale.contiguous(),
                weight_global_scale,
            )
        else:
            skipped.append(f"{base_key} (unsupported original_shape {original_shape})")
            continue

        # Store FP8 as uint8 (safetensors does not yet stably round-trip FP8
        # dtypes across all builds). The loader views ``.uint8`` →
        # ``torch.float8_e4m3fn`` at load time.
        fp8_packed_u8 = fp8_packed.view(torch.uint8).contiguous()

        fp8_packed_name = base_key + ".fp8e4m3_packed"
        fp8_scale_name = base_key + ".fp8e4m3_scale"
        fp8_global_name = base_key + ".fp8e4m3_global_scale"

        # ``global_scale`` is preserved for downstream tools that need to relate
        # FP8 back to the original NVFP4 absolute magnitude. It does not enter
        # the runtime _scaled_mm call.
        global_tensor = weight_global_scale.float().reshape(1).contiguous()

        for name, tensor in (
            (fp8_packed_name, fp8_packed_u8),
            (fp8_scale_name, fp8_scale),
            (fp8_global_name, global_tensor),
        ):
            shard_buffer[name] = tensor
            shard_bytes_acc += tensor.numel() * tensor.element_size()

        # All three tensors for this base_key now live in the current shard, so
        # the recorded ``shard`` name is correct. Flush only at base_key
        # boundaries to keep each base_key's tensors co-located.
        fp8_manifest["quantized_tensors"][base_key] = {
            "fp8_packed_key": fp8_packed_name,
            "fp8_scale_key": fp8_scale_name,
            "fp8_global_scale_key": fp8_global_name,
            "original_shape": list(map(int, original_shape)),
            "shard": _current_shard_name(),
        }

        if shard_bytes_acc >= shard_bytes:
            _flush_shard()

        n_tensors += 1
        total_in_bytes += weight_packed.numel() + weight_scale.numel() * 4
        total_out_bytes += fp8_packed_u8.numel() + fp8_scale.numel() * 4

    _flush_shard()

    fp8_manifest["shards"] = written_shards
    fp8_manifest["summary"] = {
        "tensors_repacked": n_tensors,
        "tensors_skipped": skipped,
        "input_bytes_approx": total_in_bytes,
        "output_bytes_approx": total_out_bytes,
        "size_ratio_fp8_over_nvfp4": (
            total_out_bytes / total_in_bytes if total_in_bytes else None
        ),
    }

    out_manifest_path = output_dir / "lynn_fp8_manifest.json"
    out_manifest_path.write_text(json.dumps(fp8_manifest, indent=2))

    # Sample correctness check: re-open shards and compare cos for a few tensors.
    check_results: list[dict] = []
    if sample_check_count > 0 and n_tensors > 0:
        sample_keys = list(fp8_manifest["quantized_tensors"].keys())[:sample_check_count]
        # Build a fresh weight_map for the output shards.
        out_weight_map: dict[str, str] = {}
        for shard_name in written_shards:
            with safe_open(output_dir / shard_name, framework="pt", device="cpu") as st:
                for k in st.keys():
                    out_weight_map[k] = shard_name
        for base_key in sample_keys:
            rec_out = fp8_manifest["quantized_tensors"][base_key]
            rec_in = manifest["quantized_tensors"][base_key]
            try:
                fp8_packed = _load_tensor(output_dir, out_weight_map, rec_out["fp8_packed_key"])
                fp8_scale = _load_tensor(output_dir, out_weight_map, rec_out["fp8_scale_key"])
                packed_in = _load_tensor(model_dir, weight_map, rec_in["packed_key"])
                scale_in = _load_tensor(model_dir, weight_map, rec_in["scale_key"]).float()
                global_in = _load_tensor(model_dir, weight_map, rec_in["global_scale_key"]).float()
            except Exception as exc:
                check_results.append({"key": base_key, "ok": False, "error": str(exc)})
                continue
            shape = rec_out["original_shape"]
            if len(shape) == 3:
                experts = shape[0]
                fp8_packed_3d = fp8_packed.reshape(experts, shape[1], shape[2])
                fp8_scale_3d = fp8_scale.reshape(experts, shape[1])
                packed_3d = packed_in.reshape(experts, shape[1], shape[2] // 2)
                scale_3d = scale_in.reshape(experts, shape[1], shape[2] // 16)
                bf16_fp8 = torch.cat(
                    [
                        dequantize_fp8_per_row(fp8_packed_3d[e], fp8_scale_3d[e])
                        for e in range(experts)
                    ],
                    dim=0,
                )
                bf16_fp4 = torch.cat(
                    [
                        dequantize_nvfp4_v8_rtn_weight(
                            packed_3d[e], scale_3d[e], weight_global_scale=global_in
                        )
                        for e in range(experts)
                    ],
                    dim=0,
                )
            else:
                bf16_fp8 = dequantize_fp8_per_row(fp8_packed, fp8_scale)
                bf16_fp4 = dequantize_nvfp4_v8_rtn_weight(
                    packed_in, scale_in, weight_global_scale=global_in
                )
            cos = _flat_cos(bf16_fp4, bf16_fp8)
            check_results.append(
                {"key": base_key, "ok": cos > check_cos_threshold, "cos": cos}
            )

    fp8_manifest["sample_checks"] = check_results
    out_manifest_path.write_text(json.dumps(fp8_manifest, indent=2))

    return fp8_manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Lynn NVFP4 model directory (contains lynn_quant_manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for FP8 shards + manifest (default: <model>/fp8e4m3/)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic CPU NVFP4 → FP8 round-trip and exit (no model needed)",
    )
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=0.999,
        help="cos similarity threshold for sample re-load check (default 0.999)",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=8,
        help="Number of tensors to re-load and verify after repack (default 8)",
    )
    args = parser.parse_args()

    if args.self_test:
        result = run_self_test()
        ok = bool(result["passes_cos_gate"] and result["passes_size_gate"])
        return 0 if ok else 1

    if args.model_dir is None:
        parser.error("--model-dir or --self-test is required")

    output_dir = args.output_dir or (args.model_dir / "fp8e4m3")
    fp8_manifest = repack_model(
        args.model_dir,
        output_dir,
        check_cos_threshold=args.cos_threshold,
        sample_check_count=args.sample_checks,
    )
    summary = fp8_manifest["summary"]
    print(f"[repack] {args.model_dir} → {output_dir}")
    print(f"  tensors repacked: {summary['tensors_repacked']}")
    if summary["tensors_skipped"]:
        print(f"  skipped: {summary['tensors_skipped']}")
    if summary["size_ratio_fp8_over_nvfp4"] is not None:
        print(f"  size ratio fp8/nvfp4: {summary['size_ratio_fp8_over_nvfp4']:.3f}")
    failures = [c for c in fp8_manifest.get("sample_checks", []) if not c["ok"]]
    if failures:
        print(f"  sample-check FAIL ({len(failures)}/{len(fp8_manifest['sample_checks'])}):")
        for c in failures:
            print(f"    {c}")
        return 1
    print(f"  sample-checks: {len(fp8_manifest.get('sample_checks', []))}/8 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
