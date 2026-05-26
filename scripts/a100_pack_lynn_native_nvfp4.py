#!/usr/bin/env python3
"""Pack a BF16/Lynn folded artifact into Lynn-native per-16 NVFP4.

This intentionally targets the Lynn engine manifest format, not vendor
compressed-tensors. It is the fastest path after W4A8 Recovery because it can
consume a copy-on-write folded artifact directly and preserves physical
variable-expert tensor shapes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_metadata(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name.endswith(".safetensors") or child.name == "model.safetensors.index.json":
            continue
        # Folded BF16 artifacts may carry a legacy `tensors/` side directory
        # with copy-on-write shards and many symlinks. Lynn-native NVFP4 runtime
        # consumes only the top-level model shard index produced below; copying
        # `tensors/` bloats v0/v1 packages by ~11 GiB and can introduce broken
        # symlinks when the source overlay moves.
        if child.name == "tensors":
            continue
        dst = out / child.name
        if child.is_dir():
            if not dst.exists():
                shutil.copytree(child, dst, symlinks=True)
        elif not dst.exists():
            shutil.copy2(child, dst)


def _should_quantize(key: str, tensor: torch.Tensor, keep_re: re.Pattern[str] | None) -> bool:
    if keep_re is not None and keep_re.search(key):
        return False
    is_weight = key.endswith(".weight") or key.endswith(
        ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    )
    if not is_weight:
        return False
    if not tensor.is_floating_point() or tensor.ndim < 2:
        return False
    if int(tensor.shape[-1]) % 16 != 0:
        return False
    return True


def _quantize_per16_e2m1(
    tensor: torch.Tensor,
    *,
    row_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    original_shape = list(tensor.shape)
    k = int(original_shape[-1])
    rows = int(tensor.numel() // k)
    x = tensor.detach().reshape(rows, k)
    row_chunk_size = max(1, int(row_chunk_size))
    packed_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    table = E2M1

    # Keep the peak below Spark's memory ceiling. The old full-tensor path
    # materialized [rows, groups, 16, 8] distance scores for large MoE weights
    # and could be OOM-killed before the first shard finished.
    for start in range(0, rows, row_chunk_size):
        stop = min(rows, start + row_chunk_size)
        x_chunk = x[start:stop].to(torch.float32)
        xg = x_chunk.reshape(stop - start, k // 16, 16)
        scale = (xg.abs().amax(dim=-1) / float(E2M1[-1])).clamp_min(1.0e-8)
        normalized = xg.abs() / scale.unsqueeze(-1)
        mag = torch.argmin((normalized.unsqueeze(-1) - table.view(1, 1, 1, 8)).abs(), dim=-1)
        sign = (xg < 0).to(torch.uint8) << 3
        codes = (mag.to(torch.uint8) | sign).reshape(stop - start, k)
        packed_chunks.append((codes[:, 0::2] | (codes[:, 1::2] << 4)).cpu().contiguous())
        scale_chunks.append(scale.to(torch.float16).cpu().contiguous())

    packed = torch.cat(packed_chunks, dim=0)
    scale = torch.cat(scale_chunks, dim=0)
    return packed, scale, torch.ones((), dtype=torch.float32)


def _patch_config(out: Path) -> None:
    cfg_path = out / "config.json"
    if not cfg_path.exists():
        return
    cfg = _read_json(cfg_path)
    cfg["quantization_config"] = {
        "quant_method": "lynn_native",
        "format": "nvfp4_e2m1_rowwise_per_16",
        "weight_activation_contract": "W4A16_weight_only",
        "activation_dtype": "bf16",
        "activation_contract": "w4a8_required_if_folded_manifest_present",
    }
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--keep-regex", default=r"(embed_tokens|lm_head|rotary|norm|mlp\.gate\.weight)")
    parser.add_argument("--max-shard-bytes", type=int, default=3_500_000_000)
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=1024,
        help="Rows per quantization chunk; lower uses less memory and more time.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src_model).resolve()
    out = Path(args.out_model).resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists: {out}")
        if out == src:
            raise SystemExit("refusing to overwrite source model")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    _copy_metadata(src, out)

    weight_map = _read_json(src / "model.safetensors.index.json")["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key, rel in weight_map.items():
        by_file.setdefault(rel, []).append(key)

    keep_re = re.compile(args.keep_regex) if args.keep_regex else None
    out_weight_map: dict[str, str] = {}
    manifest: dict[str, Any] = {
        "schema_version": "lynn-variable-nvfp4-pack-v1",
        "source_model": str(src),
        "out_model": str(out),
        "quantization": {
            "format": "nvfp4_e2m1_rowwise_per_16",
            "weight_activation_contract": "W4A16_weight_only",
            "activation_dtype": "bf16",
            "group_size": 16,
            "global_scale_policy": "unit",
            "keep_regex": args.keep_regex,
        },
        "runtime_contract": {
            "inference_path_required": "w4a8"
            if (src / "lynn_w4a8_alpha_fold_manifest.json").exists()
            else "nvfp4",
            "fallback_path_allowed": not (src / "lynn_w4a8_alpha_fold_manifest.json").exists(),
        },
        "kept_tensors": {},
        "quantized_tensors": {},
        "source_files": {},
    }

    shard_id = 0
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal shard_id, pending, pending_bytes
        if not pending:
            return
        shard_id += 1
        rel = f"model-{shard_id:05d}.safetensors"
        save_file(pending, out / rel, metadata={"format": "pt", "lynn_native_nvfp4": "true"})
        for k in pending:
            out_weight_map[k] = rel
        pending = {}
        pending_bytes = 0

    start = time.time()
    for rel, keys in sorted(by_file.items()):
        path = src / rel
        file_rec = {"keys": len(keys), "quantized": 0, "kept": 0}
        with safe_open(path, framework="pt", device="cpu") as st:
            for key in keys:
                tensor = st.get_tensor(key)
                if _should_quantize(key, tensor, keep_re):
                    packed, scale, global_scale = _quantize_per16_e2m1(
                        tensor,
                        row_chunk_size=args.row_chunk_size,
                    )
                    packed_key = key + ".packed"
                    scale_key = key + ".scale"
                    global_key = key + ".global_scale"
                    tensors = {
                        packed_key: packed,
                        scale_key: scale,
                        global_key: global_scale,
                    }
                    manifest["quantized_tensors"][key] = {
                        "packed_key": packed_key,
                        "scale_key": scale_key,
                        "global_scale_key": global_key,
                        "original_shape": list(tensor.shape),
                        "original_dtype": str(tensor.dtype).replace("torch.", ""),
                        "packed_shape": list(packed.shape),
                        "scale_shape": list(scale.shape),
                    }
                    file_rec["quantized"] += 1
                else:
                    tensors = {key: tensor.cpu()}
                    manifest["kept_tensors"][key] = {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                    }
                    file_rec["kept"] += 1
                for out_key, out_tensor in tensors.items():
                    nbytes = out_tensor.numel() * out_tensor.element_size()
                    if pending and pending_bytes + nbytes > args.max_shard_bytes:
                        flush()
                    pending[out_key] = out_tensor
                    pending_bytes += nbytes
        manifest["source_files"][rel] = file_rec
        print(
            f"[pack] {rel}: quantized={file_rec['quantized']} kept={file_rec['kept']} "
            f"elapsed={time.time()-start:.1f}s",
            flush=True,
        )
    flush()

    index = {
        "metadata": {
            "total_size": sum((out / rel).stat().st_size for rel in set(out_weight_map.values())),
            "lynn_native_nvfp4": "true",
        },
        "weight_map": out_weight_map,
    }
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["elapsed_seconds"] = time.time() - start
    manifest["output_shards"] = shard_id
    manifest["quantized_count"] = len(manifest["quantized_tensors"])
    manifest["kept_count"] = len(manifest["kept_tensors"])
    if (src / "lynn_w4a8_alpha_fold_manifest.json").exists():
        manifest["w4a8_fold_manifest"] = _read_json(src / "lynn_w4a8_alpha_fold_manifest.json")
    (out / "lynn_quant_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _patch_config(out)
    print(json.dumps({
        "out_model": str(out),
        "output_shards": shard_id,
        "quantized_count": manifest["quantized_count"],
        "kept_count": manifest["kept_count"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
