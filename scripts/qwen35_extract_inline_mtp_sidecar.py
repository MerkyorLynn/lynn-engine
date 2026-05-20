#!/usr/bin/env python3
"""Extract inline Qwen3.5-9B ``mtp.*`` tensors into a Lynn sidecar.

Qwen3.5-9B packages its single-layer NextN/MTP head inside the main BF16
model shards instead of a standalone ``mtp.safetensors`` file.  Lynn's runtime
expects an external sidecar path, so this tool copies exactly the inline
``mtp.*`` tensors into ``<sidecar-dir>/mtp.safetensors`` and writes a compact
JSON inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


EXPECTED_KEYS = {
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
}


def _sha256_prefix(path: Path, *, chars: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:chars]


def _model_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        files = sorted(set((data.get("weight_map") or {}).values()))
        return [model_dir / name for name in files]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in {model_dir}")
    return shards


def extract(model_dir: Path, sidecar_dir: Path) -> dict[str, Any]:
    tensors = {}
    source_files: dict[str, str] = {}
    inventory: dict[str, Any] = {}
    for shard in _model_shards(model_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith("mtp."):
                    continue
                tensor = f.get_tensor(key)
                tensors[key] = tensor
                source_files[key] = shard.name
                inventory[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": int(tensor.numel()),
                }

    missing = sorted(EXPECTED_KEYS - set(tensors))
    unexpected = sorted(set(tensors) - EXPECTED_KEYS)
    if missing:
        raise RuntimeError(f"missing expected Qwen3.5-9B MTP tensors: {missing}")

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_file = sidecar_dir / "mtp.safetensors"
    save_file(tensors, str(sidecar_file), metadata={"source": str(model_dir), "format": "qwen35-9b-inline-mtp"})
    return {
        "schema_version": "lynn-qwen35-9b-inline-mtp-extract-v1",
        "model_dir": str(model_dir),
        "sidecar_dir": str(sidecar_dir),
        "sidecar_file": str(sidecar_file),
        "sidecar_bytes": sidecar_file.stat().st_size,
        "sidecar_sha256_16": _sha256_prefix(sidecar_file),
        "tensor_count": len(tensors),
        "missing_expected": missing,
        "unexpected_mtp_keys": unexpected,
        "source_files": source_files,
        "tensors": inventory,
        "decision": "GREEN: extracted Qwen3.5-9B inline MTP tensors into a Lynn sidecar.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = extract(Path(args.model_dir), Path(args.sidecar_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
