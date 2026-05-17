#!/usr/bin/env python3
"""Audit a Qwen3-style MTP sidecar before using it for Lynn training.

This is intentionally a shape/contract gate, not a training script.  It answers
whether an external `mtp.safetensors` file is a plausible architecture oracle or
initializer for Lynn 27B.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_inventory(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            rows.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "numel": tensor.numel(),
                }
            )
    return rows


def _read_readme_tensor_manifest(sidecar_dir: Path) -> list[dict[str, Any]]:
    readme = sidecar_dir / "README.md"
    if not readme.exists():
        return []
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^(mtp\.\S+)\s+shape=\[([^\]]*)\]\s+(\S+)\s*$")
    for line in readme.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        shape = [int(x.strip()) for x in match.group(2).split(",") if x.strip()]
        rows.append(
            {
                "key": match.group(1),
                "shape": shape,
                "dtype": match.group(3),
                "numel": None,
            }
        )
    return rows


def _contains_dim(shape: list[int], dim: int | None) -> bool:
    return dim is not None and dim in shape


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("text_config") or cfg


def _allowed_shape_dims(cfg: dict[str, Any]) -> set[int]:
    """Dimensions that are expected in a Qwen3-style MTP sidecar.

    The first audit only allowed hidden/intermediate/vocab dimensions, which
    incorrectly marked q/k norm vectors as suspicious for Lynn because their
    length is the attention head dimension.
    """
    keys = (
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts_per_tok",
    )
    dims = {int(cfg[key]) for key in keys if isinstance(cfg.get(key), int) and int(cfg[key]) > 0}
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim")
    n_heads = cfg.get("num_attention_heads")
    n_kv_heads = cfg.get("num_key_value_heads")
    if isinstance(hidden, int):
        dims.add(hidden * 2)
    if isinstance(head_dim, int):
        dims.add(head_dim)
        if isinstance(n_heads, int):
            dims.add(n_heads * head_dim)
        if isinstance(n_kv_heads, int):
            dims.add(n_kv_heads * head_dim)
    dims.add(1)
    return dims


def _classify(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    hidden_size = cfg.get("hidden_size")
    intermediate_size = cfg.get("intermediate_size")
    num_layers = cfg.get("num_hidden_layers")
    vocab_size = cfg.get("vocab_size")
    mtp_layers = cfg.get("mtp_num_hidden_layers")
    mtp_dedicated_embed = cfg.get("mtp_use_dedicated_embeddings")

    key_prefixes = sorted({row["key"].split(".", 1)[0] for row in rows})
    dtypes = sorted({row["dtype"] for row in rows})
    hidden_matches = [row for row in rows if _contains_dim(row["shape"], hidden_size)]
    vocab_matches = [row for row in rows if _contains_dim(row["shape"], vocab_size)]
    allowed_dims = _allowed_shape_dims(cfg)
    suspicious = [row for row in rows if row["shape"] and all(dim not in allowed_dims for dim in row["shape"])]

    has_mtp_keys = all(row["key"].startswith("mtp.") for row in rows)
    has_hidden = bool(hidden_matches)
    likely_shared_lm_head = not vocab_matches

    if not has_mtp_keys:
        decision = "RED"
        reason = "sidecar keys are not all mtp.*"
    elif not has_hidden:
        decision = "RED"
        reason = "no tensor shape contains base hidden_size"
    elif suspicious:
        decision = "AMBER"
        reason = "some sidecar tensors have dimensions not present in base config"
    else:
        decision = "GREEN"
        reason = "sidecar shapes are plausible for qwen3_next_mtp-style audit"

    return {
        "decision": decision,
        "reason": reason,
        "base_config": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "head_dim": cfg.get("head_dim"),
            "num_hidden_layers": num_layers,
            "vocab_size": vocab_size,
            "mtp_num_hidden_layers": mtp_layers,
            "mtp_use_dedicated_embeddings": mtp_dedicated_embed,
        },
        "sidecar": {
            "tensor_count": len(rows),
            "key_prefixes": key_prefixes,
            "dtypes": dtypes,
            "all_keys_start_mtp": has_mtp_keys,
            "tensors_with_hidden_size": len(hidden_matches),
            "tensors_with_vocab_size": len(vocab_matches),
            "likely_shared_lm_head": likely_shared_lm_head,
            "suspicious_shape_count": len(suspicious),
        },
        "suspicious_shapes": suspicious[:20],
        "tensors": rows,
    }


def _audit(sidecar_dir: Path, base_model: Path) -> dict[str, Any]:
    sidecar_file = sidecar_dir / "mtp.safetensors"
    base_config_path = base_model / "config.json"
    if not base_config_path.exists():
        return {
            "decision": "RED",
            "reason": f"missing base config: {base_config_path}",
            "sidecar_file": str(sidecar_file),
        }

    cfg = _text_config(_read_json(base_config_path))
    if not sidecar_file.exists():
        readme_rows = _read_readme_tensor_manifest(sidecar_dir)
        if readme_rows:
            result = _classify(readme_rows, cfg)
            result.update(
                {
                    "schema_version": "lynn-a100-mtp-sidecar-shape-audit-v1",
                    "source": "README.md tensor manifest",
                    "complete_safetensors": False,
                    "sidecar_file": str(sidecar_file),
                    "sidecar_bytes": 0,
                    "base_model": str(base_model),
                }
            )
            return result
        return {
            "decision": "WAIT",
            "reason": f"missing sidecar file: {sidecar_file}",
            "sidecar_dir": str(sidecar_dir),
            "base_model": str(base_model),
        }
    try:
        rows = _tensor_inventory(sidecar_file)
    except Exception as exc:
        readme_rows = _read_readme_tensor_manifest(sidecar_dir)
        if readme_rows:
            result = _classify(readme_rows, cfg)
            result.update(
                {
                    "schema_version": "lynn-a100-mtp-sidecar-shape-audit-v1",
                    "source": "README.md tensor manifest",
                    "complete_safetensors": False,
                    "safetensors_error": str(exc),
                    "sidecar_file": str(sidecar_file),
                    "sidecar_bytes": sidecar_file.stat().st_size,
                    "base_model": str(base_model),
                }
            )
            return result
        return {
            "decision": "WAIT",
            "reason": f"sidecar exists but is not a complete safetensors file yet: {exc}",
            "sidecar_file": str(sidecar_file),
            "sidecar_bytes": sidecar_file.stat().st_size,
            "base_model": str(base_model),
        }

    result = _classify(rows, cfg)
    result.update(
        {
        "schema_version": "lynn-a100-mtp-sidecar-shape-audit-v1",
        "source": "safetensors",
        "complete_safetensors": True,
        "sidecar_file": str(sidecar_file),
        "sidecar_bytes": sidecar_file.stat().st_size,
        "base_model": str(base_model),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    result = _audit(Path(args.sidecar_dir), Path(args.base_model))
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["decision"] in {"GREEN", "AMBER", "WAIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
