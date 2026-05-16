#!/usr/bin/env python3
"""Lightweight A100 model inventory for BF16/W4A8/MTP planning.

This script intentionally does not load model tensors. It only reads config and
the safetensors index so it can run while a large model transfer is in progress.

It answers three operational questions:

1. Is the model directory complete enough to start training/eval?
2. Does the final BF16 artifact contain any MTP/NEXTN/draft-head tensors?
3. Which shards are still missing or partial?
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MTP_PATTERNS = (
    "mtp",
    "nextn",
    "next_n",
    "draft",
    "spec",
    "medusa",
    "eagle",
    "multi_token",
    "lookahead",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_weight_map(model_dir: Path) -> tuple[dict[str, str], str]:
    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.exists():
        return _read_json(index_path).get("weight_map", {}), index_path.name
    if single_path.exists():
        return {}, single_path.name
    return {}, "missing"


def _interesting_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "architectures",
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "num_experts_per_tok",
        "vocab_size",
        "max_position_embeddings",
        "sliding_window",
        "use_mtp",
        "num_nextn_predict_layers",
        "num_nextn_predict_tokens",
        "num_mtp_predict_layers",
        "mtp_depth",
    ]
    return {k: config.get(k) for k in keys if k in config}


def _classify_keys(keys: list[str]) -> dict[str, Any]:
    lower_hits = [
        key
        for key in keys
        if any(pattern in key.lower() for pattern in MTP_PATTERNS)
    ]
    lm_heads = [key for key in keys if key.endswith("lm_head.weight") or ".lm_head." in key]
    router_keys = [key for key in keys if ".mlp.gate." in key or key.endswith(".router.weight")]
    expert_layers: set[int] = set()
    expert_ids_by_layer: dict[int, set[int]] = {}
    layer_re = re.compile(r"layers\.(\d+)\.")
    expert_re = re.compile(r"experts\.(\d+)\.")
    for key in keys:
        if ".mlp.experts." not in key:
            continue
        lm = layer_re.search(key)
        if lm:
            layer = int(lm.group(1))
            expert_layers.add(layer)
            em = expert_re.search(key)
            if em:
                expert_ids_by_layer.setdefault(layer, set()).add(int(em.group(1)))
    expert_counts = {
        str(layer): len(ids)
        for layer, ids in sorted(expert_ids_by_layer.items())
        if ids
    }
    return {
        "mtp_nextn_like_key_count": len(lower_hits),
        "mtp_nextn_like_key_sample": lower_hits[:50],
        "lm_head_key_count": len(lm_heads),
        "lm_head_key_sample": lm_heads[:10],
        "router_key_count": len(router_keys),
        "expert_layer_count": len(expert_layers),
        "expert_counts_by_layer_sample": dict(list(expert_counts.items())[:10]),
        "expert_counts_by_layer_tail": dict(list(expert_counts.items())[-10:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    model_dir = Path(args.model)
    config = _read_json(model_dir / "config.json")
    generation_config = _read_json(model_dir / "generation_config.json")
    tokenizer_config = _read_json(model_dir / "tokenizer_config.json")
    weight_map, index_source = _load_weight_map(model_dir)

    shard_names = sorted(set(weight_map.values()))
    missing_shards = [name for name in shard_names if not (model_dir / name).exists()]
    present_shards = [name for name in shard_names if (model_dir / name).exists()]
    shard_sizes = {name: (model_dir / name).stat().st_size for name in present_shards}
    non_shard_files = sorted(
        p.name for p in model_dir.iterdir()
        if p.is_file() and not p.name.endswith(".safetensors")
    ) if model_dir.exists() else []
    keys = sorted(weight_map.keys())
    result = {
        "schema_version": "lynn-a100-model-inventory-v1",
        "model": str(model_dir),
        "exists": model_dir.exists(),
        "index_source": index_source,
        "config": _interesting_config(config),
        "generation_config_keys": sorted(generation_config.keys()),
        "tokenizer_chat_template_present": bool(tokenizer_config.get("chat_template")),
        "weight_key_count": len(keys),
        "shard_count": len(shard_names),
        "present_shard_count": len(present_shards),
        "missing_shard_count": len(missing_shards),
        "missing_shards": missing_shards[:100],
        "present_shard_bytes": sum(shard_sizes.values()),
        "largest_present_shards": sorted(shard_sizes.items(), key=lambda kv: kv[1], reverse=True)[:10],
        "non_shard_files": non_shard_files,
        "key_classification": _classify_keys(keys),
        "ready_for_config_only_checks": bool(config and tokenizer_config),
        "ready_for_tensor_load": bool(weight_map and not missing_shards),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["ready_for_config_only_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
