#!/usr/bin/env python3
"""Audit the official Qwen3.6-35B-A3B MTP index as a Lynn warm-start source.

This intentionally reads only config.json and model.safetensors.index.json from
Hugging Face. It does not download the heavy safetensors shards.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_BASE = f"https://huggingface.co/{DEFAULT_REPO}/raw/main"


def _fetch_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("text_config") or config


def audit(repo_base_url: str, base_hidden_size: int, timeout: int) -> dict[str, Any]:
    config = _text_config(_fetch_json(f"{repo_base_url}/config.json", timeout))
    index = _fetch_json(f"{repo_base_url}/model.safetensors.index.json", timeout)
    weight_map = index.get("weight_map", {})
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    mtp_files = sorted({weight_map[k] for k in mtp_keys})

    hidden_size = config.get("hidden_size")
    decision = "GREEN" if hidden_size == base_hidden_size and mtp_keys else "RED"
    reason = (
        "official Qwen3.6-35B-A3B MTP index matches Lynn hidden size"
        if decision == "GREEN"
        else "hidden size mismatch or missing mtp.* tensors"
    )

    return {
        "schema_version": "lynn-a100-qwen36-a3b-mtp-index-audit-v1",
        "decision": decision,
        "reason": reason,
        "source_repo": DEFAULT_REPO,
        "source_base_url": repo_base_url,
        "base_hidden_size": base_hidden_size,
        "source_config": {
            "hidden_size": hidden_size,
            "vocab_size": config.get("vocab_size"),
            "num_experts": config.get("num_experts"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "mtp_num_hidden_layers": config.get("mtp_num_hidden_layers"),
            "mtp_use_dedicated_embeddings": config.get("mtp_use_dedicated_embeddings"),
        },
        "mtp": {
            "tensor_count": len(mtp_keys),
            "keys": mtp_keys,
            "shard_files": mtp_files,
        },
        "next_steps": [
            "download only the listed MTP shards",
            "extract mtp.* tensors into a Lynn sidecar directory",
            "shape-check tensors before using them as warm-start weights",
            "fine-tune the MTP predictor on top of the W4A8-stable Lynn base",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-base-url", default=DEFAULT_BASE)
    parser.add_argument("--base-hidden-size", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out")
    args = parser.parse_args()

    result = audit(args.repo_base_url.rstrip("/"), args.base_hidden_size, args.timeout)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["decision"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
