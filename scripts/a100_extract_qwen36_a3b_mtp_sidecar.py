#!/usr/bin/env python3
"""Download and extract the official Qwen3.6-35B-A3B `mtp.*` tensors.

This keeps MTP prep lightweight by downloading only the two shard files that
contain `mtp.*` weights, then emits a Lynn-friendly `mtp.safetensors` sidecar.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_REPO = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_BASE = f"https://huggingface.co/{DEFAULT_REPO}/resolve/main"


def _fetch_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _download(url: str, dest: Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(dest)


def _ensure_file(url: str, dest: Path, timeout: int) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    _download(url, dest, timeout)


def _extract_mtp(shards: list[Path], out_file: Path) -> dict[str, Any]:
    tensors = {}
    source_files: dict[str, str] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith("mtp."):
                    continue
                tensors[key] = f.get_tensor(key)
                source_files[key] = shard.name

    if not tensors:
        raise RuntimeError("no mtp.* tensors found in downloaded shards")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_file))
    return {
        "tensor_count": len(tensors),
        "keys": sorted(tensors.keys()),
        "source_files": source_files,
        "sidecar_file": str(out_file),
        "sidecar_bytes": out_file.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-base-url", default=DEFAULT_BASE)
    ap.add_argument("--base-hidden-size", type=int, default=2048)
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--sidecar-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    base = args.repo_base_url.rstrip("/")
    cfg = _fetch_json(f"{base}/config.json", args.timeout)
    text_cfg = cfg.get("text_config") or cfg
    hidden_size = text_cfg.get("hidden_size")
    index = _fetch_json(f"{base}/model.safetensors.index.json", args.timeout)
    weight_map = index.get("weight_map", {})
    mtp_keys = sorted(key for key in weight_map if key.startswith("mtp."))
    shard_files = sorted({weight_map[key] for key in mtp_keys})

    if hidden_size != args.base_hidden_size:
        raise RuntimeError(f"hidden size mismatch: {hidden_size} != {args.base_hidden_size}")
    if not mtp_keys or not shard_files:
        raise RuntimeError("index contains no mtp.* tensors")

    download_dir = Path(args.download_dir)
    sidecar_dir = Path(args.sidecar_dir)
    shard_paths = []
    for shard in shard_files:
        path = download_dir / shard
        _ensure_file(f"{base}/{shard}", path, args.timeout)
        shard_paths.append(path)

    sidecar_file = sidecar_dir / "mtp.safetensors"
    extracted = _extract_mtp(shard_paths, sidecar_file)
    result = {
        "schema_version": "lynn-a100-qwen36-a3b-mtp-sidecar-extract-v1",
        "source_repo": DEFAULT_REPO,
        "source_base_url": base,
        "base_hidden_size": args.base_hidden_size,
        "source_hidden_size": hidden_size,
        "download_dir": str(download_dir),
        "downloaded_shards": [str(path) for path in shard_paths],
        "mtp": extracted,
        "decision": "GREEN: extracted official Qwen3.6-35B-A3B mtp.* tensors into a Lynn sidecar directory.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
