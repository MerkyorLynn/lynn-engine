#!/usr/bin/env python3
"""Generate and verify Qwen3.5-9B release checksum manifests.

Manifest format is tab-separated:

    <sha256>\t<size_bytes>\t<relative_path>

Only Python stdlib is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str
    size: int
    path: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rel(path: Path, base: Path) -> str:
    rel = path.relative_to(base).as_posix()
    if rel == "." or rel.startswith("../") or rel.startswith("/"):
        raise ValueError(f"path is not safely relative to base: {path}")
    return str(PurePosixPath(rel))


def iter_input_files(paths: Iterable[Path]) -> tuple[Path, list[Path]]:
    raw_paths = [p.expanduser().resolve() for p in paths]
    if not raw_paths:
        raise ValueError("generate requires at least one --paths entry")

    missing = [str(p) for p in raw_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input path(s): " + ", ".join(missing))

    base_candidates = [p if p.is_dir() else p.parent for p in raw_paths]
    common_base = Path(os.path.commonpath([str(p) for p in base_candidates]))
    if len(raw_paths) == 1 and raw_paths[0].is_dir():
        common_base = raw_paths[0].parent

    files: list[Path] = []
    for path in raw_paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
        else:
            raise ValueError(f"unsupported input path: {path}")

    unique = sorted(set(files), key=lambda p: normalize_rel(p, common_base))
    return common_base, unique


def parse_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected '<sha256>\\t<size_bytes>\\t<relative_path>'"
                )
            digest, size_raw, rel = parts
            if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
                raise ValueError(f"{path}:{line_number}: invalid sha256 digest")
            try:
                size = int(size_raw)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid size: {size_raw}") from exc
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts or rel in {"", "."}:
                raise ValueError(f"{path}:{line_number}: unsafe relative path: {rel}")
            entries.append(ManifestEntry(digest.lower(), size, rel))
    return entries


def cmd_generate(args: argparse.Namespace) -> int:
    base, files = iter_input_files([Path(p) for p in args.paths])
    rows: list[ManifestEntry] = []
    for file_path in files:
        rel = normalize_rel(file_path, base)
        rows.append(ManifestEntry(sha256_file(file_path), file_path.stat().st_size, rel))

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row.sha256}\t{row.size}\t{row.path}\n")

    summary = {
        "ok": True,
        "mode": "generate",
        "manifest": str(out),
        "base": str(base),
        "file_count": len(rows),
        "total_bytes": sum(row.size for row in rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def verify_entries(entries: list[ManifestEntry], root: Path) -> tuple[bool, list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    ok = True
    for entry in sorted(entries, key=lambda e: e.path):
        file_path = root / Path(*PurePosixPath(entry.path).parts)
        result: dict[str, object] = {
            "path": entry.path,
            "expected_size": entry.size,
            "expected_sha256": entry.sha256,
        }
        if not file_path.exists():
            result.update({"ok": False, "status": "missing"})
            ok = False
            results.append(result)
            continue
        if not file_path.is_file():
            result.update({"ok": False, "status": "not_file"})
            ok = False
            results.append(result)
            continue

        actual_size = file_path.stat().st_size
        result["actual_size"] = actual_size
        if actual_size != entry.size:
            result.update({"ok": False, "status": "size_mismatch"})
            ok = False
            results.append(result)
            continue

        actual_sha = sha256_file(file_path)
        result["actual_sha256"] = actual_sha
        if actual_sha != entry.sha256:
            result.update({"ok": False, "status": "sha256_mismatch"})
            ok = False
        else:
            result.update({"ok": True, "status": "ok"})
        results.append(result)
    return ok, results


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = Path(args.manifest).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"root directory not found: {root}")

    entries = parse_manifest(manifest)
    ok, results = verify_entries(entries, root)
    summary = {
        "ok": ok,
        "mode": "verify",
        "manifest": str(manifest),
        "root": str(root),
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_count": len(entries),
        "ok_count": sum(1 for row in results if row.get("ok") is True),
        "error_count": sum(1 for row in results if row.get("ok") is not True),
        "total_expected_bytes": sum(entry.size for entry in entries),
        "results": results,
    }

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("ok", "file_count", "ok_count", "error_count")}, indent=2))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or verify Qwen3.5-9B release checksums.")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate checksums.sha256 from files and directories.")
    generate.add_argument("--paths", nargs="+", required=True, help="Files or directories to include.")
    generate.add_argument("--out", required=True, help="Output manifest path.")
    generate.set_defaults(func=cmd_generate)

    verify = sub.add_parser("verify", help="Verify a checksum manifest against a root directory.")
    verify.add_argument("--manifest", required=True, help="Manifest generated by this script.")
    verify.add_argument("--root", required=True, help="Root directory for relative manifest paths.")
    verify.add_argument("--out", required=True, help="JSON summary output path.")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostic
        print(f"[qwen35-checksums] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
