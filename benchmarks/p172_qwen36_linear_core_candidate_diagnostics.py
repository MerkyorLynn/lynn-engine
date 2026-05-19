#!/usr/bin/env python3
"""P172: read-only diagnostics for P169 linear-core fixtures.

This helper intentionally avoids importing torch, engine, or CUDA-facing code.
It reads safetensors metadata/data bytes directly, builds a per-tensor
shape/dtype/hash manifest, and performs a structural candidate-output-dir
preflight for future fused linear-core kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "lynn-qwen36-linear-core-p172-candidate-diagnostics-v1"
P169_FIXTURE_SCHEMA = "lynn-qwen36-linear-core-fixture-v1"

REQUIRED_FIXTURE_KEYS = [
    "h_norm",
    "conv_state_in",
    "recurrent_state_in",
    "linear_core_out",
    "conv_state_out",
    "recurrent_state_out",
]
OPTIONAL_ABI_KEYS = [
    "z",
    "core_attn",
    "core_attn_out",
]
REQUIRED_CANDIDATE_KEYS = [
    "linear_core_out",
    "conv_state_out",
    "recurrent_state_out",
]


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: list[int]
    data_offsets: list[int]
    nbytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_range(path: Path, start: int, end: int) -> str:
    h = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as f:
        f.seek(start)
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"short read while hashing {path}:{start}-{end}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"not a safetensors file or too small: {path}")
        (header_len,) = struct.unpack("<Q", raw_len)
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header JSON in {path}: {exc}") from exc
    return 8 + header_len, header


def _tensor_manifest(path: Path) -> dict[str, TensorInfo]:
    data_start, header = _read_safetensors_header(path)
    tensors: dict[str, TensorInfo] = {}
    for name, meta in sorted(header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            raise ValueError(f"malformed safetensors tensor metadata for {name!r} in {path}")
        offsets = [int(x) for x in meta["data_offsets"]]
        if len(offsets) != 2 or offsets[0] > offsets[1]:
            raise ValueError(f"bad data_offsets for {name!r} in {path}: {offsets}")
        start = data_start + offsets[0]
        end = data_start + offsets[1]
        tensors[name] = TensorInfo(
            dtype=str(meta.get("dtype", "")),
            shape=[int(x) for x in meta.get("shape", [])],
            data_offsets=offsets,
            nbytes=offsets[1] - offsets[0],
            sha256=_sha256_range(path, start, end),
        )
    return tensors


def _load_fixture_manifest(fixtures_dir: Path) -> dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"P169 fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version") or manifest.get("schema")
    if schema != P169_FIXTURE_SCHEMA:
        raise ValueError(f"unexpected P169 fixture schema in {manifest_path}: {schema!r}")
    if not isinstance(manifest.get("fixtures"), list):
        raise ValueError(f"P169 fixture manifest has no fixture list: {manifest_path}")
    return manifest


def _candidate_path(candidate_dir: Path, fixture_file: str) -> Path | None:
    candidates = [
        candidate_dir / fixture_file,
        candidate_dir / Path(fixture_file).name,
        candidate_dir / f"{Path(fixture_file).stem}.safetensors",
    ]
    return next((path for path in candidates if path.exists()), None)


def _shape_dtype_match(ref: TensorInfo, got: TensorInfo) -> bool:
    return ref.shape == got.shape and ref.dtype == got.dtype


def _fixture_row(fixtures_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    fixture_file = str(item["file"])
    fixture_path = fixtures_dir / fixture_file
    tensors = _tensor_manifest(fixture_path)
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in tensors]
    optional_present = [key for key in OPTIONAL_ABI_KEYS if key in tensors]
    return {
        "fixture_file": fixture_file,
        "path": str(fixture_path),
        "file_sha256": _sha256_file(fixture_path),
        "layer_id": int(item["layer_id"]) if "layer_id" in item else None,
        "prompt_id": int(item["prompt_id"]) if "prompt_id" in item else None,
        "passed_fixture_abi": not missing,
        "missing_required_fixture_keys": missing,
        "optional_abi_keys_present": optional_present,
        "tensor_count": len(tensors),
        "tensors": {key: value.to_dict() for key, value in tensors.items()},
        "_tensor_info": tensors,
    }


def _candidate_row(
    candidate_dir: Path,
    fixture: dict[str, Any],
    required_candidate_keys: list[str],
    require_hash_match: bool,
) -> dict[str, Any]:
    fixture_file = str(fixture["fixture_file"])
    ref_tensors: dict[str, TensorInfo] = fixture["_tensor_info"]
    found = _candidate_path(candidate_dir, fixture_file)
    if found is None:
        return {
            "fixture_file": fixture_file,
            "candidate_file": None,
            "passed_candidate_preflight": False,
            "fail_reasons": ["candidate file missing"],
            "missing_required_candidate_keys": required_candidate_keys,
            "tensor_count": 0,
            "tensors": {},
            "comparisons": {},
        }

    tensors = _tensor_manifest(found)
    missing = [key for key in required_candidate_keys if key not in tensors]
    comparisons: dict[str, Any] = {}
    fail_reasons = [f"missing candidate tensor {key}" for key in missing]

    for key in required_candidate_keys:
        if key not in tensors or key not in ref_tensors:
            continue
        shape_dtype_ok = _shape_dtype_match(ref_tensors[key], tensors[key])
        hash_equal = ref_tensors[key].sha256 == tensors[key].sha256
        comparisons[key] = {
            "shape_dtype_match": shape_dtype_ok,
            "hash_equal_to_fixture": hash_equal,
            "fixture_sha256": ref_tensors[key].sha256,
            "candidate_sha256": tensors[key].sha256,
        }
        if not shape_dtype_ok:
            fail_reasons.append(f"{key} shape/dtype mismatch")
        if require_hash_match and not hash_equal:
            fail_reasons.append(f"{key} hash mismatch")

    return {
        "fixture_file": fixture_file,
        "candidate_file": str(found),
        "candidate_file_sha256": _sha256_file(found),
        "passed_candidate_preflight": not fail_reasons,
        "fail_reasons": fail_reasons,
        "missing_required_candidate_keys": missing,
        "tensor_count": len(tensors),
        "tensors": {key: value.to_dict() for key, value in tensors.items()},
        "comparisons": comparisons,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_dir = Path(args.fixtures)
    manifest = _load_fixture_manifest(fixtures_dir)
    required_candidate_keys = [
        part.strip()
        for part in args.required_candidate_keys.split(",")
        if part.strip()
    ]
    if not required_candidate_keys:
        raise ValueError("--required-candidate-keys resolved to an empty set")

    fixture_rows = [_fixture_row(fixtures_dir, item) for item in manifest["fixtures"]]
    candidate_dir = Path(args.candidate_output_dir) if args.candidate_output_dir else None
    candidate_rows = (
        [
            _candidate_row(candidate_dir, row, required_candidate_keys, args.require_hash_match)
            for row in fixture_rows
        ]
        if candidate_dir is not None
        else []
    )

    fixture_passed = sum(1 for row in fixture_rows if row["passed_fixture_abi"])
    candidate_passed = sum(1 for row in candidate_rows if row["passed_candidate_preflight"])
    optional_counts = {
        key: sum(1 for row in fixture_rows if key in row["optional_abi_keys_present"])
        for key in OPTIONAL_ABI_KEYS
    }
    exact_hash_matches = {
        key: sum(
            1
            for row in candidate_rows
            if row.get("comparisons", {}).get(key, {}).get("hash_equal_to_fixture") is True
        )
        for key in required_candidate_keys
    }

    public_fixture_rows = []
    for row in fixture_rows:
        clean = dict(row)
        clean.pop("_tensor_info", None)
        public_fixture_rows.append(clean)

    return {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixtures": str(fixtures_dir),
        "fixture_manifest": str(fixtures_dir / "manifest.json"),
        "candidate_output_dir": str(candidate_dir) if candidate_dir is not None else None,
        "required_fixture_keys": REQUIRED_FIXTURE_KEYS,
        "optional_abi_keys": OPTIONAL_ABI_KEYS,
        "required_candidate_keys": required_candidate_keys,
        "require_hash_match": bool(args.require_hash_match),
        "summary": {
            "total_fixtures": len(fixture_rows),
            "fixture_abi_passed": fixture_passed,
            "fixture_abi_all_passed": fixture_passed == len(fixture_rows),
            "candidate_preflight_passed": candidate_passed if candidate_dir is not None else None,
            "candidate_preflight_all_passed": (
                candidate_passed == len(candidate_rows) if candidate_dir is not None else None
            ),
            "optional_abi_key_fixture_counts": optional_counts,
            "candidate_exact_hash_matches": exact_hash_matches if candidate_dir is not None else None,
        },
        "fixtures_manifest": public_fixture_rows,
        "candidate_preflight": candidate_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a read-only P172 manifest and candidate-output-dir preflight for P169 linear-core fixtures."
    )
    parser.add_argument("--fixtures", required=True, help="P169 fixture directory containing manifest.json and safetensors.")
    parser.add_argument("--out", required=True, help="JSON report path.")
    parser.add_argument("--candidate-output-dir", default="", help="Optional candidate safetensors directory to preflight.")
    parser.add_argument(
        "--required-candidate-keys",
        default=",".join(REQUIRED_CANDIDATE_KEYS),
        help="Comma-separated candidate tensors required for structural admission.",
    )
    parser.add_argument(
        "--require-hash-match",
        action="store_true",
        help="Also fail candidate preflight unless required candidate tensors exactly match fixture bytes.",
    )
    args = parser.parse_args()

    report = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
