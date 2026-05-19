#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/Models/Lynn/Qwen3.5-9B}"
MANIFEST="${MANIFEST:-$MODEL_ROOT/checksums.sha256}"
OUT="${OUT:-$ROOT/reports/qwen35_9b/checksum_verify_$(date +%Y%m%d_%H%M%S).json}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local_qwen35_9b_verify_checksums.sh [options]

Verifies a Qwen3.5-9B checksums.sha256 manifest. It never downloads models and
never reports PASS when the manifest or model files are absent.

Options:
  --root PATH          Model root (default: ~/Models/Lynn/Qwen3.5-9B).
  --manifest PATH      Manifest path (default: <root>/checksums.sha256).
  --out PATH           JSON summary path.
  --dry-run            Print planned verification and fail if required files are absent.
  -h, --help           Show help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      MODEL_ROOT="${2:-}"
      shift 2
      ;;
    --manifest)
      MANIFEST="${2:-}"
      shift 2
      ;;
    --out)
      OUT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[qwen35-verify] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CHECKSUM_TOOL="$ROOT/scripts/qwen35_9b_release_checksums.py"

if [[ ! -f "$CHECKSUM_TOOL" ]]; then
  echo "[qwen35-verify] ERROR: checksum tool missing: $CHECKSUM_TOOL" >&2
  exit 2
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
  cat >&2 <<EOF
[qwen35-verify] ERROR: model root is missing: $MODEL_ROOT

Expected after download:
  $MODEL_ROOT/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf
  $MODEL_ROOT/nvfp4-w4a16/...
  $MODEL_ROOT/checksums.sha256
EOF
  exit 4
fi

if [[ ! -f "$MANIFEST" ]]; then
  cat >&2 <<EOF
[qwen35-verify] ERROR: checksum manifest is missing: $MANIFEST

Generate or download checksums.sha256 first, then rerun:
  $PYTHON_BIN scripts/qwen35_9b_release_checksums.py generate \
    --paths "$MODEL_ROOT/q4_k_m" "$MODEL_ROOT/nvfp4-w4a16" \
    --out "$MODEL_ROOT/checksums.sha256"
EOF
  exit 4
fi

if [[ ! -s "$MANIFEST" ]]; then
  echo "[qwen35-verify] ERROR: checksum manifest is empty: $MANIFEST" >&2
  exit 4
fi

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON_BIN" - "$MANIFEST" "$MODEL_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

manifest = Path(sys.argv[1])
root = Path(sys.argv[2])
missing: list[str] = []
with manifest.open("r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) != 3:
            print(f"[qwen35-verify] ERROR: invalid manifest line: {line}", file=sys.stderr)
            raise SystemExit(2)
        rel = parts[2]
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or rel in {"", "."}:
            print(f"[qwen35-verify] ERROR: unsafe manifest path: {rel}", file=sys.stderr)
            raise SystemExit(2)
        if not (root / Path(*pure.parts)).is_file():
            missing.append(rel)
if missing:
    print("[qwen35-verify] ERROR: dry-run found missing manifest files:", file=sys.stderr)
    for rel in missing[:50]:
        print(f"  {rel}", file=sys.stderr)
    if len(missing) > 50:
        print(f"  ... {len(missing) - 50} more", file=sys.stderr)
    raise SystemExit(4)
PY
  cat <<EOF
[qwen35-verify] DRY_RUN=1
[qwen35-verify] Would run:
  $PYTHON_BIN "$CHECKSUM_TOOL" verify --manifest "$MANIFEST" --root "$MODEL_ROOT" --out "$OUT"
[qwen35-verify] Root, manifest, and manifest-listed files exist; sha256 not computed in dry-run.
EOF
  exit 0
fi

exec "$PYTHON_BIN" "$CHECKSUM_TOOL" verify \
  --manifest "$MANIFEST" \
  --root "$MODEL_ROOT" \
  --out "$OUT"
