#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Qwen MTP Public Head Inventory — shell wrapper
#
# Scans a Qwen model directory and lists all MTP tensor keys, shapes, dtypes,
# and SHA256 prefixes. CPU-only, no GPU required.
#
# Usage:
#   bash scripts/qwen_mtp_public_head_inventory.sh --model-dir /path/to/model
#   MODEL_DIR=/path/to/model bash scripts/qwen_mtp_public_head_inventory.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${MODEL_DIR:-}"
OUT="${OUT:-}"
SHA_PREFIX_LEN="${SHA_PREFIX_LEN:-16}"

# Parse CLI args (pass-through to Python)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; ARGS+=(--model-dir "$2"); shift 2 ;;
    --out)       OUT="$2"; ARGS+=(--out "$2"); shift 2 ;;
    --sha256-prefix-len) SHA_PREFIX_LEN="$2"; ARGS+=(--sha256-prefix-len "$2"); shift 2 ;;
    --help|-h)
      echo "Usage: $0 --model-dir /path/to/model [--out report.json]"
      echo ""
      echo "Scans safetensors for MTP tensor keys. CPU-only, no GPU."
      echo ""
      echo "Options:"
      echo "  --model-dir PATH    Model directory (required)"
      echo "  --out PATH          Output JSON (default: stdout)"
      echo "  --sha256-prefix-len N  SHA256 hex prefix length (default: 16)"
      exit 0
      ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "[mtp-inventory] ERROR: --model-dir is required" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[mtp-inventory] ERROR: directory not found: $MODEL_DIR" >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/qwen_mtp_public_head_inventory.py" "${ARGS[@]}"
