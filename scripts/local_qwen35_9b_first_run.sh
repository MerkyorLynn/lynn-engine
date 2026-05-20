#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_SCRIPT="$ROOT/scripts/local_qwen35_9b_llamacpp_smoke.sh"
MODEL_PATH="${MODEL:-}"
LLAMA_SERVER_PATH="${LLAMA_SERVER:-}"
PORT_VALUE="${PORT:-18197}"
REASONING_VALUE="${REASONING:-auto}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local_qwen35_9b_first_run.sh [options]

Small first-run wrapper for the Mac Qwen3.5-9B Q4_K_M-imatrix llama.cpp path.
For a true one-command install, prefer:
  bash scripts/local_qwen35_9b_setup.sh --download --smoke --serve

Options:
  --model PATH          Q4_K_M GGUF path; same as MODEL env.
  --llama-server PATH   llama-server binary; same as LLAMA_SERVER env.
  --port PORT           Endpoint port; same as PORT env (default: 18197).
  --reasoning auto|on   llama.cpp reasoning mode; same as REASONING env (default: auto).
  --dry-run             Print the smoke command without requiring model/server.
  -h, --help            Show help.

Examples:
  MODEL=~/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf bash scripts/local_qwen35_9b_first_run.sh
  bash scripts/local_qwen35_9b_first_run.sh --model ~/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf --port 18198
  bash scripts/local_qwen35_9b_first_run.sh --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="${2:-}"
      shift 2
      ;;
    --llama-server)
      LLAMA_SERVER_PATH="${2:-}"
      shift 2
      ;;
    --port)
      PORT_VALUE="${2:-}"
      shift 2
      ;;
    --reasoning)
      REASONING_VALUE="${2:-}"
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
      echo "[first-run] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$REASONING_VALUE" in
  auto|on) ;;
  *)
    echo "[first-run] --reasoning must be auto or on" >&2
    exit 2
    ;;
esac

if [[ ! -f "$SMOKE_SCRIPT" ]]; then
  cat >&2 <<EOF
[first-run] ERROR: smoke script is missing:
  $SMOKE_SCRIPT

Fix:
  git checkout origin/main -- scripts/local_qwen35_9b_llamacpp_smoke.sh
  # or merge the Mac llama.cpp smoke-chain branch before running first-run.
EOF
  exit 3
fi

if [[ ! -x "$SMOKE_SCRIPT" ]]; then
  cat >&2 <<EOF
[first-run] ERROR: smoke script is not executable:
  $SMOKE_SCRIPT

Fix:
  chmod +x scripts/local_qwen35_9b_llamacpp_smoke.sh
EOF
  exit 3
fi

print_download_placeholders() {
  cat <<EOF
[first-run] ERROR: Qwen3.5-9B Q4_K_M-imatrix GGUF is not available.

Use the product setup command to download, configure, register, and smoke:

  bash scripts/local_qwen35_9b_setup.sh --download --smoke

Search paths checked:
  - MODEL / --model: ${MODEL_PATH:-<unset>}
  - $ROOT/models
  - $HOME/models
  - /Users/lynn/Downloads/Lynn/models

Manual fallbacks:

  # Hugging Face / ModelScope mirror, when published:
  huggingface-cli download Merkyor/Qwen3.5-9B-GGUF-imatrix Qwen3.5-9B-Q4_K_M-imatrix.gguf \
    --local-dir "$HOME/models" --local-dir-use-symlinks False

  modelscope download --model Merkyor/Qwen3.5-9B-GGUF-imatrix Qwen3.5-9B-Q4_K_M-imatrix.gguf \
    --local_dir "$HOME/models"

  curl -L --fail --continue-at - --create-dirs \
    --output "$HOME/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf" \
    https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf

After download:
  MODEL="$HOME/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf" bash scripts/local_qwen35_9b_first_run.sh
EOF
}

find_existing_model() {
  if [[ -n "$MODEL_PATH" ]]; then
    if [[ -s "$MODEL_PATH" ]]; then
      printf '%s\n' "$MODEL_PATH"
      return 0
    fi
    return 1
  fi

  local roots=(
    "$ROOT/models"
    "$HOME/models"
    "/Users/lynn/Downloads/Lynn/models"
  )
  local root
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r candidate; do
      [[ -s "$candidate" ]] || continue
      printf '%s\n' "$candidate"
      return 0
    done < <(find "$root" -maxdepth 5 -type f \( \
      -iname '*Qwen3.5*9B*Q4*K*M*imatrix*.gguf' -o \
      -iname '*qwen3.5*9b*q4*k*m*imatrix*.gguf' -o \
      -iname '*Qwen3.5*9B*Q4_K_M*imatrix*.gguf' -o \
      -iname '*qwen3.5*9b*q4_k_m*imatrix*.gguf' -o \
      -iname '*Qwen3.5*9B*Q4*K*M*.gguf' -o \
      -iname '*qwen3.5*9b*q4*k*m*.gguf' -o \
      -iname '*Qwen3.5*9B*Q4_K_M*.gguf' -o \
      -iname '*qwen3.5*9b*q4_k_m*.gguf' \
    \) 2>/dev/null | sort)
  done
  return 1
}

resolved_model=""
if [[ "$DRY_RUN" != "1" ]]; then
  resolved_model="$(find_existing_model || true)"
  if [[ -z "$resolved_model" ]]; then
    print_download_placeholders >&2
    exit 4
  fi
else
  resolved_model="${MODEL_PATH:-/absolute/path/to/Qwen3.5-9B-Q4_K_M-imatrix.gguf}"
fi

SMOKE_CMD=(
  "$SMOKE_SCRIPT"
  --model "$resolved_model"
  --port "$PORT_VALUE"
  --reasoning "$REASONING_VALUE"
)

if [[ -n "$LLAMA_SERVER_PATH" ]]; then
  SMOKE_CMD+=(--llama-server "$LLAMA_SERVER_PATH")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  SMOKE_CMD+=(--dry-run)
  echo "[first-run] DRY_RUN=1 — no model or llama-server validation performed."
  echo "[first-run] Smoke command that would run:"
  printf '  '
  printf '%q ' "${SMOKE_CMD[@]}"
  printf '\n'
  exit 0
fi

cat <<EOF
[first-run] Starting Qwen3.5-9B Mac first-run smoke.
[first-run] model=$resolved_model
[first-run] port=$PORT_VALUE
[first-run] reasoning=$REASONING_VALUE
EOF

exec "${SMOKE_CMD[@]}"
