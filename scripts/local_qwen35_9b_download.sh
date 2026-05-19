#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-$HOME/Models/Lynn/Qwen3.5-9B}"
SOURCE="hf"
ARTIFACT="all"
DRY_RUN=0

Q4KM_FILE="${Q4KM_FILE:-Qwen3.5-9B-Q4_K_M.gguf}"
DL_BASE_URL="${DL_BASE_URL:-https://dl.merkyorlynn.com/models/qwen35-9b}"
HF_REPO_Q4KM="${HF_REPO_Q4KM:-TODO_HF_Q4KM_REPO}"
HF_REPO_NVFP4="${HF_REPO_NVFP4:-TODO_HF_NVFP4_W4A16_REPO}"
MS_REPO_Q4KM="${MS_REPO_Q4KM:-TODO_MODELSCOPE_Q4KM_REPO}"
MS_REPO_NVFP4="${MS_REPO_NVFP4:-TODO_MODELSCOPE_NVFP4_W4A16_REPO}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/local_qwen35_9b_download.sh --source hf|ms|dl --artifact q4km|nvfp4|all [--dry-run]

Purpose:
  Emit first-release download commands and checksum scaffolds for Qwen3.5-9B.
  This script does not download large model files. Review and run the printed
  commands manually after replacing TODO source IDs and checksum values.

Options:
  --source hf|ms|dl        Download source scaffold: Hugging Face, ModelScope, or Lynn CDN.
  --artifact q4km|nvfp4|all Artifact scaffold to emit.
  --dry-run                Print commands only; do not create directories or scaffold files.
  --model-root PATH        Target root (default: ~/Models/Lynn/Qwen3.5-9B).
  -h, --help               Show this help.

Environment overrides:
  MODEL_ROOT               Same as --model-root.
  Q4KM_FILE                GGUF file name.
  DL_BASE_URL              Lynn CDN base URL.
  HF_REPO_Q4KM             Hugging Face Q4_K_M repo ID.
  HF_REPO_NVFP4            Hugging Face NVFP4 W4A16 repo ID.
  MS_REPO_Q4KM             ModelScope Q4_K_M repo ID.
  MS_REPO_NVFP4            ModelScope NVFP4 W4A16 repo ID.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --artifact)
      ARTIFACT="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --model-root)
      MODEL_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[qwen35-9b-download] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$SOURCE" in
  hf|ms|dl) ;;
  *)
    echo "[qwen35-9b-download] --source must be one of: hf, ms, dl" >&2
    exit 2
    ;;
esac

case "$ARTIFACT" in
  q4km|nvfp4|all) ;;
  *)
    echo "[qwen35-9b-download] --artifact must be one of: q4km, nvfp4, all" >&2
    exit 2
    ;;
esac

q4km_dir="$MODEL_ROOT/q4_k_m"
nvfp4_dir="$MODEL_ROOT/nvfp4-w4a16"
checksum_file="$MODEL_ROOT/checksums.sha256.TODO"
commands_file="$MODEL_ROOT/download_${SOURCE}_${ARTIFACT}.commands.txt"

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

emit_q4km_command() {
  case "$SOURCE" in
    hf)
      print_command huggingface-cli download "$HF_REPO_Q4KM" "$Q4KM_FILE" --local-dir "$q4km_dir" --local-dir-use-symlinks False
      ;;
    ms)
      print_command modelscope download --model "$MS_REPO_Q4KM" "$Q4KM_FILE" --local_dir "$q4km_dir"
      ;;
    dl)
      print_command curl -L --fail --continue-at - --create-dirs --output "$q4km_dir/$Q4KM_FILE" "$DL_BASE_URL/q4_k_m/$Q4KM_FILE"
      ;;
  esac
}

emit_nvfp4_command() {
  case "$SOURCE" in
    hf)
      print_command huggingface-cli download "$HF_REPO_NVFP4" --local-dir "$nvfp4_dir" --local-dir-use-symlinks False
      ;;
    ms)
      print_command modelscope download --model "$MS_REPO_NVFP4" --local_dir "$nvfp4_dir"
      ;;
    dl)
      print_command curl -L --fail --continue-at - --create-dirs --output "$nvfp4_dir/qwen35-9b-nvfp4-w4a16.tar.zst" "$DL_BASE_URL/nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.tar.zst"
      print_command tar --zstd -xf "$nvfp4_dir/qwen35-9b-nvfp4-w4a16.tar.zst" -C "$nvfp4_dir"
      ;;
  esac
}

emit_checksum_scaffold() {
  cat <<EOF

# Checksum scaffold: replace TODO_SHA256_* before release.
# Suggested file: $checksum_file
TODO_SHA256_Q4KM  q4_k_m/$Q4KM_FILE
TODO_SHA256_NVFP4_MANIFEST  nvfp4-w4a16/lynn_quant_manifest.json
TODO_SHA256_NVFP4_INDEX  nvfp4-w4a16/model.safetensors.index.json
TODO_SHA256_NVFP4_TOKENIZER  nvfp4-w4a16/tokenizer.json

# Verify after final checksums are published:
#   cd "$MODEL_ROOT" && shasum -a 256 -c checksums.sha256
EOF
}

emit_plan() {
  echo "[qwen35-9b-download] source=$SOURCE artifact=$ARTIFACT"
  echo "[qwen35-9b-download] model_root=$MODEL_ROOT"
  echo "[qwen35-9b-download] large downloads are not executed by this script"
  echo ""

  if [[ "$ARTIFACT" == "q4km" || "$ARTIFACT" == "all" ]]; then
    echo "# Q4_K_M GGUF command"
    emit_q4km_command
    echo ""
  fi

  if [[ "$ARTIFACT" == "nvfp4" || "$ARTIFACT" == "all" ]]; then
    echo "# Lynn-native NVFP4 W4A16 command"
    emit_nvfp4_command
    echo ""
  fi

  emit_checksum_scaffold
}

if [[ "$DRY_RUN" == "1" ]]; then
  emit_plan
  exit 0
fi

mkdir -p "$q4km_dir" "$nvfp4_dir"
emit_plan > "$commands_file"

cat > "$checksum_file" <<EOF
TODO_SHA256_Q4KM  q4_k_m/$Q4KM_FILE
TODO_SHA256_NVFP4_MANIFEST  nvfp4-w4a16/lynn_quant_manifest.json
TODO_SHA256_NVFP4_INDEX  nvfp4-w4a16/model.safetensors.index.json
TODO_SHA256_NVFP4_TOKENIZER  nvfp4-w4a16/tokenizer.json
EOF

emit_plan
cat <<EOF

[qwen35-9b-download] wrote command scaffold: $commands_file
[qwen35-9b-download] wrote checksum scaffold: $checksum_file
[qwen35-9b-download] no large files were downloaded
EOF
