#!/usr/bin/env bash
set -euo pipefail

# Local Mac/Linux launcher for the Qwen3.5-9B Q4_K_M GGUF route.
#
# This is the product-facing counterpart to Lynn Engine's NVIDIA/NVFP4 route:
# it starts a llama.cpp OpenAI-compatible endpoint that agent CLIs can use with
# the same base_url/model shape as the Lynn server.

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18099}"
SERVED_NAME="${SERVED_NAME:-qwen35-9b-q4km}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-}"
PARALLEL="${PARALLEL:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
LLAMA_SERVER="${LLAMA_SERVER:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
LLAMA_REASONING_ARGS="${LLAMA_REASONING_ARGS:---jinja --reasoning auto}"
GGUF="${GGUF:-}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/Models}"
LOG_DIR="${LOG_DIR:-$HOME/.lynn-engine/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/qwen35_9b_q4km_llamacpp_${PORT}.log}"

mkdir -p "$LOG_DIR"

find_llama_server() {
  if [[ -n "$LLAMA_SERVER" && -x "$LLAMA_SERVER" ]]; then
    printf '%s\n' "$LLAMA_SERVER"
    return 0
  fi
  for candidate in \
    "$(command -v llama-server 2>/dev/null || true)" \
    "$(command -v llama.cpp-server 2>/dev/null || true)" \
    "/opt/homebrew/bin/llama-server" \
    "/usr/local/bin/llama-server" \
    "$HOME/llama.cpp/build/bin/llama-server" \
    "$HOME/llama.cpp/build/tools/server/llama-server" \
    "$HOME/llama.cpp/build-cuda/bin/llama-server"; do
    if [[ -n "${candidate:-}" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

find_gguf() {
  if [[ -n "$GGUF" && -s "$GGUF" ]]; then
    printf '%s\n' "$GGUF"
    return 0
  fi

  local roots=(
    "$MODEL_ROOT"
    "$HOME/Models"
    "$HOME/Downloads"
    "$HOME/.cache/huggingface/hub"
    "$PWD"
  )
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r candidate; do
      if [[ -s "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done < <(find "$root" -maxdepth 5 -type f \( \
      -iname '*Qwen3.5*9B*Q4*K*M*.gguf' -o \
      -iname '*qwen3.5*9b*q4*k*m*.gguf' -o \
      -iname '*Qwen3.5*9B*Q4_K_M*imatrix*.gguf' -o \
      -iname '*qwen3.5*9b*q4_k_m*imatrix*.gguf' \
    \) 2>/dev/null | sort)
  done
  return 1
}

server_bin="$(find_llama_server || true)"
if [[ -z "$server_bin" ]]; then
  cat >&2 <<'EOF'
[qwen35-q4km-local] llama-server not found.

Install one of:
  brew install llama.cpp
  # or build llama.cpp:
  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_METAL=ON
  cmake --build ~/llama.cpp/build -j

Then rerun this script, or set LLAMA_SERVER=/path/to/llama-server.
EOF
  exit 3
fi

gguf_path="$(find_gguf || true)"
if [[ -z "$gguf_path" ]]; then
  cat >&2 <<EOF
[qwen35-q4km-local] Qwen3.5-9B Q4_K_M GGUF not found.

Expected search roots include:
  $MODEL_ROOT
  $HOME/Models
  $HOME/Downloads

Set GGUF=/absolute/path/to/model.gguf, or download a Q4_K_M / imatrix GGUF
from the Lynn/Merkyor mirror or HuggingFace/ModelScope, then rerun:

  GGUF=/path/to/Qwen3.5-9B-Q4_K_M.gguf bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
EOF
  exit 4
fi

if [[ -z "$THREADS" ]]; then
  THREADS="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)"
fi

cat <<EOF
[qwen35-q4km-local] server=$server_bin
[qwen35-q4km-local] model=$gguf_path
[qwen35-q4km-local] endpoint=http://$HOST:$PORT/v1
[qwen35-q4km-local] served_name=$SERVED_NAME
[qwen35-q4km-local] ctx=$CTX_SIZE threads=$THREADS parallel=$PARALLEL n_gpu_layers=$N_GPU_LAYERS
[qwen35-q4km-local] log=$LOG_FILE
EOF

exec "$server_bin" \
  --model "$gguf_path" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --threads "$THREADS" \
  --parallel "$PARALLEL" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  -a "$SERVED_NAME" \
  $LLAMA_REASONING_ARGS \
  $LLAMA_EXTRA_ARGS \
  2>&1 | tee "$LOG_FILE"
