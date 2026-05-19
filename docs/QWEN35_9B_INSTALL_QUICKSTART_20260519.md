# Qwen3.5-9B Local Install Quickstart · 2026-05-19

This is the first-release local deployment path for Qwen3.5-9B. It intentionally covers install, artifact download scaffolding, and endpoint smoke tests only. It does not change CUDA kernels.

## Release paths

| Platform | Runtime | Artifact | Status |
|---|---|---|---|
| macOS Apple Silicon | llama.cpp | Q4_K_M GGUF | first-release path |
| NVIDIA Linux Blackwell | Lynn Engine | Lynn-native NVFP4 W4A16 | first-release path |
| Windows | WSL2 or Docker | same as NVIDIA Linux | beta; no native Windows promise |

Default model root:

```bash
export MODEL_ROOT="$HOME/Models/Lynn/Qwen3.5-9B"
```

Generate download commands and checksum TODOs without downloading large files:

```bash
bash scripts/local_qwen35_9b_download.sh --source dl --artifact all --dry-run
```

Available sources:

```bash
bash scripts/local_qwen35_9b_download.sh --source hf --artifact all --dry-run
bash scripts/local_qwen35_9b_download.sh --source ms --artifact all --dry-run
bash scripts/local_qwen35_9b_download.sh --source dl --artifact all --dry-run
```

Checksum status: public SHA256 values are TODO. Until the final `checksums.sha256` is published, treat all checksum lines emitted by the script as placeholders to be replaced before release.

### Verify checksums after download

After downloading Q4_K_M from Hugging Face, ModelScope, or `dl.merkyorlynn.com`, place the released `checksums.sha256` under the model root and verify before running smoke tests:

```bash
export MODEL_ROOT="$HOME/Models/Lynn/Qwen3.5-9B"
bash scripts/local_qwen35_9b_verify_checksums.sh \
  --root "$MODEL_ROOT" \
  --manifest "$MODEL_ROOT/checksums.sha256" \
  --out reports/qwen35_9b/local_checksum_verify.json
```

For release packaging, generate a manifest from local Q4_K_M and NVFP4 artifacts without downloading anything:

```bash
python3 scripts/qwen35_9b_release_checksums.py generate \
  --paths "$MODEL_ROOT/q4_k_m" "$MODEL_ROOT/nvfp4-w4a16" \
  --out "$MODEL_ROOT/checksums.sha256"
```

Verification fails nonzero if any file is missing, has a size mismatch, or has a SHA256 mismatch.

## macOS: llama.cpp + Q4_K_M GGUF

### 1. Install llama.cpp

Homebrew path:

```bash
brew install llama.cpp
```

Source build path:

```bash
git clone https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DGGML_METAL=ON
cmake --build "$HOME/llama.cpp/build" -j
export LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
```

### 2. Prepare Q4_K_M artifact

Print the exact download command for the selected source:

```bash
bash scripts/local_qwen35_9b_download.sh --source dl --artifact q4km --dry-run
```

After replacing any TODO source placeholders and downloading manually, expected layout:

```text
~/Models/Lynn/Qwen3.5-9B/
├── q4_k_m/
│   └── Qwen3.5-9B-Q4_K_M.gguf
└── checksums.sha256
```

### 3. Start OpenAI-compatible endpoint

```bash
export GGUF="$MODEL_ROOT/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf"
bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh \
  --host 127.0.0.1 \
  --port 18099 \
  --model-name qwen35-9b-q4km
```

### 4. Smoke test

In another shell:

```bash
curl -fsS http://127.0.0.1:18099/v1/models

curl -fsS http://127.0.0.1:18099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-9b-q4km","messages":[{"role":"user","content":"Say OK in one short sentence."}],"temperature":0,"max_tokens":32}'

bash scripts/local_qwen35_9b_q4km_smoke.sh --base-url http://127.0.0.1:18099/v1 --model qwen35-9b-q4km
```

Passing smoke criteria:

- `/v1/models` returns HTTP 200.
- `/v1/chat/completions` returns non-empty text.
- `local_qwen35_9b_q4km_smoke.sh` passes short-answer and JSON-object checks.

## NVIDIA Linux: Lynn Engine + NVFP4 W4A16

Target environment:

- Linux host with NVIDIA Blackwell GPU.
- CUDA/PyTorch environment already installed for Lynn Engine.
- Lynn Engine repo checked out locally.
- Qwen3.5-9B Lynn-native NVFP4 W4A16 artifact downloaded to the model root.

### 1. Prepare NVFP4 artifact

Print the download command scaffold:

```bash
bash scripts/local_qwen35_9b_download.sh --source dl --artifact nvfp4 --dry-run
```

Expected layout after manual download:

```text
~/Models/Lynn/Qwen3.5-9B/
├── nvfp4-w4a16/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-*.safetensors
│   └── lynn_quant_manifest.json
└── checksums.sha256
```

### 2. Start Lynn OpenAI-compatible endpoint

```bash
cd /path/to/lynn-engine
export MODEL_DIR="$HOME/Models/Lynn/Qwen3.5-9B/nvfp4-w4a16"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m server.openai_http \
  --model "$MODEL_DIR" \
  --served-name qwen35-9b-nvfp4-w4a16 \
  --host 127.0.0.1 \
  --port 18191 \
  --dtype bfloat16
```

### 3. Smoke test

In another shell:

```bash
curl -fsS http://127.0.0.1:18191/health
curl -fsS http://127.0.0.1:18191/v1/models

curl -fsS http://127.0.0.1:18191/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35-9b-nvfp4-w4a16","messages":[{"role":"user","content":"Say OK in one short sentence."}],"temperature":0,"max_tokens":32}'
```

Optional R6000-style TPS smoke:

```bash
python benchmarks/p25_server_decode_tps_probe.py \
  --url http://127.0.0.1:18191/v1 \
  --model qwen35-9b-nvfp4-w4a16 \
  --chat \
  --max-tokens 128 256 512 \
  --runs 3 \
  --out reports/qwen35_9b/local_nvfp4_w4a16_p25_smoke.json
```

Passing smoke criteria:

- `/health` and `/v1/models` return HTTP 200.
- Chat completion returns non-empty text.
- Optional P25 smoke writes a JSON report without runtime errors.

## Windows beta

Windows is supported only through Linux-compatible environments for the first release:

1. WSL2 Ubuntu with NVIDIA GPU access, then follow the NVIDIA Linux path inside WSL2.
2. Docker with NVIDIA Container Toolkit, mounting `~/Models/Lynn/Qwen3.5-9B` into the container.

Native Windows binaries are not promised for the first Qwen3.5-9B release.

## Artifact and checksum TODOs

- TODO: publish final Hugging Face repo IDs for Q4_K_M and NVFP4 W4A16.
- TODO: publish final ModelScope repo IDs for Q4_K_M and NVFP4 W4A16.
- TODO: publish final Lynn download URLs under `dl.merkyorlynn.com`.
- TODO: publish `checksums.sha256` with final SHA256 values for GGUF, NVFP4 shards, manifests, and tokenizer files.
