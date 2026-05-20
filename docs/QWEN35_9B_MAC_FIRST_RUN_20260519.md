# Qwen3.5-9B Mac First-Run Wrapper · 2026-05-19

This is the lightweight first-run wrapper for the Mac Q4_K_M-imatrix GGUF + `llama.cpp` path. For the product setup path that downloads, registers the Lynn provider, and smokes the endpoint, use `scripts/local_qwen35_9b_setup.sh`.

Recommended product setup:

```bash
bash scripts/local_qwen35_9b_setup.sh --download --smoke
source ~/Models/Lynn/Qwen3.5-9B/lynn-qwen35-9b-q4km.env
bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

## Quick start

If you already have the GGUF:

```bash
MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M-imatrix.gguf \
bash scripts/local_qwen35_9b_first_run.sh
```

If the GGUF is in a standard location, the wrapper can discover it:

```bash
bash scripts/local_qwen35_9b_first_run.sh
```

Discovery paths:

1. `MODEL` or `--model`
2. `./models`
3. `~/models`
4. `/Users/lynn/Downloads/Lynn/models`

## Options

```bash
bash scripts/local_qwen35_9b_first_run.sh \
  --model /absolute/path/to/Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  --llama-server /absolute/path/to/llama-server \
  --port 18197 \
  --reasoning auto
```

Supported pass-through flags:

- `--model`
- `--llama-server`
- `--port`
- `--reasoning auto|on`
- `--dry-run`

`--dry-run` does not require a real model or `llama-server`; it prints the smoke command that would run.

## Missing model behavior

If no Q4_K_M GGUF is found, the wrapper exits with code `4` and points the user to the product setup path:

```bash
bash scripts/local_qwen35_9b_setup.sh --download --smoke
```

Manual fallback commands:

```bash
huggingface-cli download Merkyor/Qwen3.5-9B-GGUF-imatrix Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  --local-dir "$HOME/models" --local-dir-use-symlinks False

modelscope download --model Merkyor/Qwen3.5-9B-GGUF-imatrix Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  --local_dir "$HOME/models"

curl -L --fail --continue-at - --create-dirs \
  --output "$HOME/models/Qwen3.5-9B-Q4_K_M-imatrix.gguf" \
  https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf
```

The user can run the selected download command manually, or let the setup script do it.

## Missing smoke script behavior

The wrapper requires:

```text
scripts/local_qwen35_9b_llamacpp_smoke.sh
```

If it is missing or not executable, first-run fails loudly before looking for the model.

## Smoke output

The delegated smoke script starts `llama-server`, runs `/health`, one chat completion, one OpenAI `tools` call, and a 256-token decode TPS smoke. It writes:

```text
reports/qwen35_9b/mac_smoke_<stamp>.json
```

The default endpoint port is `18197`, overridable by `PORT` or `--port`.
