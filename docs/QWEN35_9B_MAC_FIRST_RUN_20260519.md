# Qwen3.5-9B Mac First-Run Wrapper · 2026-05-19

This is the one-command first-run wrapper for the Mac Q4_K_M GGUF + `llama.cpp` path. It delegates to `scripts/local_qwen35_9b_llamacpp_smoke.sh` and never downloads large model files automatically.

## Quick start

If you already have the GGUF:

```bash
MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf \
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
  --model /absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf \
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

If no Q4_K_M GGUF is found, the wrapper exits with code `4` and prints download placeholders for all three planned distribution paths:

```bash
# Hugging Face placeholder:
huggingface-cli download TODO_HF_QWEN35_9B_Q4KM_REPO Qwen3.5-9B-Q4_K_M.gguf \
  --local-dir "$HOME/models" --local-dir-use-symlinks False

# ModelScope placeholder:
modelscope download --model TODO_MODELSCOPE_QWEN35_9B_Q4KM_REPO Qwen3.5-9B-Q4_K_M.gguf \
  --local_dir "$HOME/models"

# Lynn CDN placeholder:
curl -L --fail --continue-at - --create-dirs \
  --output "$HOME/models/Qwen3.5-9B-Q4_K_M.gguf" \
  https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf
```

The user must run the selected download command manually after replacing TODO placeholders with final release IDs. The wrapper does not download the GGUF.

## Missing smoke script behavior

The wrapper requires:

```text
scripts/local_qwen35_9b_llamacpp_smoke.sh
```

If it is missing or not executable, first-run fails loudly before looking for the model.

## Smoke output

The delegated smoke script starts `llama-server`, runs `/health`, one chat completion, and a 256-token decode TPS smoke. It writes:

```text
reports/qwen35_9b/mac_smoke_<stamp>.json
```

The default endpoint port is `18197`, overridable by `PORT` or `--port`.
