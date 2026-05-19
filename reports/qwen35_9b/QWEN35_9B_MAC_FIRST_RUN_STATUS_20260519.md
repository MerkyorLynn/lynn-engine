# Qwen3.5-9B Mac First-Run Status · 2026-05-19

## Scope

Adds a thin first-run wrapper for the Mac Q4_K_M GGUF + `llama.cpp` path. The wrapper converts the docs + smoke chain into one command while preserving the no-large-download guarantee.

## Added files

- `scripts/local_qwen35_9b_first_run.sh`
- `docs/QWEN35_9B_MAC_FIRST_RUN_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_MAC_FIRST_RUN_STATUS_20260519.md`

## Dependency

Requires the existing smoke script:

```text
scripts/local_qwen35_9b_llamacpp_smoke.sh
```

If that script is missing or not executable, first-run exits with code `3` and prints a repair command.

## Current validation

| Check | Status | Notes |
|---|---|---|
| `bash -n scripts/local_qwen35_9b_first_run.sh` | PASS | Syntax validated locally. |
| `scripts/local_qwen35_9b_first_run.sh --dry-run` | PASS | Does not require real GGUF or `llama-server`; prints delegated smoke command. |
| Missing-model path | IMPLEMENTED | Exits `4` and prints HF / ModelScope / `dl.merkyorlynn.com` placeholders. |
| Forbidden directories | PASS | No intended changes under `engine/`, `csrc/`, `server/`, or `benchmarks/`. |
| Real Mac first-run smoke | PENDING | Requires local Q4_K_M GGUF and `llama-server`. |

## Runtime command

```bash
MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf \
LLAMA_SERVER=/absolute/path/to/llama-server \
PORT=18197 \
bash scripts/local_qwen35_9b_first_run.sh
```

Expected delegated report:

```text
reports/qwen35_9b/mac_smoke_<stamp>.json
```

## No-download guarantee

The wrapper only prints download placeholders. It does not run `huggingface-cli`, `modelscope`, `curl`, or any other large-file download command.
