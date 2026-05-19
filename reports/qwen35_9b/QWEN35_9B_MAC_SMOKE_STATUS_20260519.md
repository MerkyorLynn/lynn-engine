# Qwen3.5-9B Mac Smoke Status · 2026-05-19

## Scope

Mac first-release smoke chain for Qwen3.5-9B Q4_K_M GGUF through `llama.cpp`.

## Added files

- `scripts/local_qwen35_9b_llamacpp_smoke.sh`
- `docs/QWEN35_9B_MAC_LLAMA_CPP_SMOKE_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_MAC_SMOKE_STATUS_20260519.md`

## Current validation

| Check | Status | Notes |
|---|---|---|
| `bash -n scripts/local_qwen35_9b_llamacpp_smoke.sh` | PASS | Syntax validated locally. |
| Forbidden directories | PASS | No changes intended under `engine/`, `csrc/`, `server/`, or `benchmarks/`. |
| Real Mac model smoke | PENDING | Requires local Q4_K_M GGUF and `llama-server`. |

## Runtime smoke command

```bash
MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf \
LLAMA_SERVER=/absolute/path/to/llama-server \
PORT=18197 \
bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

Expected output artifact:

```text
reports/qwen35_9b/mac_smoke_<stamp>.json
```

## Pass criteria

- GGUF discovery succeeds from `MODEL`, `./models`, `~/models`, or `/Users/lynn/Downloads/Lynn/models`.
- `llama-server` discovery succeeds from `LLAMA_SERVER`, `PATH`, or common `llama.cpp` build paths.
- Port is free before model load.
- `/health` returns HTTP 200.
- One chat completion returns non-empty text.
- One 256-token decode TPS smoke returns non-empty text and writes JSON.

## Fail-loud diagnostics

The script exits with clear remediation for:

- missing GGUF: set `MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf`;
- missing binary: install `llama.cpp` or set `LLAMA_SERVER=/absolute/path/to/llama-server`;
- occupied port: run `lsof -nP -iTCP:<port> -sTCP:LISTEN`, kill the stale process, or set another `PORT`.
