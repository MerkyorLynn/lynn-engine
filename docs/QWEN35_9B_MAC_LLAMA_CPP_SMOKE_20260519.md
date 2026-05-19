# Qwen3.5-9B Mac llama.cpp Smoke Chain · 2026-05-19

This smoke chain validates the first-release Mac path for Qwen3.5-9B: Q4_K_M GGUF through `llama.cpp` with an OpenAI-compatible endpoint. It is a local executable check and does not touch CUDA kernels or Lynn Engine server code.

## Script

```bash
bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

Default behavior:

- Discovers the Q4_K_M GGUF.
- Discovers `llama-server`.
- Starts `llama-server` on `127.0.0.1:${PORT:-18197}`.
- Uses `--jinja --reasoning auto` by default.
- Runs `/health`, one chat completion, and one 256-token decode TPS smoke.
- Writes `reports/qwen35_9b/mac_smoke_<stamp>.json`.

## Discovery order

GGUF discovery:

1. `MODEL=/absolute/path/to/Qwen3.5-9B-Q4_K_M.gguf`
2. `./models`
3. `~/models`
4. `/Users/lynn/Downloads/Lynn/models`

`llama-server` discovery:

1. `LLAMA_SERVER=/absolute/path/to/llama-server`
2. `llama-server` in `PATH`
3. `/opt/homebrew/bin/llama-server`
4. `/usr/local/bin/llama-server`
5. `~/llama.cpp/build/bin/llama-server`
6. `~/llama.cpp/build/tools/server/llama-server`
7. `~/src/llama.cpp/build/bin/llama-server`
8. `~/dev/llama.cpp/build/bin/llama-server`

## Install llama.cpp on Mac

Homebrew:

```bash
brew install llama.cpp
```

Source build with Metal:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_METAL=ON
cmake --build ~/llama.cpp/build -j
export LLAMA_SERVER=~/llama.cpp/build/bin/llama-server
```

## Run examples

Explicit model path:

```bash
MODEL=/Users/lynn/Downloads/Lynn/models/Qwen3.5-9B-Q4_K_M.gguf \
  bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

Override port:

```bash
PORT=18198 bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

Force reasoning mode on:

```bash
REASONING=on bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

Dry-run discovery and print the server command:

```bash
bash scripts/local_qwen35_9b_llamacpp_smoke.sh --dry-run
```

## Failure behavior

The script fails loudly before loading the model when:

- The GGUF is missing or `MODEL` points to an empty/missing file.
- `llama-server` is missing or `LLAMA_SERVER` is not executable.
- The selected port is already in use.

Port fix command shown by the script:

```bash
lsof -nP -iTCP:18197 -sTCP:LISTEN
kill <pid>
```

Alternative port:

```bash
PORT=18198 bash scripts/local_qwen35_9b_llamacpp_smoke.sh
```

## Report schema

The JSON report contains:

```json
{
  "schema": "lynn-qwen35-9b-mac-llamacpp-smoke-v1",
  "ok": true,
  "base_url": "http://127.0.0.1:18197/v1",
  "model": "qwen35-9b-q4km",
  "model_path": "...Qwen3.5-9B-Q4_K_M.gguf",
  "llama_server": ".../llama-server",
  "server_log": "reports/qwen35_9b/mac_smoke_<stamp>.server.log",
  "checks": {
    "health": {"ok": true, "http_status": 200},
    "chat": {"ok": true, "elapsed_seconds": 0.0, "completion_tokens": 0, "decode_tps": 0.0},
    "decode_tps_256": {"ok": true, "elapsed_seconds": 0.0, "completion_tokens": 0, "decode_tps": 0.0}
  },
  "errors": []
}
```

The 256-token TPS value is a smoke metric, not a release benchmark. It is meant to prove the Mac endpoint can decode through the OpenAI-compatible API.
