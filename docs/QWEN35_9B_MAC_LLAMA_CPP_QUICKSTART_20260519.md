# Qwen3.5-9B Mac llama.cpp Quickstart

Date: 2026-05-19

## Recommended Setup (macOS Apple Silicon)

```bash
# 1. Install llama.cpp (one-time)
brew install llama.cpp

# 2. Download, configure, and run a transient smoke test
bash scripts/local_qwen35_9b_setup.sh --download --smoke

# 3. Start the persistent local endpoint
source ~/Models/Lynn/Qwen3.5-9B/lynn-qwen35-9b-q4km.env
bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

That's it. Your endpoint is live at `http://127.0.0.1:18099/v1`.

The setup script writes a reusable env file:

```bash
~/Models/Lynn/Qwen3.5-9B/lynn-qwen35-9b-q4km.env
```

It contains `GGUF`, `LLAMA_SERVER`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
the served model name.

## Connect Your Coding Agent

All agents use the same three values:

```
base_url = http://127.0.0.1:18099/v1
api_key  = local
model    = qwen35-9b-q4km
```

### Claude Code

```bash
export OPENAI_BASE_URL=http://127.0.0.1:18099/v1
export OPENAI_API_KEY=local
claude --model qwen35-9b-q4km
```

### Cline (VS Code)

1. Open Cline settings
2. API Provider: **OpenAI Compatible**
3. Base URL: `http://127.0.0.1:18099/v1`
4. API Key: `local`
5. Model ID: `qwen35-9b-q4km`

### Continue (VS Code / JetBrains)

Add to `~/.continue/config.yaml`:

```yaml
models:
  - title: Qwen3.5-9B Local
    provider: openai
    model: qwen35-9b-q4km
    apiBase: http://127.0.0.1:18099/v1
    apiKey: local
```

### OpenCode

Add to `~/.opencode/config.yaml`:

```yaml
providers:
  local-qwen:
    kind: openai
    apiKey: local
    baseURL: http://127.0.0.1:18099/v1
    models:
      qwen35-9b-q4km:
        maxTokens: 32768
        contextWindow: 32768
        supportsStreaming: true
```

### aider

```bash
aider --openai-api-base http://127.0.0.1:18099/v1 \
      --openai-api-key local \
      --model openai/qwen35-9b-q4km
```

## Verify Offline (No Model Needed)

To check setup decisions without downloading or starting anything:

```bash
bash scripts/local_qwen35_9b_setup.sh --dry-run
```

To check the server launcher resolves paths and would start correctly:

```bash
DRY_RUN=1 bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

This validates llama-server discovery + GGUF discovery without actually starting.

To check smoke test script in offline mode:

```bash
bash scripts/local_qwen35_9b_q4km_smoke.sh --dry-run
```

## Customization

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 18099 | Server port |
| `CTX_SIZE` | 32768 | Context window tokens |
| `PARALLEL` | 4 | Concurrent request slots |
| `N_GPU_LAYERS` | 999 | GPU offload (999=all) |
| `GGUF` | auto | Explicit GGUF path |
| `MODEL_ROOT` | ~/Models | Model search root |
| `LLAMA_SERVER` | auto | Explicit server binary path |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "llama-server not found" | `brew install llama.cpp` or build from source |
| "GGUF not found" | Set `GGUF=/path/to/model.gguf` or download to ~/Models |
| OOM on 8GB Mac | Use `CTX_SIZE=16384` to reduce memory |
| Slow first response | Model is loading into GPU memory; wait ~10s |
| Port conflict | Use `PORT=8080` or another free port |

## Platform Notes

- **macOS Apple Silicon**: Full Metal GPU acceleration, best local experience
- **macOS Intel**: CPU-only, works but ~3x slower
- **Linux NVIDIA**: Use `DGGML_CUDA=ON` build for GPU acceleration
- **Linux CPU**: Functional fallback, ~5 TPS
- **Windows**: Use WSL2 with the Linux instructions

## What's Next

Once the endpoint is running, you have a fully private, offline-capable
AI coding assistant. No API keys, no cloud, no data leaves your machine.

For the NVIDIA-optimized path with Lynn Engine's native FP4/FP8 kernels,
see the NVFP4 track documentation.
