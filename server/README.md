# Lynn Engine HTTP Server

OpenAI-compatible REST API wrapping Lynn engine. Designed as a swap-in
replacement for vLLM at the brain layer (single-prompt, no batching).

## Quick start (DGX Spark or any Blackwell box)

```bash
# Stop vLLM Qwen first to free 60 GB unified mem (Lynn needs 67 GB resident)
docker stop vllm-qwen35a3b

# Run Lynn HTTP server inside the same vLLM container image
docker run --rm --gpus all --ipc=host --user 1000:1000 \
  -v /home/merkyor/models:/models \
  -v /tmp/lynn-engine:/work \
  -w /work \
  -e PYTHONPATH=/work \
  -p 127.0.0.1:18099:18099 \
  nvcr.io/nvidia/vllm:26.03.post1-py3 \
  bash -c "pip install -q --user transformers==5.8.0 fastapi uvicorn pydantic && \
           python3 -m server.openai_http \
             --model /models/Qwen3.6-35B-A3B-FP8 \
             --host 0.0.0.0 --port 18099 \
             --served-name Lynn-Qwen3.6-35B-A3B"
```

Wait ~5 min for "READY" message (40 layers × 6 s each load).

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| GET | `/health` | `{status: ok | loading, model: ...}` |
| GET | `/v1/models` | List served models (single entry) |
| POST | `/v1/completions` | Text completion (OpenAI Completions schema) |
| POST | `/v1/chat/completions` | Chat completion (applies tokenizer chat_template) |

## Structured Guard

For JSON-only calls, the server accepts the OpenAI-style response format and
uses a small serving guard:

```json
{"response_format": {"type": "json_object"}}
```

This forces a JSON-object prefix and returns only the first balanced object.
For internal experiments, `lynn_format_guard` can specify:

```json
{
  "forced_prefix": "{\n  \"",
  "stop_after": "balanced_json",
  "stop_before": ["<think>", "</think>"]
}
```

Supported `stop_after` modes: `balanced_json`, `code_fence`, `bullet_count`.

## Brain integration

In Lynn brain `.env`:

```bash
QWEN_LOCAL_BASE=http://127.0.0.1:18099/v1   # was http://127.0.0.1:18002/v1
```

Or A/B test by swapping primary based on a flag.

## Differences vs vLLM

| Feature | vLLM | Lynn engine |
|---|---|---|
| Concurrent requests | Yes (PagedAttention) | **No** — serialized via asyncio.Lock |
| Streaming | Yes (SSE) | **TODO** (stub in code) |
| `tool_choice` / `tool_call_parser` | Yes (qwen3_coder) | **No** — raw text only |
| `logprobs` in response | Yes | **No** (accepted but ignored) |
| `temperature` > 0 | Yes (sampling) | **No** — greedy only (top-1) |
| KV cache | PagedAttention | Contiguous, in-memory only |

## Performance (Phase 3.1 baseline)

- Prefill T=5: ~1 s
- Decode (incremental, no Triton fuse yet): ~200 ms / token = **5 t/s**
- Phase 3.2 target: ~50-80 ms / token = 12-20 t/s
- Phase 3.3 target: ~10-15 ms / token = 60-100 t/s (matches vLLM)

## When to use Lynn engine vs vLLM

- **Use vLLM (production now)**: tool calling, multi-batch, sampling, all streaming
- **Use Lynn engine (Phase 3+)**: numerical reference, training validation,
  research/profiling, custom kernel experiments
- **Swap to Lynn for production**: when Phase 3.3 hits ≥ 60 t/s AND tool-call
  parser is implemented (Phase 4)
