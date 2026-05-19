# Qwen3.5-9B Mac Q4_K_M QA Runbook

**Date:** 2026-05-19
**Track:** Mac Stable — Q4_K_M imatrix GGUF + llama.cpp
**Scope:** This runbook covers ONLY the Mac/Q4_K_M stable track. It does NOT
cover NVIDIA NVFP4, W4A8, or any other variant.

---

## Prerequisites

- macOS Apple Silicon (M1/M2/M3/M4)
- llama.cpp installed (`brew install llama.cpp` or built from source)
- Qwen3.5-9B Q4_K_M GGUF downloaded to `~/Models/Lynn/Qwen3.5-9B/q4_k_m/`
- Python 3.10+ (for smoke output parsing)

## Step 1: Start the Server

```bash
bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

This will:
- Auto-discover the GGUF in `~/Models`, `~/Downloads`, or common paths
- Auto-discover `llama-server` from PATH or Homebrew
- Start on port 18099 with model name `qwen35-9b-q4km`
- Print a banner with endpoint details

Wait for `"all slots are idle"` or similar ready message in logs.

## Step 2: Run QA Smoke

In a **separate terminal**:

```bash
bash scripts/local_qwen35_9b_release_qa_smoke.sh
```

This tests:
1. `/v1/models` endpoint responds with the model
2. English chat returns a coherent answer
3. Chinese chat returns non-empty Chinese text
4. `response_format: json_object` produces valid JSON
5. Multi-turn context (remembers "42" across 2 turns)
6. 32K long context (skipped by default)

### Output

JSON report written to:
```
reports/qwen35_9b/local_qwen35_9b_release_qa_smoke_<timestamp>.json
```

### Expected Result

```
  Results: 5 passed, 0 failed (6 total, 1 skipped)
  Status:  ALL PASS ✓
```

## Step 3: Long Context (Optional)

To also run the 32K needle-in-haystack test:

```bash
SKIP_LONG=0 bash scripts/local_qwen35_9b_release_qa_smoke.sh
```

This sends ~32K chars with a hidden needle and verifies retrieval. It may take
30-60 seconds on Apple Silicon. Only run this if you have time and want to
verify the full 32K context window.

## Customization

| Variable | Default | Description |
|----------|---------|-------------|
| `BASE_URL` | `http://127.0.0.1:18099/v1` | Server endpoint |
| `MODEL` | `qwen35-9b-q4km` | Model name to test |
| `TIMEOUT` | `120` | Request timeout (seconds) |
| `SKIP_LONG` | `1` | Set to `0` to enable 32K long-context test |
| `OUT_DIR` | `reports/qwen35_9b` | Output directory |

## Interpreting Results

| Test | What it proves |
|------|---------------|
| `models_endpoint` | Server is alive and serving the correct model |
| `english_chat` | Basic generation works |
| `chinese_chat` | Multilingual capability intact |
| `json_format` | Structured output works (critical for agent use) |
| `multi_turn` | Context carry-over works (critical for agent use) |
| `long_context_32k` | 32K context window functional (optional) |

## If Tests Fail

| Failure | Likely cause | Fix |
|---------|-------------|-----|
| Health check fails | Server not running | Start server first (Step 1) |
| `models_endpoint` fails | Wrong port or model name | Check `--port` and `-a` flags |
| `json_format` fails | Model struggles with JSON | This is a quality signal, not a crash |
| `long_context_32k` fails | OOM or timeout | Reduce `CTX_SIZE` or use `parallel=1` |

## Relation to Full QA Checklist

This smoke covers items A.3, A.4, A.5, and A.6 from the full QA checklist at
`reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md`. Items A.1 (download/checksum)
and A.2 (server startup) are prerequisites that should be verified manually before running
this smoke.

Items A.7 (stress boundary) and A.8 (agent integration) require separate, longer tests
and are not covered here.
