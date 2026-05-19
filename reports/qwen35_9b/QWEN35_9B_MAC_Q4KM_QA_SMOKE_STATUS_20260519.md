# Qwen3.5-9B Mac Q4_K_M QA Smoke Status

**Date:** 2026-05-19
**Track:** Mac Stable — Q4_K_M imatrix GGUF + llama.cpp
**Status:** `NOT_RUN` — script ready, awaiting local execution

---

## Current State

The automated QA smoke script is ready at:

```
scripts/local_qwen35_9b_release_qa_smoke.sh
```

It has NOT been executed yet. The server must be started on real macOS hardware
with the Q4_K_M GGUF loaded before running.

## Test Coverage

| Test ID | Description | Status |
|---------|-------------|--------|
| `models_endpoint` | /v1/models returns model | NOT_RUN |
| `english_chat` | English Q&A returns coherent answer | NOT_RUN |
| `chinese_chat` | Chinese Q&A returns non-empty text | NOT_RUN |
| `json_format` | response_format=json_object produces valid JSON | NOT_RUN |
| `multi_turn` | 2-turn conversation retains context | NOT_RUN |
| `long_context_32k` | 32K needle-in-haystack retrieval | SKIP (opt-in) |

## How to Run

```bash
# Terminal 1: Start server
bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh

# Terminal 2: Run smoke (after server shows "ready")
bash scripts/local_qwen35_9b_release_qa_smoke.sh
```

## Output

When executed, the report will be written to:

```
reports/qwen35_9b/local_qwen35_9b_release_qa_smoke_<YYYYMMDD_HHMMSS>.json
```

## Acceptance Criteria

- All 5 core tests PASS (long context is optional)
- JSON report written successfully
- No server crash during test execution

## References

- Runbook: `docs/QWEN35_9B_MAC_Q4KM_QA_RUNBOOK_20260519.md`
- Full QA checklist: `reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md`
- Evidence index: `reports/qwen35_9b/QWEN35_9B_RELEASE_EVIDENCE_INDEX_20260519.md`
