# Qwen3.5-9B Mac First-Run Runbook Review · 2026-05-19

## Scope

Documentation-only runbook for the Mac first-run path: Qwen3.5-9B Q4_K_M imatrix GGUF through llama.cpp / LM Studio, then local OpenAI-compatible client integration.

## Files

- `docs/QWEN35_9B_MAC_FIRST_RUN_RUNBOOK_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_MAC_FIRST_RUN_RUNBOOK_REVIEW_20260519.md`

## Coverage

The runbook includes:

- Recommended Mac download: Qwen3.5-9B Q4_K_M imatrix GGUF.
- Three-source layout: `dl.merkyorlynn.com`, Hugging Face, ModelScope.
- File size and sha256 placeholder fields.
- Common damaged-download checks.
- `llama-server` example with `-c 32768` and `-ngl 99`.
- Apple Silicon Metal build note, with NVIDIA/CPU called out as separate tracks.
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` smoke examples.
- LM Studio flow.
- Claude Code, Cline, and OpenCode base URL / model name examples.
- thinking-on 32K GPQA positioning as capability mode, not a short-answer TPS benchmark.
- Troubleshooting for context size, long outputs, occupied ports, memory pressure, slow speed, and GGUF damage.

## Consistency checks

- Matches `docs/QWEN35_9B_RELEASE_SITE_COPY_20260519.md` positioning: Mac stable track is Q4_K_M imatrix GGUF + llama.cpp / LM Studio; NVIDIA stable track is Lynn Engine + NVFP4 W4A16.
- Avoids claiming final thinking-on 32K GPQA full-score results.
- Does not present 35B as part of the 9B Mac first-run path.
- Does not count MTP as TPS credit.

## Constraints

- Pure Markdown.
- No code changes.
- No PR opened for this task.
- No GPU commands run.
- No `git add -A` used.
