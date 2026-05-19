# Qwen3.5-9B Release Docs Sync · 2026-05-19

## Scope

User-facing Qwen3.5-9B release docs were synchronized with latest main facts. This is documentation-only; no GPU commands were run and no engine, csrc, server, or script files were modified.

## Files changed

- `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md`
- `docs/QWEN35_9B_MODEL_CARD_DRAFT_20260519.md`
- `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_RELEASE_DOCS_SYNC_20260519.md`

## Facts synced

- Stable Mac path is Q4_K_M imatrix GGUF through `llama.cpp`.
- Stable NVIDIA path is Lynn-native NVFP4 W4A16.
- Current NVFP4 W4A16 package size is 8.248 GiB because BF16 `embed_tokens` and `lm_head` are kept.
- W4A8 / FP4xFP8 resident remains experimental, not default.
- P193-style W4A8 promotion requires P185/P190/P197/P198 gates and blocks promotion if the W4A8 quality report is missing.
- Checksum workflows now reference `scripts/qwen35_9b_release_checksums.py` and `scripts/local_qwen35_9b_verify_checksums.sh`.
- 32K thinking-on GPQA is documented as long-running/in-progress only; no final full 198-question score is claimed.

## Validation

```bash
git diff --check
```

Expected result: PASS.
