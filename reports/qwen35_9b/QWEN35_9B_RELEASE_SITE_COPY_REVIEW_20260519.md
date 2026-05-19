# Qwen3.5-9B Release Site Copy Review · 2026-05-19

## Scope

Drafted first-release website and download-page copy for Qwen3.5-9B. This is documentation-only and does not modify runtime code.

## Files

- `docs/QWEN35_9B_RELEASE_SITE_COPY_20260519.md`
- `docs/QWEN35_9B_DOWNLOAD_LAYOUT_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_RELEASE_SITE_COPY_REVIEW_20260519.md`

## Copy decisions

- First screen positions Lynn Engine as local inference with two stable tracks:
  - Mac stable track: Qwen3.5-9B Q4_K_M imatrix GGUF + llama.cpp.
  - NVIDIA performance track: Lynn Engine + NVFP4 W4A16.
- Domain split is explicit:
  - `engine.merkyorlynn.com` for homepage, docs, install guide, and model card.
  - `dl.merkyorlynn.com` for large artifacts and checksums.
- Download layout includes:
  - Mac Q4_K_M GGUF.
  - NVIDIA NVFP4 W4A16.
  - NVIDIA compact NVFP4 candidate, clearly marked candidate.
  - `checksums.sha256`.
  - model card.
  - release gate JSON.
- Installation copy is intentionally simple:
  - Mac: llama.cpp, LM Studio, CLI.
  - NVIDIA Linux: Lynn Engine one-command script placeholder.
  - Windows NVIDIA: roadmap, WSL2 first.

## Guardrails

The copy avoids overclaiming:

- 9B thinking-on 32K GPQA is still running; no final 198-question score is claimed.
- 35B is a side track, not part of the 9B first-release path.
- MTP is excluded from TPS credit.
- W4A8 / FP4xFP8 resident remains experimental.
- NVIDIA compact NVFP4 is a candidate, not stable.

## Validation

No tests or GPU commands were run by design. Only the three allowed documentation/report files were staged and committed.
