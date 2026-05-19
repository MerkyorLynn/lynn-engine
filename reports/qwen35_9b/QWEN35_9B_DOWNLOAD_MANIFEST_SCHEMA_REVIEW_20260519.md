# Qwen3.5-9B Download Manifest Schema Review · 2026-05-19

## Scope

Documentation-only release infrastructure note for the Qwen3.5-9B download manifest and schema. No runtime code was touched.

## Files

- `docs/QWEN35_9B_DOWNLOAD_MANIFEST_SCHEMA_20260519.md`
- `reports/qwen35_9b/QWEN35_9B_DOWNLOAD_MANIFEST_SCHEMA_REVIEW_20260519.md`

## Coverage

The manifest schema covers:

- Mac Q4_K_M imatrix GGUF for the stable `mac_llamacpp` track.
- BF16 reference artifact for calibration and quality ceiling use.
- NVIDIA NVFP4 compatibility artifact for the `nvidia_lynn_engine` track.

Required artifact fields are documented:

- `artifact_id`
- `model_id`
- `variant`
- `quant`
- `runtime_track`
- `filename`
- `size_bytes`
- `sha256`
- `sources`
- `recommended`
- `status`
- `quality_metrics`
- `speed_metrics`
- `license`
- `notice`
- `created_at`

## Mirror consistency process

The doc specifies the expected three-source verification flow:

1. Manifest version lock.
2. `HEAD` / `Content-Length` / range support checks.
3. Interrupted download and `.partial` handling.
4. SHA256 verification across `dl.merkyorlynn.com`, HF, and ModelScope.
5. Release-blocking treatment for any unknown size, unknown sha256, or mirror mismatch.

## Page consumption model

The doc keeps the site architecture split clear:

- `engine.merkyorlynn.com` reads manifest JSON and renders download cards.
- `dl.merkyorlynn.com` serves large files and static release assets.
- GitHub stores schema, small JSON manifests, and reviewable release gate metadata.

## Consistency notes

- Mac stable track remains Q4_K_M imatrix GGUF + llama.cpp / LM Studio / CLI.
- NVIDIA track remains Lynn Engine + NVFP4 compatibility artifact.
- BF16 is reference-only, not the default first-run artifact.
- Unknown sha256 values are placeholders marked release-blocking before public publish.
- The doc avoids claiming final 9B thinking-on 32K GPQA full results.
- 35B is excluded from the 9B first-release download path.
- MTP is excluded from TPS credit.

## Constraints

- Pure Markdown.
- No GPU commands run.
- No PR opened.
- No `git add -A` used.
