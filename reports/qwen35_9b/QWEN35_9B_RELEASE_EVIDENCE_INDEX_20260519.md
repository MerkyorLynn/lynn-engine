# Qwen3.5-9B Release Evidence Index

**Date:** 2026-05-19
**Status:** IN PROGRESS (32K thinking GPQA long-eval live)
**Branch:** `claude/qwen35-9b-release-evidence-index-20260519`

---

## Release Tracks

### Track 1: Mac Stable — Q4_K_M imatrix GGUF + llama.cpp

| Evidence | Status | Source |
|----------|--------|--------|
| Model artifact | 5.49 GiB GGUF | `Qwen3.5-9B-Q4_K_M.gguf` |
| Quality: MMLU 500 5-shot | 76.00% (380/500) | `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.json` |
| Quality: GPQA Diamond (thinking-off) | 37.37% (74/198) | same source |
| Quality: GPQA Diamond (thinking-on) | **IN PROGRESS** — 32K eval live on R6000 | `scripts/r6000_qwen35_9b_q4km_thinking32_gpqa_long.sh` |
| Speed: single 512 | 164.5 TPS | `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.json` |
| Speed: concurrent x8 | 166.1 TPS total | same source |
| Speed: 32K single slot | 55.5 TPS | same source |
| Launcher script | exists | `scripts/local_qwen35_9b_q4km_llamacpp_server.sh` |
| Smoke test | exists | `scripts/local_qwen35_9b_q4km_smoke.sh` |
| Mac QA checklist | PENDING_QA | `reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md` |
| Install quickstart | exists | `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md` |

### Track 2: NVIDIA Stable — Lynn-native NVFP4 W4A16 (8.25 GiB)

| Evidence | Status | Source |
|----------|--------|--------|
| Model artifact | 8.25 GiB (P199 audited) | P199 size audit |
| Quality: MMLU 500 5-shot | 75.20% | `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.summary.json` |
| Quality: GPQA Diamond (thinking-off) | 42.93% (85/198) | `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.summary.json` |
| Speed: safe profile decode | ~60-62 TPS | `reports/qwen35_9b/QWEN35_9B_NVFP4_LINEAR_GRAPH_SERVING_P150_20260519.md` |
| P193 boundary gate | CLOSED (MMA fragment + P185 absent) | `reports/qwen35_9b/p193_native_boundary_admission_p191_real_mma.json` |
| P199 size audit | GREEN (8.25 GiB breakdown complete) | `reports/qwen35_9b/P199_QWEN35_9B_NVFP4_SIZE_AUDIT_20260519.md` |
| Compact 5.70 GiB tier | CANDIDATE only — not promoted | `reports/qwen35_9b/compact_nvfp4_shrink_gate_20260519_live_compact.json` |

### Track 3: Experimental — W4A8 / FP4xFP8 Resident (NOT for release)

| Evidence | Status | Source |
|----------|--------|--------|
| P197 fake-W4A8 drift | AMBER (32/40 exact, combined 0.975) | `reports/qwen35_9b/p197_fp4xfp8_token_drift_20260519_1850_fake_p197.json` |
| P191 MMA fragment layout | BROKEN (cosine -0.015 vs scalar) | `reports/qwen35_9b/P191_R6000_E4M3_E2M1_MMA_LAYOUT_GREEN_20260519.md` |
| P196 structured content gate | documented | `reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` |
| P197b drift isolation | NOT YET RUN (R6000 busy) | branch `claude/qwen35-9b-w4a8-drift-fix-20260519` |
| Production promotion | **BLOCKED** | MMA fragment fix + P185 quality gate required |

---

## Blockers

| Blocker | Affects | Resolution Path |
|---------|---------|-----------------|
| Mac QA not executed | Track 1 Mac stable | Run QA checklist on real macOS hardware |
| 32K GPQA thinking-on in progress | Track 1 final quality number | Wait for R6000 long-eval to complete |
| P193 CLOSED (MMA fragment) | Track 3 only | Fix SM120a fragment layout (P191) |
| Compact NVFP4 no quality gate | Compact 5.70 GiB variant | Run MMLU/GPQA on compact artifact |
| P185 W4A8 quality report absent | Track 3 only | Run MMLU/GPQA with W4A8 enabled |

---

## Live Jobs (R6000)

| Job | Port | Status | Output Path |
|-----|------|--------|-------------|
| Q4_K_M GPQA thinking32 198q | 18197 | RUNNING | `thinking32/qwen35_9b_q4km_gpqa_thinking32_20260519_*.jsonl` |

---

## Key Artifacts by File

| Path | Content |
|------|---------|
| `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md` | Release decision: 2 tracks, promotion rules |
| `docs/QWEN35_9B_MODEL_CARD_DRAFT_20260519.md` | Public model card draft |
| `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md` | User-facing install guide |
| `reports/qwen35_9b/QWEN35_9B_OVERNIGHT_STATUS_20260519.md` | Overnight matrix snapshot |
| `reports/qwen35_9b/QWEN35_9B_RELEASE_QA_STATUS_20260519.md` | QA checklist (237 items) |
| `reports/qwen35_9b/P199_QWEN35_9B_NVFP4_SIZE_AUDIT_20260519.md` | NVFP4 8.25 GiB size breakdown |
| `reports/qwen35_9b/P201_QWEN35_9B_THINKING32_GPQA_SUMMARY_20260519.md` | Thinking32 summarizer status |
| `reports/qwen35_9b/compact_nvfp4_shrink_gate_20260519_live_compact.json` | Compact 5.70 GiB tier gate |
| `reports/qwen35_9b/r6000_qwen35_9b_q4km_cuda_baseline_20260519_1732.json` | Q4_K_M CUDA speed baseline |
| `reports/qwen35_9b/p197_fp4xfp8_token_drift_20260519_1850_fake_p197.json` | W4A8 token drift results |

---

## What This Index Does NOT Prove

1. **Mac QA has not been executed on real hardware.** The checklist exists but all items are PENDING.
2. **32K thinking-on GPQA is not final.** The eval is live on R6000; current partial results are not reported here.
3. **Compact NVFP4 (5.70 GiB) is not validated.** The shrink tier is a candidate; no MMLU/GPQA has been run on it.
4. **W4A8/FP4xFP8 is not release-ready.** AMBER drift + broken MMA fragment = experimental only.
5. **35B A3B is not part of this release.** It is a separate research track on Spark.

---

## Recommended Release Posture

| Decision | Reasoning |
|----------|-----------|
| Ship Q4_K_M Mac track as stable | Speed proven, quality acceptable, install story complete |
| Ship NVFP4 W4A16 NVIDIA track as safe | Quality superior to Q4_K_M on GPQA, size known (8.25 GiB) |
| Do NOT promote W4A8/FP4xFP8 | AMBER drift unresolved, MMA layout broken |
| Do NOT ship compact NVFP4 | No quality gate executed |
| Wait for thinking32 GPQA to finish | It is a capability signal, not a blocker for stable tracks |
