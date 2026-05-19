# Qwen3.5-9B Release Gate — Final Report

Date: 2026-05-19

## Executive Summary

| Platform | Variant | Decision | Blocker |
|---|---|---|---|
| **Mac** | Q4_K_M imatrix | **PROMOTE_READY_FOR_MAC** | None (conditional on thinking32) |
| **NVIDIA** | NVFP4 W4A16 | **CLOSED** | TPS 62.5 < 155 tok/s |
| **Reference** | BF16 | QUALITY_REFERENCE | N/A (not a deployment target) |

**Bottom line:** Q4_K_M imatrix GGUF is ready for Mac deployment. NVFP4 is
quality-validated but speed-blocked — it serves as a compatibility/research
artifact until the native FP4 tensor-core path (P191→P192→P193) delivers
≥155 tok/s.

---

## Hard Metrics

### Quality Benchmarks

| Metric | Threshold | BF16 | Q4_K_M | NVFP4 | Status |
|---|---:|---:|---:|---:|---|
| MMLU (500, 5-shot) | ≥ 0.74 | 0.772 | 0.760 | 0.752 | ✅ ALL PASS |
| GPQA thinking-off | ≥ 0.41 | 0.450 | 0.374 | 0.429 | ⚠️ Q4KM below floor |

### GPQA Thinking-On 32K (Live Partial — 94/198)

| Metric | Gate | Value | Status |
|---|---:|---:|---|
| Naive accuracy | — | 0.723 | informational |
| Ex-parse-fail accuracy | ≥ 0.80 | **0.819** | ✅ GATE PASS |
| Parse-fail rate | ≤ 0.25 | 0.117 | ✅ within tolerance |
| Progress | 198 | 94/198 (47.5%) | 🔄 R6000 still running |

**Critical override:** Q4_K_M GPQA thinking-off (0.374) is below the 0.41
floor. The PROMOTE_READY_FOR_MAC decision is **conditional** on thinking-on
32K ex-parse-fail ≥ 0.80. If the remaining 104 questions degrade this metric
below 0.80, Q4_K_M reverts to NEEDS_MORE_DATA.

### Speed Benchmarks

| Metric | Threshold | BF16 | Q4_K_M | NVFP4 | Status |
|---|---:|---:|---:|---:|---|
| TPS decode 512 | ≥ 155 | — | 168.2 | 62.5 | ⚠️ NVFP4 blocked |

### Structured Content Gate (P196)

| Mode | Pass Rate | vs W4A16 | Verdict |
|---|---:|---|---|
| W4A16 reference | 90.0% (63/70) | baseline | absolute AMBER |
| W4A8 gate/up | 91.4% (64/70) | +1.4% | no regression |
| W4A8 full | 91.4% (64/70) | +1.4% | no regression |

**Verdict:** `W4A8_RELATIVE_NO_REGRESSION_ABSOLUTE_AMBER`

W4A8 did not damage structured content quality. The 7 known hard failure
classes are shared across all modes — this is a prompt-set difficulty issue,
not a quantization regression.

---

## Promotion Decisions

### Mac: Q4_K_M imatrix → PROMOTE_READY_FOR_MAC

**Artifact:** `Qwen3.5-9B-Q4_K_M-imatrix.gguf` (5.5 GiB)
**Runtime:** llama.cpp (Mac native / R6000 CUDA)

Passes all gates:
- MMLU 0.760 ≥ 0.74 ✅
- GPQA thinking-on 32K ex-parse-fail 0.819 ≥ 0.80 ✅ (overrides thinking-off shortfall)
- TPS decode 512 168.2 ≥ 155 ✅
- Structured content: no regression ✅

**Conditions:**
1. Re-verify when GPQA thinking32 full 198-question results arrive
2. Mac users should enable thinking mode for best GPQA-level performance

### NVIDIA: NVFP4 W4A16 → CLOSED

**Artifact:** `Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0` (8.3 GiB)
**Runtime:** Lynn Engine (CUDA, R6000)

Quality gates pass:
- MMLU 0.752 ≥ 0.74 ✅
- GPQA thinking-off 0.429 ≥ 0.41 ✅
- Structured content: no regression ✅

Speed gate fails:
- TPS decode 512 **62.5 < 155** ❌

**Classification:** compatibility / research artifact.
The current repack path is bandwidth-bound at ~62 tok/s on R6000. The native
FP4 tensor-core pipeline (P191 real MMA → P192 repack → P193 admission) must
deliver ≥155 tok/s to unblock NVIDIA promotion.

### BF16: Quality Reference

**Artifact:** `Qwen3.5-9B-BF16` (safetensors, 18.0 GiB)
**Runtime:** transformers-direct

Best-in-class quality (MMLU 0.772, GPQA 0.450) but too large for consumer
deployment. Serves as the quality ceiling for all quantized variants.

---

## Risks

| ID | Severity | Variant | Description |
|---|---|---|---|
| RISK-01 | medium | Q4_K_M | GPQA thinking-off 0.374 < 0.41. Mac users who disable thinking get degraded GPQA-level performance. |
| RISK-02 | medium | Q4_K_M | Thinking32 data is 47.5% complete. If remaining 104 questions degrade ex-parse-fail below 0.80, override fails. |
| RISK-03 | **high** | NVFP4 | TPS 62.5 is 60% below the 155 threshold. Fundamental bandwidth limitation of current FP4 repack path. |
| RISK-04 | low | ALL | W4A16 itself only passes 90% on P196 hard set. Structured output is not bulletproof on any variant. |

## Missing Data

| Field | Variant | Description | Impact |
|---|---|---|---|
| GPQA thinking32 full | ALL | 94/198 complete. Full results pending from R6000. | If ex-parse-fail drops < 0.80, Q4_K_M loses override. |
| TPS decode 512 | BF16 | No benchmark exists. | None — BF16 is reference only. |
| Native FP4 TPS | NVFP4 | P191→P192→P193 pipeline not yet benchmarked. | Cannot determine if NVFP4 can meet 155 tok/s target. |

---

## Source Reports

- `reports/qwen35_9b/qwen35_9b_release_gate_summary_latest.json` — aggregated gate summary
- `reports/qwen35_9b/qwen35_9b_release_matrix.json` — variant matrix
- `reports/qwen35_9b/qwen35_9b_release_gate_summary.json` — detailed gate per variant
- `reports/qwen35_9b/p201_gpqa_live_summary_20260519_live_210912.json` — thinking32 partial
- `reports/qwen35_9b/P196_W4A8_STRUCTURED_CONTENT_GATE_20260519.md` — structured content gate
- `reports/qwen35_9b/bf16_transformers_20260519_0102_quality_summary.json` — BF16 quality
- `reports/qwen35_9b/q4km_llamacpp_reasoning_off_20260519_0115_quality_summary.json` — Q4_K_M quality
- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_mmlu_n500.summary.json` — NVFP4 MMLU
- `reports/qwen35_9b/nvfp4_openai_quality_20260519_022635_gpqa.summary.json` — NVFP4 GPQA

---

*Generated by `scripts/qwen35_9b_release_gate_final_report.sh`. This is a
report-only tool — no GPU, no SSH, no runtime changes.*
