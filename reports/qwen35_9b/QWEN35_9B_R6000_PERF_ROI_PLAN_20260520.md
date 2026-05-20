# Qwen3.5-9B R6000 Performance ROI Plan - 2026-05-20

## Decision

Make Qwen3.5-9B the R6000 performance mainline. Keep MTP as a validated-but-not-promoted sidecar path. The highest ROI path is now a true W4A8 / FP4xFP8 dense boundary, not more speculative decoding work.

## Current Baselines

| Path | Status | TPS / Result | Notes |
|---|---|---:|---|
| Lynn-native NVFP4 W4A16 release graph | DEFAULT-safe | 61.69 TPS @ 512 | P25 ready, graph reused |
| llama.cpp Q4_K_M | competitor | ~100+ TPS class on R6000; 168 TPS seen in earlier single path | mature Q4_K_M kernels/layout |
| 9B official MTP K1 | correct | 75.68% accept, ~20-24 effective TPS | sequential verifier cost loses |
| 9B official MTP K2 | correct | 80.56% accept, 23.10 TPS packed profile; 29.77 TPS release eager; 27.87 TPS release graph attempt | verifier/fallback cost loses |
| W4A8 fake-quant quality | quality-safe | MMLU/GPQA and structured comparable to W4A16 | speed claim not native yet |
| True FP8 resident P190 | AMBER/RED | 65.7 TPS, 1.09x, 0/6 exact | math island exists; boundary too loose |

## Why MTP Is Not The Next R6000 Lever

The 9B official inline MTP head is real and working:

- Extracted 15 tensors from BF16 shards to a 486 MB sidecar.
- Sequential K1 accept: 75.68%.
- Batched K2 accept: 80.56%.

But current Lynn verifier cost dominates. Even K2 with high accept stays below the non-speculative graph baseline. MTP should remain an opt-in correctness and future-work track until K2 can reuse graph/native boundaries without eager full-attn fallbacks.

## Highest ROI Path

**Build a true SM120a FP4xFP8 dense FFN boundary.**

The quality case is already good enough for 9B:

- Spark/R6000 W4A8 fake-quant quality is comparable to W4A16.
- P196 structured content gate showed no broad W4A8 quality regression.
- The current failure is runtime/native-boundary readiness, not model quality.

The implementation case is also clear:

- `torch._scaled_mm` cannot express the desired FP8 activation x E2M1 weight layout.
- CUTLASS/CuTe exposes `mma.sync.aligned.kind::f8f6f4...e4m3.e2m1` on R6000.
- P195 fixture gate showed a real native island, but resident all-layer integration drifted because the boundary was too fragmented.

## Work Plan

1. **Fixture-first native dense FFN boundary**
   - Use existing P192/P195 sidecar and fixtures.
   - Implement a single-layer native candidate that consumes prepacked E2M1 weights and FP8 activations.
   - Keep output in a native-owned buffer through gate/up -> SiLU*up -> down where possible.

2. **Resident narrow attach**
   - Attach only to one exact-safe or high-cosine layer group first.
   - Run P37/token drift and 70-prompt structured gate.
   - Do not promote exact-red candidates to default.

3. **Boundary coarsening**
   - Reduce Python/Torch crossings: gate/up and down should not bounce through Python separately.
   - Prefer load-time/offline repack over per-token transpose/contiguous work.

4. **Promotion gates**
   - P37 exact or explicitly accepted AMBER content gate.
   - P25 512 must beat 61.7 TPS materially; target first milestone 75-85 TPS, then 90-100 TPS.
   - Structured/tool-call gates must pass before release language changes.

## Helper Stream Split

- Codex/R6000 main: own 9B native boundary, run R6000 gates, keep stable defaults.
- Claude/Spark side: own 35B Spark MTP/W4A8 experiments and Spark quality runs.
- Helper CLIs: bounded tasks only: fixture PoC, repack audit, promotion report generation.

## Bottom Line

9B W4A8 quality is not the problem. The next useful work is a true FP4xFP8 dense kernel and coarser resident boundary. MTP is proven usable but not yet fast enough to be the 9B speed story.
