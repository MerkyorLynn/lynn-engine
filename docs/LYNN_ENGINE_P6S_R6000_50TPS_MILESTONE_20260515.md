# Lynn Engine P6-S R6000 50TPS Milestone

Date: 2026-05-15
Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96GB
Model: Lynn 27B variable-pruned NVFP4 artifact

## Executive Summary

P6-S turns the 27B NVFP4 Lynn engine from a ~30TPS eager decode path into a
working opt-in resident decode path above the short-term 50TPS target.

The current serving-shaped fast path reaches:

- 6-prompt smoke: 63.85 / 65.44 / 65.42 / 65.15 / 65.38 / 65.35 TPS
- 96-token resource probe: 64.40 TPS
- Hybrid block-graph probe: 15.12 ms/token = 66.12 TPS
- Output quality: coherent Chinese, Python, math, SQL, and engineering prompts

This is still slow-dequant resident BF16 execution from the NVFP4 checkpoint,
not final native packed FP4 GEMM. Native packed FP4 remains the route to the
100TPS and 200TPS targets.

## Current Fast-Path Environment

```bash
PYTHONPATH=/root/autodl-tmp/lynn-engine
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
LYNN_MOE_IMPL=triton
LYNN_QK_NORM_ROPE_BACKEND=triton
LYNN_LINEAR_ATTN_INPROJ_FUSED=1
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_LINEAR_STATE_UPDATE=inplace
```

Do not enable `LYNN_RMSNORM_GATED_BACKEND=triton` by default. Its microbench
speedup did not translate to full-token speed and regressed the full-token
profile.

## Milestones

| Milestone | Result | Meaning |
|---|---:|---|
| P6-K eager best | 32.73 ms/token, 30.55 TPS | Triton MoE + fused Q/K norm RoPE baseline |
| P6-N no-allocation eager | 30.29 ms/token, 33.02 TPS | Static token/position buffers remove per-step allocation |
| P6-O fused linear in-proj | 30.10 ms/token, 33.23 TPS | Four linear-attn projections fused; small win |
| P6-M full-token graph ceiling | 13.38 ms/token, 74.73 TPS | Launch/Python overhead is a major bottleneck |
| P6-P 3-layer linear block graph | 0.95 ms/block, 2.33x | Linear block is graph-friendly |
| P6-Q hybrid graph probe | 15.12 ms/token, 66.12 TPS | Linear blocks graphed, full-attn eager |
| P6-R graph parity | max_abs 0.0, cosine ~1.0 | Graph block exactly matches eager block |
| P6-S resident graph smoke | 63-65 TPS | Real generate path, coherent outputs |

## Resource Snapshot

Measured with the current opt-in fast path:

| Stage | GPU Mem Used | GPU Util | Power |
|---|---:|---:|---:|
| before load | 0 MB / 97887 MB | 0% | 35.7W |
| after load | 72019 MB / 97887 MB | 34% | 146.4W |
| after generate | 62687 MB / 97887 MB | 92% | 176.2W |

Notes:

- After-load memory includes temporary allocator/cache pressure; steady generate
  settled around 62.7GB in this run.
- This is slow-dequant resident BF16 memory, not final native packed FP4 memory.
- Native packed FP4 should reduce the weight-resident footprint materially.

## Correctness Trap Found And Fixed

The first resident graph implementation reached ~64TPS but generated
`Mo!!!!...`. Root cause:

- CUDA Graph replay captured kernels that referenced recurrent/conv state tensor
  addresses.
- The eager decode path updated linear-attn state by replacing Python dict
  entries with new tensors.
- Graph replay kept writing/reading the original addresses, so recurrent state
  did not advance across tokens.

Fix:

```bash
LYNN_LINEAR_STATE_UPDATE=inplace
```

When enabled, `_decode_layer` updates linear-attn recurrent and conv state with
`copy_()` into stable tensors. P6-R then confirmed graph block output and state
parity with eager:

- output max_abs: 0.0
- recurrent/conv state max_abs: 0.0

## 6-Prompt Smoke

Prompts covered:

- MoE active parameters explanation
- Python recursive factorial
- RoPE vs ALiBi comparison
- arithmetic word problem
- SQLite grouped count query
- correctness gate vs performance engineering short essay

All produced coherent outputs. TPS values:

```text
63.85, 65.44, 65.42, 65.15, 65.38, 65.35
```

## Framework Comparison Policy

The Lynn 27B artifact is a variable-pruned NVFP4 checkpoint. Most upstream
engines do not support this artifact shape directly:

- llama.cpp: GGUF path does not natively support this variable-expert NVFP4
  artifact today.
- vLLM / SGLang / TRT-LLM: standard loaders do not directly support Lynn's
  variable-pruned NVFP4 layout.

Therefore framework comparisons must be split into:

1. Same artifact: Lynn engine only, because it is the only current loader.
2. Reference variants: Q4_K_M or standard v8-RTN artifacts where upstream
   engines can run, clearly marked as not apples-to-apples.

## Next Steps Toward 100TPS

1. Productize the P6-S fast path in resident/openai server instead of benchmark
   scripts.
2. Reduce graph capture overhead by caching one graph set per active request
   slot instead of capturing per request.
3. Make full-attention decode graph-friendly by preallocating fixed-length
   views or bucketing context lengths.
4. Resume native packed FP4 GEMM for projection/MoE weights to reduce both
   memory and compute bandwidth.
5. Re-run V8/V9 strict once final 27B Recovery v1.1 is merged and quantized.
