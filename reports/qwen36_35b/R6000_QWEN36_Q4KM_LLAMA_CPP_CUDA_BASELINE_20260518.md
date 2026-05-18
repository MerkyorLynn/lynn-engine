# R6000 Qwen3.6-35B-A3B Q4_K_M llama.cpp CUDA Baseline

Date: 2026-05-18

## Scope

This report records the first clean R6000 llama.cpp CUDA baseline for
`Qwen3.6-35B-A3B-Q4_K_M-imatrix.gguf`.

The earlier CPU-only `llama-server` binary accepted `--n-gpu-layers` but did
not link `ggml-cuda`; those numbers are invalid for competitive comparison.
The clean run uses `/root/autodl-tmp/llama.cpp/build-cuda/bin/llama-server`,
built with CUDA 12.8 and `CMAKE_CUDA_ARCHITECTURES=120`.

## Short Context

Source report:
`reports/qwen36_35b/r6000_qwen36_q4km_llamacpp_matrix_20260518_1455_q4km_cuda_short_r6000.json`

| Probe | Wall TPS |
|---|---:|
| Single 128 tokens | 154.39 |
| Single 256 tokens | 202.47 |
| Single 512 tokens | 207.29 |
| Concurrent 2, total | 306.34 |
| Concurrent 4, total | 400.39 |
| Concurrent 8, total | 500.66 |
| 8192 chars prompt, 64 decode | 114.22 |

The llama.cpp server log reports CUDA eval throughput around 214 tok/s for
the 256/512-token single-stream runs, so the 200+ wall TPS is not just an
HTTP accounting artifact.

## Long Context

Source report:
`reports/qwen36_35b/r6000_qwen36_q4km_llamacpp_matrix_20260518_1458_q4km_cuda_long_r6000.json`

| Prompt size | Prompt tokens | Decode tokens | Wall TPS |
|---|---:|---:|---:|
| 32768 chars | 5839 | 128 | 91.13 |
| 65536 chars | 11664 | 128 | 84.19 |

The corresponding llama.cpp eval lines remain around 200 tok/s during decode;
the lower wall TPS includes prefill. Prompt eval is around 7.7k-7.8k tok/s in
the long-context runs.

## Comparison Anchor

| Runtime | Quant class | Current R6000 signal |
|---|---|---:|
| llama.cpp CUDA | Q4_K_M-imatrix, W4A16-class | 207 tok/s single 512, 501 tok/s concurrent 8 total |
| Lynn Engine safe default | Lynn-native W4A16 NVFP4 | ~104-107 decode TPS, structured clean |
| Lynn Engine AMBER | W4A16 NVFP4 with opt-in fast flags | ~110-114 decode TPS, structured clean but exact drift |

Q4_K_M is a W4A16-class path: 4-bit weights with high-precision activations
and accumulation. It is not W4A8 or W4A4. Its quality results on Spark
(`83.00%` MMLU, `50.00%` GPQA) make it the right quality-stable speed
reference for Lynn-native W4A16.

## Engineering Implication

The llama.cpp gap points to kernel/runtime structure, not model quality:

1. Offline repack into decode-friendly block layouts.
2. Fewer per-token kernel launches.
3. Fused CUDA boundaries for GDN/SSM and top-k MoE.
4. CUDA graph reuse over stable decode subgraphs.

For Lynn, the immediate target is not a C++ HTTP rewrite. The measured host
gap in Lynn is already small. The next useful work is a strict, numerically
safe kernel island:

- active MoE boundary fusion without changing routing/math order
- linear/GDN boundary fusion around the current in-proj/recurrent/conv/norm
  segments
- full-attention cache and RoPE allocation cleanup without QKV reordering

## Native MoE P125 Side Note

The latest P125 strict-boundary allowlist probes remain closed:

| Scope | Candidate | Exact | Min prefix | Median speedup |
|---|---|---:|---:|---:|
| full-attn layers | strict_fused_boundary | 0/3 | 3 | 0.971x |
| full-attn layers | cuda_scalar_contract | 0/3 | 3 | 1.085x |
| full-attn layers | grouped_per16_nonatomic | 0/3 | 2 | 1.087x |
| linear-attn layers | strict_fused_boundary | 0/3 | 3 | 0.997x |
| linear-attn layers | cuda_scalar_contract | 0/3 | 3 | 1.040x |
| linear-attn layers | grouped_per16_nonatomic | 0/3 | 3 | 1.067x |

These are useful research probes but not promotion candidates. The next native
MoE attempt must preserve the Triton numerical contract more tightly before
chasing speed.
