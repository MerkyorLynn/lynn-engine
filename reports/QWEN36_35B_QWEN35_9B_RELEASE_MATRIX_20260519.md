# Qwen3.6-35B + Qwen3.5-9B Release Matrix

Date: 2026-05-19  
Scope: Qwen3.6-35B-A3B W4A16/NVFP4 MoE serving and Qwen3.5-9B dense serving.

## Executive Readout

35B release posture: keep the Lynn W4A16 safe default on exact Triton active-MoE
math, with `LYNN_ROUTER_TOPK_OUT_BUFFER=1` as the current default candidate.
The safe serving line is now the 107-108 decode TPS class; llama.cpp Q4_K_M on
R6000 remains the external speed reference at 207 single-stream TPS and 501
total TPS at concurrency 8.

9B release posture: Q4_K_M is the immediate speed path, while Lynn NVFP4 is the
NVIDIA-native path. The Lynn 9B default is P150 linear graph, with P173 and
P174 kept as exact opt-in building blocks rather than a new default.

## Quality Matrix

| Line | Variant | Stack | MMLU 500 | GPQA Diamond | Release posture |
|---|---|---|---:|---:|---|
| 35B | BF16 official | OpenAI-compatible eval | 86.40% (432/500) | 45.45% (90/198) | quality ceiling |
| 35B | W4A16 NVFP4 official | Lynn Engine CUDA | 84.40% (422/500) | 49.49% (98/198) | default NVIDIA serving quality |
| 35B | Q4_K_M imatrix | llama.cpp CUDA | 83.00% (415/500) | 50.00% (99/198) | portable speed reference |
| 9B | BF16 official | transformers direct | 77.20% (386/500) | 44.95% (89/198) | quality ceiling |
| 9B | Lynn NVFP4 | Lynn Engine CUDA | 75.20% (376/500) | 42.93% (85/198) | NVIDIA-native release path |
| 9B | Q4_K_M | llama.cpp CUDA | 76.00% (380/500) | 37.37% (74/198) | fastest portable path; GPQA lower |

## Speed Matrix

| Line | Candidate / route | Exactness / gate | 128 | 256 | 512 | Concurrent x8 | Status |
|---|---|---|---:|---:|---:|---:|---|
| 35B | Lynn W4A16 safe default full | P37 exact, hard structured pass | - | - | 107.43 decode TPS | - | previous default-class baseline |
| 35B | Lynn W4A16 router top-k out | P37 exact, hard structured 40/40 | - | - | 108.06 decode TPS | - | default candidate |
| 35B | llama.cpp Q4_K_M imatrix R6000 | quality/speed reference | 154.39 wall TPS | 202.47 wall TPS | 207.29 wall TPS | 500.66 total TPS | external speed target |
| 9B | llama.cpp Q4_K_M R6000 | quality/speed complete | 121.8 TPS | 165.2 TPS | 168.2 TPS | 420.6 total TPS | fastest 9B path |
| 9B | Lynn NVFP4 P150 linear graph | P25 ready, graph reused on 9/9 requests | 60.80 decode TPS | 61.47 decode TPS | 61.69 decode TPS | ~60.11 total TPS in P151 | default Lynn 9B path |
| 9B | Lynn NVFP4 P173 dense gate/up + RoPE cache | P25 ready, graph reused on 9/9 requests | 61.85 decode TPS | 62.45 decode TPS | 62.55 decode TPS | not rerun | opt-in candidate |
| 9B | Lynn NVFP4 P174 full-attn tail graph | local exact 8/8 full-attn layers | - | - | local tail replay only | - | opt-in building block |

P174 local note: full-attention tail graph replay is exact on 8/8 layers and
saves about 0.017-0.019 ms per full-attention layer after input copy, roughly
0.13-0.15 ms/token across the 8 full-attention layers. It is useful plumbing,
not a standalone service TPS promotion.

## Decision Matrix

| Bucket | 35B MoE | 9B dense |
|---|---|---|
| Default | W4A16 safe default with exact Triton active-MoE and `LYNN_ROUTER_TOPK_OUT_BUFFER=1`; current default line is 108 TPS class | P150 linear graph profile: `LYNN_LINEAR_STATE_UPDATE=inplace`, reusable/prewarmed linear block graph |
| Opt-in | Exact but small boundaries such as effective-scale, prepared Triton active-MoE, shared in-place, active scratch | P173 dense gate/up + RoPE cache; P174 full-attn tail graph; dense gate/up fused |
| Research-only | Native packed MoE gate/up replacement; MoE sidecar/scratch/effective-scale when they do not beat default serving | Full-attn graph slots until capture/reuse ABI is service-safe; larger dense FFN/TensorCore repack work |
| Closed | Native packed MoE resident backends that fail P37; router softmax-out as standalone knob; prepared/shared-inplace default promotion attempts below safe default | Full 35B fast profile on 9B due to drift; packed dense FFN microprobes that drift or are flat; mm-out-only path |

## 35B Native MoE Exactness Status

Do not promote native packed MoE replacement to P25/structured gates. The latest
P160-P162 diagnostics show:

- FP4 decode and per-term products are bit-exact against Triton.
- Drift begins inside the 256-term FP32 `tl.sum` reduction tree for gate/up.
- Common native reduction trees reproduce native output, not Triton output.

The release-safe path is therefore to keep Triton active-MoE as the exact
authority and optimize larger exact boundaries around it.

## Next Step: 35B P168 Linear-Core Focus

P168 confirms the next high-ROI 35B work is the linear/GDN core, not another
router/shared-expert micro knob. Across 30 linear-attention layers, the repeated
measured work is:

| Segment | Sum across 30 layers | Mean / layer | Share |
|---|---:|---:|---:|
| fused native FP4 in-proj | 2.107 ms/token | 0.0702 ms | 38.4% |
| recurrent fused prepare / GDN | 1.132 ms/token | 0.0377 ms | 20.6% |
| conv update | 0.984 ms/token | 0.0328 ms | 18.0% |
| gated RMSNorm | 0.605 ms/token | 0.0202 ms | 11.0% |
| out proj BF16 | 0.449 ms/token | 0.0150 ms | 8.2% |
| split qkv / repeat | 0.206 ms/token | 0.0069 ms | 3.8% |

Recommended next probe:

1. Build a fixture-style linear-core contract for representative linear layers.
2. Target an exact caller-owned scratch boundary around `in_proj -> conv -> recurrent`.
3. Keep default serving unchanged until the boundary passes local exactness,
   P37, P25, and hard structured gates.

## Source Anchors

- 35B quality: `reports/qwen36_35b/spark_qwen36_official_*_20260518.json`
- 35B default: `reports/qwen36_35b/P163_ROUTER_BOUNDARY_PROMOTION_20260519.md`
- 35B P168: `reports/qwen36_35b/P168_LINEAR_CORE_SEGMENT_CENSUS_20260519.md`
- 35B llama.cpp reference: `docs/QWEN36_W4A16_KERNEL_REFACTOR_PLAN_20260518.md`
- 9B matrix: `reports/qwen35_9b/qwen35_9b_release_matrix.md`
- 9B P150: `reports/qwen35_9b/QWEN35_9B_NVFP4_LINEAR_GRAPH_SERVING_P150_20260519.md`
- 9B P173: `reports/qwen35_9b/P173_ROPECACHE_DENSEGATEUP_SERVICE_GATE_20260519.md`
- 9B P174: `reports/qwen35_9b/P174_FULL_ATTN_TAIL_GRAPH_PROBE_STUB.md`
