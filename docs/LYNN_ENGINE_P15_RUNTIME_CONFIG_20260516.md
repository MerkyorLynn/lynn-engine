# Lynn Engine P15 Runtime Config Notes (2026-05-16)

P15 was a regression-hunt pass after Spark/R6000 runs exposed an easy-to-miss
runtime flag trap. The short version:

> `LYNN_PACKED_DECODE=1` is **not** part of the R6000 best TPS configuration.

It packs every decode linear projection onto the generic packed native path.
That sounds attractive, but Q/K/V/O and other small decode projections are
slower there than on the current BF16/native-fused mix. The correct P10/P15
R6000 profile is narrower:

- MoE active experts use packed NVFP4 grouped kernels.
- Linear-attention fused in-projection uses native FP4.
- `lm_head` may use opt-in native FP4 for deterministic greedy serving.
- Generic decode linears stay on the existing BF16 path.

## R6000 Config Trap

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Common environment:

```bash
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_PACKED_SHARED_EXPERT=0
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
```

Critical setting:

```bash
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
```

## A/B Result

Same benchmark, same model, same R6000, group-size 20 graph gate:

| `LYNN_PACKED_DECODE` | groups | full graph path | replay-only |
|---|---:|---:|---:|
| `0` | **107.11 tok/s** | **103.48 tok/s** | **107.23 tok/s** |
| `1` | 90.91 tok/s | 88.15 tok/s | 91.01 tok/s |

This restores the historical P10-P result. The earlier ~91 TPS ceiling was a
configuration regression, not a model or kernel limit.

## MoE Segment Profile

P15 also added `benchmarks/p15_moe_packed_segment_profile.py` to split MoE
decode into measurable pieces.

Representative layer 28, default shared BF16 path:

| Segment | Latency |
|---|---:|
| router + top-k | 0.038 ms |
| active packed NVFP4 experts | 0.116 ms |
| shared BF16 expert | 0.061 ms |
| current full MoE | 0.203 ms |

Enabling packed shared expert regresses:

| Shared expert path | Latency | Quality |
|---|---:|---|
| BF16 | **0.061 ms** | reference |
| packed scalar bridge | 0.142 ms | cosine 0.99999 vs BF16 |
| packed native_fast_2d | 0.233 ms | cosine ~0.962 vs BF16 |

Decision:

- Keep `LYNN_PACKED_SHARED_EXPERT=0`.
- Do not promote native FP4 shared expert until a better kernel and parity gate
  exist.
- Keep the P15 profiler as a regression gate for future MoE work.

## Production Guidance

For R6000 today:

```bash
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_SHARED_EXPERT=0
```

Use the 103/107 TPS numbers as benchmark ceilings and the 88-89 TPS OpenAI
server path as the stable multi-request serving baseline. Further speed work
should target real per-layer compute, not generic packed-decode enablement.
