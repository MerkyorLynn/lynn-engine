# Stage 6 R5-C3A — Gate/Up Prefill Timing Trace

Date: 2026-06-04

Verdict: **trace-only speed evidence; no grouped-MoE FP4-MMA POC, decode TPS,
server behavior, RC quality, or default promotion is banked.**

## What Was Measured

This run reuses the banked R5-C2C real D-row scatter harness at a batched
prefill-shaped gate/up problem:

```text
tokens = 256
top_k = 8
active experts = 8
tokens_per_expert = [256, 256, 256, 256, 256, 256, 256, 256]
per-expert GEMM = M256 x N1024 x K2048
iterations = 20
```

Artifact:
[r5c3a_gateup_prefill_timing_smoke_20260604_203052](r5c3a_gateup_prefill_timing_smoke_20260604_203052/summary.md)

## Result

| Path | Median/avg runtime | Throughput | Scope |
|---|---:|---:|---|
| CUTLASS native NVF4+UE4M3 grouped gate/up, Cooperative | 0.0206432 ms | 416.114 TFLOPS | trace-only |
| CUTLASS native NVF4+UE4M3 grouped gate/up, Pingpong | 0.02376 ms | 361.529 TFLOPS | trace-only |
| Torch BF16 `bmm` same grouped shape | 0.043456 ms median | 197.670 TFLOPS | trace-only baseline |

The best FP4-MMA schedule is about **2.11x** the same-shape BF16 `bmm` trace
baseline. This is the first positive R6000 speed signal for the FP4-MMA lane,
but it is still only gate/up prefill/batch evidence.

## Correctness Preconditions

The timed CUTLASS run also passed the R5-C2C correctness gates:

- `PASS_R5C2C_REAL_D_ROW_SLOT_SCATTER_SMOKE`
- Cooperative/Pingpong host-reference verification passed.
- Real D/ref row digests were captured for 2048 rows per schedule.
- D/ref row digest match, scatter match, and fault injection all passed.

## Non-Claims

- Does not bank `banked_grouped_moe_fp4_mma_poc`.
- Does not bank full MoE speed: SwiGLU, down projection, router, and weighted
  reduction are still out of scope.
- Does not bank decode TPS; this is a prefill/batch gate/up shape.
- Does not bank server behavior, RC quality, or runtime default promotion.

## Next Gate

R5-C3B should add the missing full-MoE correctness side:

1. materialize or otherwise compare post-gate/up values strongly enough to feed
   SwiGLU;
2. run down projection and weighted top-k reduction against BF16/P3 reference;
3. only then measure full active-MoE prefill speed and decide whether the R6000
   FP4-MMA lane can become a real Lynn kernel candidate.
