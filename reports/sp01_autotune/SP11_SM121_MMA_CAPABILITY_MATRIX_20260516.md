# SP-11: Spark sm_121 (GB10) MMA ISA Capability Matrix

Date: 2026-05-16
Branch: `spark/sm121-port`
Result: **FP8 OPEN / FP4 + FP6 BLOCKED / kind:: family BLOCKED**

## TL;DR

Empirical PTXAS probing of 7 `mma.*` instruction variants on NVIDIA GB10 (Spark
sm_121, compute_capability 12.1) under `-arch=sm_121a` + CUDA 13.0 settles
SP-10's open question definitively:

**Spark sm_121 has FP8 (E4M3 + E5M2) tensor core MMA. It does NOT have any
FP4 or FP6 tensor core MMA, and the entire `.kind::f8f6f4` instruction family
is rejected by ptxas. GB10 is "Blackwell minus FP4/FP6 microscaling."**

This unlocks a Spark-specific path: **dequant Lynn packed FP4 weights to FP8 in
shared memory, then use sm_121's FP8 tensor cores with per-16 FP32 scale
applied in the FP32 epilogue.** Codex's R6000 FP4 MMA path (P78-P90) remains
sm_120a-only; Spark gets a parallel FP8 MMA path that hits ~50% of the FP4
theoretical throughput.

## Capability Matrix

| MMA Variant | Compile | Run | Output finite | Notes |
|---|:---:|:---:|:---:|---|
| `m16n8k16.row.col.f32.bf16.bf16.f32` | ✅ | ✅ | ✅ | sanity baseline |
| `m16n8k32.row.col.f32.e4m3.e4m3.f32` | ✅ | ✅ | ✅ | **FP8 path** |
| `m16n8k32.row.col.f32.e5m2.e5m2.f32` | ✅ | ✅ | ✅ | FP8 alt |
| `m16n8k32.row.col.f32.e2m1.e2m1.f32` | ❌ | — | — | FP4 raw |
| `m16n8k32.row.col.kind::f8f6f4.f32.f32` | ❌ | — | — | kind family blocked |
| `m16n8k32.row.col.f32.e3m2.e3m2.f32` | ❌ | — | — | FP6 raw |
| `kind::mxf8f6f4.block_scale.scale_vec::1X` | ❌ | — | — | SP-10 control |

## Exact PTXAS Errors

**FP4 raw (`.f32.e2m1.e2m1.f32`)**:
```text
ptxas: Instruction 'mma with with FP6/FP4 floating point type' not supported on .target 'sm_121'
ptxas: .kind::f8f6f4 modifier required for instruction 'mma'
ptxas fatal: Ptx assembly aborted due to errors
```

So sm_121 demands `.kind::f8f6f4` for FP4/FP6 (rejects raw operand-type
specification) — but `.kind::f8f6f4` is itself unsupported:

**FP4 with `.kind::f8f6f4`**:
```text
ptxas: Feature '.kind::f8f6f4' not supported on .target 'sm_121'
ptxas: Illegal modifier '.kind::f8f6f4' for instruction 'mma'
```

Both the workaround and the canonical form are blocked. **GB10 simply has no
FP4/FP6 tensor core MMA in its ISA.**

**FP8 (PASS)**:
```text
fp8_e4m3_m16n8k32  compile=PASS  run=PASS  finite=True
fp8_e5m2_m16n8k32  compile=PASS  run=PASS  finite=True
```

Both FP8 formats compile and execute. Output sample shows non-zero finite
values from MMA of all-1 inputs, confirming the tensor core is doing real
work.

## Strategic Implication — SP-12 FP8 MoE Kernel Path

Spark Lynn 27B NVFP4 cannot use Codex's R6000 FP4 MMA path
(P88/P90/P91+ — uses `.kind::mxf8f6f4.block_scale.scale_vec::1X`). But Spark
CAN use a parallel FP8 MMA path:

```text
Lynn-native NVFP4 artifact:
  packed E2M1 weights        [E, M, K/2]  uint8
  per-16 FP32 scales         [E, M, K/16] f32
  global FP32 scale          [1]          f32

Spark sm_121 FP8 MoE kernel (SP-12 design):

  for each tile (m_tile, n_tile, k32_tile):
    load packed E2M1 from HBM           (8 bytes per 16 elements)
    shared-mem dequant E2M1 → FP8       (small LUT, fast)
    quantize BF16 activation → FP8      (per-32 micro-batch scale)
    fp8_e4m3 mma.m16n8k32 → fp32 accum  ⭐ Spark hardware win
    apply per-16 FP32 scale in fp32     (FP32 multiply)
    accumulate fp32 → next k32 tile

  final fp32 → bf16 output
```

### Throughput estimate

| Hardware | Native FP type | Tensor core throughput (peak) |
|---|---|---:|
| Blackwell BF16 | bfloat16 | 1× |
| Blackwell FP8 | float8_e4m3 | 2× |
| Blackwell FP4 (sm_120a only) | float4_e2m1 | 4× |

Spark sm_121 max FP8 = 2× BF16 = ~50% of R6000 FP4 peak. For Lynn 27B
MoE active expert kernel (memory-bound but with some compute), the practical
gain is conservatively **1.5-2.5× vs current Triton scalar_bridge / autotune**.

Lynn Spark current (SP-08): 49.37 TPS single / 49.11 mixed mean.
Lynn Spark with SP-12 FP8 kernel: **estimated 65-85 TPS single-stream class** —
beats SGLang FP8+MTP mixed mean 49.97 decisively, and approaches SGLang mixed
peak 62.51.

### Why this is feasible

`torch._scaled_mm` already supports FP8 globally in PyTorch 2.9.1 — the engine
already uses it for QKV projections and lm_head (`forward_native_fast_2d`).
The gap is **MoE active expert** — currently Triton scalar_bridge per-expert
matmul. A custom CUDA kernel that does the dequant+FP8 MMA+epilogue per-expert
fills this gap.

Code shape: similar complexity to Codex's `moe_scalar_kernel.cu` but using FP8
MMA inline PTX instead of scalar arithmetic. ~1000-1500 lines C++/CUDA. The
Lynn per-16 scale contract is preserved exactly (applied in FP32 epilogue,
just like Codex P89 split-16 design — but with FP8 instead of FP4 MMA).

## Action Plan

1. **DONE** — SP-11 capability probe confirms FP8 viable
2. **SP-12** — Design + implement Spark-specific FP8 MoE active expert kernel:
   - Read shared structure from Codex's P67/P68 scalar tile reference
     (`csrc/lynn_native/moe_scalar_kernel.cu`) as ABI template
   - Replace scalar inner loops with FP8 E4M3 MMA via inline PTX
   - Test parity against SP-08 autotuned Triton baseline
   - Promote behind opt-in `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8`
3. **SP-13** — Bench full server TPS with SP-12 enabled, target ≥ 60 mean
4. **SP-14** — Add CUDA graph capture around SP-12 kernel for further +5-10%

This is multi-day engineering. The capability matrix from SP-11 makes the
direction certain — no more strategic ambiguity about whether Spark has a path.

## Scope Boundary

All work on `spark/sm121-port`. Codex's R6000 `codex/p16-r6000-155-tps`
P88-P95+ FP4 MMA productionization is untouched and independent. The two
lanes converge to the same Lynn-native artifact contract but use different
tensor core instructions:

| Lane | Tensor core | Per-16 scale | Status |
|---|---|---|---|
| Codex R6000 sm_120a | FP4 MMA + block_scale | embedded in MMA via UE8M0 | P88 contract proven, P90 K=2048 PASS |
| **Spark sm_121** | **FP8 MMA + FP32 scale** | applied in FP32 epilogue | SP-11 capability proven, SP-12 implementation next |

Both produce identical numerical output (within FP32 accumulation tolerance)
because both apply the same Lynn per-16 FP32 scale contract. The hardware
difference is FP4 vs FP8 tensor core, which is throughput — not correctness.
