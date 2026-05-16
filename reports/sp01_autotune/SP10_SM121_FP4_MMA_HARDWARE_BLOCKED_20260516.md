# SP-10: sm_121 Hardware Lacks FP4 Block-Scale MMA — Codex P78-P90 Path Spark-Incompatible

Date: 2026-05-16
Branch: `spark/sm121-port`
Finding: **HARDWARE LIMITATION**

## TL;DR

Spark GB10 sm_121 PTXAS **rejects the FP4 block-scaled MMA instruction**
(`mma.kind::mxf8f6f4.block_scale.scale_vec::1X`) at compile time, even with
`-arch=sm_121a`. This is the exact instruction Codex's P78-P90 FP4 MMA path
on R6000 sm_120a depends on. Codex's production FP4 MMA kernel **cannot be
ported to Spark by simple arch-flag substitution**.

Spark sm_121 must continue on the Triton-autotuned path (SP-01..SP-08,
+13.9% over baseline) until either:
- NVIDIA exposes an sm_121a feature target with FP4 block-scale support, or
- A different sm_121-compatible MMA instruction is identified, or
- We accept that single-stream FP4 tensor core perf is R6000-only and
  position Spark on different value axes (long ctx, multi-service, stddev).

## Evidence

P89 split-16 scale tile contract probe pulled from R6000 main line
(`benchmarks/p89_sm120a_per16_scale_tile_contract.py`), deployed to Spark
and built inside `lmsysorg/sglang:dev-cu13` docker with:

```bash
LYNN_NATIVE_CUDA_ARCH=sm_121a
```

NVCC accepts the flag and compiles `bindings.cpp`. But PTXAS rejects when
linking the FP4 MMA kernel:

```text
[2/3] /usr/local/cuda/bin/nvcc ... -arch=sm_121a ... p89_kernel.cu

ptxas /tmp/tmpxft_*-7_p89_kernel.compute_121.ptx, line 230;
  error : Instruction 'mma with block scale' not supported on .target 'sm_121'
  error : Feature '.kind::mxf8f6f4' not supported on .target 'sm_121'
  error : Feature '.block_scale' not supported on .target 'sm_121'
  error : Feature '.scale_vec::1X' not supported on .target 'sm_121'

ptxas fatal : Ptx assembly aborted due to errors
```

Note the target reported by ptxas: `sm_121` (the `a` suffix was either
silently dropped by nvcc because sm_121a is undefined, or the `a` does not
unlock block-scale MMA on this part).

## Hardware Feature Set Reality

| Capability | R6000 sm_120 / sm_120a | Spark GB10 sm_121 |
|---|---|---|
| Workstation/datacenter binding | Workstation (RTX PRO 6000) | Grace-Blackwell SoC (DGX Spark) |
| Compute capability | 12.0 | 12.1 |
| `sm_120a` feature target | Documented, FP4 MMA + block_scale | N/A |
| `sm_121a` feature target | N/A | Either undefined or strict subset of sm_120a |
| `mma.kind::mxf8f6f4.block_scale.scale_vec::1X` | ✅ | ❌ PTXAS rejects |
| Basic FP4 storage / Triton FP4 kernels | ✅ | ✅ |
| Lynn scalar tile reference (P67/P68 csrc) | ✅ | ✅ (SP-09 PASS) |

GB10 is a Blackwell-family chip but its **ISA feature set is a strict subset
of the consumer/workstation Blackwell sm_120a parts**. The MX-block-scaled
FP4 tensor core instruction sequence is NOT present.

This explains:

1. Why SP-09 build smoke (smoke kernel + P65 ABI guard) succeeded — those
   kernels don't use block-scale FP4 MMA.
2. Why P89 split-16 scale probe builds bindings.cpp fine but fails at the
   MMA kernel link step.
3. Why Spark Triton FP4 path is software-emulated through `torch._scaled_mm`
   FP8 codepath + per-16 dequant, not native tensor core MMA.

## Strategic Implication

Codex's R6000 main line P78-P90 path is **structurally sm_120-only** under
current NVIDIA tooling (CUDA 13.0). Spark sm_121 cannot share the same
production kernel binary.

Two productive responses:

### A. Decouple Spark engine and R6000 engine value props

| Property | R6000 sm_120 | Spark sm_121 |
|---|---|---|
| Native FP4 MMA throughput | YES (target 100+ TPS via P90+) | NO |
| Triton-autotuned MoE | yes | yes (current production +13.9%) |
| Long-context linear-attn scaling | yes | YES (6.77× SGLang @ 16k) |
| Multi-service unified-mem | limited 96G | YES (119G) |
| Single-stream peak TPS | R6000 wins | accept second tier |
| Single-stream mixed mean | R6000 wins | within 2% of SGLang FP8+MTP |
| Steadiness (stddev) | unknown | 0.17 = 37× SGLang |

Spark's value is NOT single-stream peak. Spark is the **long-ctx + multi-service +
deterministic latency** platform. R6000 is the **single-stream peak TPS** platform.

### B. Re-investigate sm_121 instruction options

Worth probing whether sm_121 has any FP4/FP6/FP8 MMA primitives at all,
even without block_scale:

- `mma.m16n8k32.row.col.kind::f8f6f4` (without `.block_scale`) — may work
- `mma.m16n8k64.row.col.bf16.fp4.fp4.bf16` — newer Blackwell variants
- `mma.kind::mxf4` (without f8f6) — only FP4 path

If any of these work on sm_121, we can build a custom Lynn kernel that
applies per-16 FP32 scales in FP32 epilogue (similar to P89 split-16 but
without block_scale instruction).

This requires deeper PTX inspection. Not done in this session.

## Action Items

1. **DONE** — restart production server with SP-01..SP-08 autotune
   (currently best Spark Lynn 27B NVFP4 config: 49.37 single / 49.11 mixed).
2. **DOC** this finding so future sessions know the constraint upfront.
3. **PUNT** Codex main P78-P90 work — keep watching for non-block_scale FP4
   MMA variants but don't pull anything that uses `BLOCKSCALED::SM120` into
   Spark production csrc.
4. **PIVOT** Spark engine roadmap toward long-ctx + multi-service + stddev
   wins rather than single-stream peak.

## Files

- `benchmarks/p89_sm120a_per16_scale_tile_contract.py` — Codex probe pulled
  (kept committed even though it fails on Spark — captures the exact ptxas
  error message and shows future contributors the issue).

## Scope

`spark/sm121-port` only. Codex main `codex/p16-r6000-155-tps` is the R6000
FP4 MMA productionization line and continues uninterrupted.
