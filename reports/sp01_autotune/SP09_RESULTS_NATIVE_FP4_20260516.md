# SP-09: Lynn-Native FP4 CSRC on Spark sm_121 — Build PASS, Parity PASS, Perf NO

Date: 2026-05-16
Branch: `spark/sm121-port`

## TL;DR

**Lynn-native CUDA FP4 extension successfully builds + runs on Spark sm_121
with `-arch=sm_121a`, producing math-equivalent output (cosine 0.9999998+
across 6 representative MoE layers).** However the scalar tile reference
kernel currently in production `csrc/` is 0.732× the speed of Spark's
SP-01.5/SP-08 autotuned Triton baseline, so wiring it as a runtime backend
would regress TPS. The infrastructure is ready and waiting for Codex's P88+
FP4 MMA fused kernel to land in production csrc.

## What Was Validated

### ✅ csrc compile on Spark sm_121

```bash
LYNN_NATIVE_CUDA_ARCH=sm_121a \
  python3 benchmarks/sp09_native_csrc_smoke.py

[sp09] build OK in 32.2s
[sp09] add_one(zeros(8)) = [1.0]*8  ok=True
[sp09] active_moe_grouped_per16_contract  shape/layout guard PASS, "not implemented yet"
```

The same `sm_120a` → `sm_121a` substitution that works for the Spark Triton
JIT path also works for the Lynn native CUDA extension. This confirms
Codex's P81 sm_120a policy generalizes to sm_121 with one env-var change.

### ✅ Numerical parity on real Lynn 27B weights

Probed 6 MoE layers (2, 8, 14, 20, 28, 36) on
`lynn-27b-variable-recovery-step5000-nvfp4-final` with real router output:

| Layer | Cosine vs Triton | rel_l2 vs Triton |
|---|---:|---:|
| 2 | 0.9999999 | 5.40e-06 |
| 8 | 1.0000000 | 6.07e-06 |
| 14 | 1.0000000 | 4.21e-06 |
| 20 | 1.0000001 | 5.13e-06 |
| 28 | 1.0000000 | 4.93e-06 |
| 36 | 0.9999998 | 5.85e-06 |

All layers pass the strict P37 cosine ≥ 0.9999 promotion gate. The native
scalar tile reference produces **math-equivalent** output to my SP-01.5/SP-08
autotuned Triton baseline. Drift is at FP4 quantization noise floor.

### ❌ Perf does NOT beat SP-01.5/SP-08 Triton on Spark

| Layer | Triton (SP-01.5/SP-08 autotuned) | Native scalar tile (tile_inter=2, tile_hidden=2) | Speedup |
|---|---:|---:|---:|
| 2 | 0.1203 ms | 0.1582 ms | 0.760× |
| 8 | 0.1211 ms | 0.1762 ms | 0.687× |
| 14 | 0.1203 ms | 0.1614 ms | 0.745× |
| 20 | 0.1212 ms | 0.1733 ms | 0.699× |
| 28 | 0.1204 ms | 0.1602 ms | 0.752× |
| 36 | 0.1204 ms | 0.1614 ms | 0.746× |

**Mean speedup: 0.732× (native is 27% slower).**

## Why the Gap

Codex's R6000 P68 report measured `native tile vs Triton fast_decode` at
1.108× (native won). The baselines differ:

- **R6000 baseline**: `nvfp4_grouped_gate_up_silu_fast_decode` (P63 Triton kernel)
  vs `nvfp4_grouped_down_weighted_sum` (vanilla Triton).
- **Spark baseline (SP-01.5/SP-08)**: same kernels but `@triton.autotune`
  wrapped with 24+ candidates each, sm_121 picks better launch params.

Cumulative SP-01..SP-08 gave +13.9% on the Triton path. The Codex 1.108×
native win disappears when the Triton baseline is already 1.139× ahead of
the original — 1.108 / 1.139 = 0.973× expected, observed 0.732× (an
additional 25% gap from sm_121 vs sm_120 Triton codegen specifics).

This is **not** a problem with the native kernel — it's a problem with the
comparison: scalar CUDA reference math against a heavily autotuned Triton
kernel of the same algorithm. The native path's real value is its identity
as a **fused tensor-core kernel scaffold**, which P67/P68 explicitly is not.

## What This Unblocks

When Codex's main line publishes:

1. **P88+ FP4 MMA fused kernel** wired into `csrc/lynn_native/bindings.cpp` —
   the path that uses `cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS` (or
   sm_121 equivalent) tensor core instructions.
2. **Per-16 FP32 scale → e8m0 conversion** or native FP32 scale epilogue
   (P88 doc lists this as the remaining blocker after register layout work).

I can immediately `LYNN_NATIVE_CUDA_ARCH=sm_121a` rebuild on Spark — the
build/load infrastructure (this commit) is already verified end-to-end.

Expected at that point: native FP4 tensor core MMA gives 2-3× over scalar
ops (Blackwell native FP4 throughput is 2× FP8 = 4× FP16). Combined with my
SP-01.5/SP-08 autotune already on the dispatch wrapper, Lynn Spark TPS
could reach **80-130 single-stream** — beats SGLang FP8+MTP mixed mean and
approaches llama.cpp Nemotron-Nano 205 TPS class **if** the model size
penalty (Lynn 27B vs Nemotron 8B) is bridged by tensor core math.

## Files Added (commit-ready)

```
csrc/lynn_native/bindings.cpp                                  ← from R6000 P65
csrc/lynn_native/moe_scalar_kernel.cu                          ← from R6000 P65-P68
csrc/lynn_native/smoke_kernel.cu                               ← from R6000 P27
engine/native_cuda.py                                          ← from R6000 P81 arch policy
engine/nvfp4_layout.py                                         ← from R6000 P59 dual route
benchmarks/sp09_native_csrc_smoke.py                           ← build smoke
benchmarks/sp09_native_active_moe_microbench.py                ← full microbench (Spark autotuned baseline)
benchmarks/p68_grouped_per16_active_tile_reference_probe.py    ← Codex P68 (uncalled, R6000-specific)
benchmarks/p88_sm120a_real_gateup_tile_contract.py             ← Codex P88 (FP4 MMA probe, sm_121a test not yet run)
reports/sp01_autotune/sp09_native_microbench_1424.json         ← microbench data
```

## Decision

**Do NOT wire `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16` into Spark
production dispatch.** Current scalar reference would regress TPS by 27%.

Keep the build infrastructure ready (this commit). Monitor Codex main line.
When P89+ adds the FP4 MMA fused kernel to `csrc/lynn_native/`, immediately
re-pull + rebuild on Spark. The end-to-end build/parity infrastructure is
proven and waiting.

## Scope Boundary

All work on `spark/sm121-port`. Codex's R6000 main line P78-P88 work is
untouched and unconstrained by this Spark experiment.
