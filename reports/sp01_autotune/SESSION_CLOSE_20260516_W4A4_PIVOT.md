# Session Close 2026-05-16 — W4A4-First Strategic Pivot

## Decision (this session)

After SP-01 through SP-12-F empirical work on Spark sm_121 + cross-reference
with Codex R6000 P98-P100, **the long-term Lynn primary path is W4A4-first**,
implemented as two parallel lanes:

| Lane | Owner | Scope |
|---|---|---|
| **A100 artifact line** | training | MTP/NEXTN head training + activation-aware W4A4 retrain → new model artifact |
| **R6000 runtime line** | Codex | stable production baseline (BF16 act + FP4 weight) + native FP4 runtime gate ready to consume new W4A4 package when it lands |
| **Spark sm_121** | this Spark branch | stable SP-08 production (49 TPS class), long-ctx + multi-service positioning, native FP8 kernel ready for W4A4 artifact arrival |

## Why W4A4 Becomes Primary

This session proved by **empirical convergence** from both engines:

- **Spark SP-12-F result (this branch)**: inline BF16 → E2M1 activation
  quantization produces garbage decode output ("一句话/" instead of
  attention explanation) and -40% TPS regression — kernel-level numerical
  contract (SP-12-D max_abs 7.15e-07 in microbench) does NOT survive in
  the full-model decode loop because Lynn 27B was trained with BF16
  activations, not E2M1.
- **Codex R6000 P98 result (main line)**: same class of failure — full
  generate gate `new_ids_all_match=false` with the split16 FP4 path because
  activation re-quantization changes greedy decode semantics; graph capture
  also breaks due to non-graph-safe PyTorch quantization ops.
- **Codex R6000 P100 result**: even weight-only optimization (native down
  kernel with BF16 activation preserved) fails strict greedy parity at
  +1.049x speed because FP32 accumulation order differs from Triton.

**Conclusion**: any path that targets the full Blackwell FP4/FP8 tensor
core throughput requires the model to be trained with low-precision
activations from the start (W4A4 model). Post-hoc activation quantization
on a BF16-trained model breaks output coherence, regardless of whether the
kernel math is bit-correct against an isolated reference.

The two viable production routes:

- **Route A — BF16 activation lock-in** (Codex R6000 P99 default): Keep
  BF16 activation; optimize weight side + dispatch only. Limited by
  hardware tensor core requiring matching operand types.
- **Route B — W4A4 retrain**: Train the model with E2M1 activation. Then
  the kernel-level numerical contract aligns with model semantics. Full
  Blackwell FP4/FP8 tensor core throughput becomes available.

The user has chosen Route B as the long-term primary direction.

## What This Session Shipped on `spark/sm121-port`

20+ commits, complete history at <https://github.com/MerkyorLynn/lynn-engine/tree/spark/sm121-port>.

### Production stable (default-on can ship today)

| Layer | Commit / Status |
|---|---|
| **SP-01..SP-08 Triton autotune** | `LYNN_SP_TRITON_AUTOTUNE=1` env. Cumulative +13.9% over original baseline (43.33 → 49.37 single mean, 43.26 → 49.11 mixed). Beats SGLang FP8+MTP on single mean (+12.9%), single peak (+4.4%), and stddev (37× steadier). Ties on mixed mean within sample noise. |

### Research-stage infrastructure (waits for W4A4 artifact)

| Layer | Status |
|---|---|
| **csrc/lynn_native/** on sm_121a | Codex P65 ABI builds clean on Spark with `-arch=sm_121a`; smoke + P65 contract guard pass. (Commit `cc95f47`) |
| **SP-12 FP8 kernel series** | A-D numerical contract proven; E-v2 2.67× speedup over D-v1 via vectorized loads + register-LUT; F integration wires into server. ALL OF THIS IS READY-WHEN-W4A4-LANDS but currently breaks decode quality due to BF16-activation training. (Commits `4cb6cde`, `65cb2d5`, `116b656`, `a050ca2`, `6158cb9`, `2989f20`, `0413e25`, `7db7033`, `044f6fc`) |
| **engine/spark_fp8.py** | Self-contained JIT-built FP8 MMA dispatch module. Activated via `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8` env. Currently fails greedy contract with BF16-trained Lynn 27B. (Commit `2989f20`) |

### Empirical findings (hardware truths)

| Finding | Commit | Implication |
|---|---|---|
| sm_121 PTXAS rejects `mma.kind::mxf8f6f4.block_scale.scale_vec::1X` | `e30b070` | Codex R6000 P78-P95 sm_120a FP4 MMA path **physically inaccessible to Spark** |
| sm_121 ISA capability matrix probed empirically | `49e5ccd` | FP8 E4M3+E5M2 OPEN, all FP4/FP6/kind:: BLOCKED; Spark gets the FP8 mirror of Codex's FP4 path |
| SP-12-F inline activation re-quant breaks production | this doc | Independently confirms Codex P99 conclusion on Lynn 27B |

## Spark sm_121 Stable Position (what serves now)

```bash
# Production launch — currently best Spark sm_121 Lynn 27B config:
LYNN_SP_TRITON_AUTOTUNE=1 \
  bash scripts/spark/run_27b_nvfp4_server.sh

# Numbers (last bench, this session):
#   single mean 49.37  peak 49.38  beats SGLang single by +12.9%
#   mixed mean  49.11  peak 49.39  ties SGLang within 1σ
#   stddev 0.17        37× steadier than SGLang FP8+MTP
```

DO NOT enable `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8` against the
current Lynn 27B BF16-trained artifact — it will produce garbled output.
Wait for the W4A4 artifact from the A100 lane.

## When A100 W4A4 Artifact Lands

The infrastructure on this branch becomes immediately useful:

1. Swap in the new W4A4 artifact path (replace
   `/models/lynn-27b-variable-recovery-step5000-nvfp4-final` in
   `scripts/spark/run_27b_nvfp4_server.sh`)
2. Run quality smoke (the same 6-prompt + V8/V9 gates)
3. Enable `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8` — engine/spark_fp8.py
   will JIT-build the kernel on first call
4. Re-bench TPS. Microbench projection: 98.7 us per layer = 3.95 ms per
   step for active MoE → estimated 60-75 TPS depending on what other
   path components consume of the 20 ms decode budget.

If the W4A4 retrain also includes MTP/NEXTN heads, Spark can additionally
push toward SGLang's mixed-peak class (60+ TPS) via speculative decoding
on top of the FP8 MMA path.

## File Inventory Added This Session

```
# Production
engine/spark_fp8.py                                            ← FP8 dispatch module
engine/moe_packed_nvfp4.py                                     ← +spark_fp8 dispatch branch
scripts/spark/run_27b_nvfp4_server.sh                          ← env passthrough for backend + arch + graph flags

# Research probes (each one ships independently testable code + reproducible bench)
benchmarks/sp01_tps_bench.py                                   ← TPS harness (SGLang-matched)
benchmarks/sp01_sm121_autotune_microbench.py                   ← MoE kernel microbench
benchmarks/sp09_native_csrc_smoke.py                           ← Lynn-native csrc build smoke
benchmarks/sp09_native_active_moe_microbench.py                ← scalar tile parity vs Triton
benchmarks/sp11_sm121_mma_capability_probe.py                  ← ISA capability map
benchmarks/sp12a_sm121_fp8_e2m1_tile_probe.py                  ← FP8+LUT bit-exact tile
benchmarks/sp12b_sm121_fp8_per16_scale_probe.py                ← K=2048 split-16
benchmarks/sp12c_sm121_fp8_8row_tile_probe.py                  ← 8-row production shape
benchmarks/sp12d_spark_fp8_active_moe_probe.py                 ← full active-MoE chain
benchmarks/sp12e_v2_spark_fp8_vectorized_probe.py              ← 2.67x optimization

# Imported from Codex R6000 main as reference probes (build-test only on Spark)
benchmarks/p67_grouped_per16_down_tile_probe.py                ← Codex P67
benchmarks/p68_grouped_per16_active_tile_reference_probe.py    ← Codex P68
benchmarks/p88_sm120a_real_gateup_tile_contract.py             ← Codex P88 (sm_120a only)
benchmarks/p89_sm120a_per16_scale_tile_contract.py             ← Codex P89 (sm_120a only)

# Comprehensive docs
docs/SPARK_SM121_PORT_NOTES.md                                 ← top-level Spark branch state
docs/LYNN_ENGINE_SP01_SM121_TRITON_AUTOTUNE_20260516.md
docs/LYNN_ENGINE_SP02_NGRAM_SPEC_DECODE_PLAN_20260516.md
reports/sp01_autotune/SP01_RESULTS_20260516_1112.md
reports/sp01_autotune/SP01_5_RESULTS_20260516_1139.md
reports/sp01_autotune/SPARK_VS_SGLANG_FINAL_20260516.md
reports/sp01_autotune/SP09_RESULTS_NATIVE_FP4_20260516.md
reports/sp01_autotune/SP10_SM121_FP4_MMA_HARDWARE_BLOCKED_20260516.md
reports/sp01_autotune/SP11_SM121_MMA_CAPABILITY_MATRIX_20260516.md
reports/sp01_autotune/SP12_SPARK_FP8_ACTIVE_MOE_RESULTS_20260516.md
reports/sp01_autotune/SESSION_CLOSE_20260516_W4A4_PIVOT.md     ← this file

# csrc (imported from Codex P65 + P67-P68 reference for sm_121a build)
csrc/lynn_native/bindings.cpp
csrc/lynn_native/moe_scalar_kernel.cu
csrc/lynn_native/smoke_kernel.cu

# Engine helpers
engine/native_cuda.py                                          ← Codex P81 arch policy (sm_121a auto)
engine/nvfp4_layout.py                                         ← Codex P59 dual-route layout detector
```

## New Conversation Starter — Brief

Open a new session with:

> Spark sm_121 work resumed. Current branch `spark/sm121-port` ships
> `LYNN_SP_TRITON_AUTOTUNE=1` as production default (49 TPS class, beats
> SGLang single + 37x steadier). SP-12 FP8 kernel infrastructure on
> sm_121a is research-ready but blocked on W4A4 artifact from the A100
> training lane (current BF16-trained Lynn 27B produces garbled output
> when activation is re-quantized to E2M1 for FP8 MMA, per Codex P98-P100
> + this branch's SP-12-F). Read
> `reports/sp01_autotune/SESSION_CLOSE_20260516_W4A4_PIVOT.md` for full
> context. Next milestone: when W4A4 artifact lands on Spark, enable
> `LYNN_NATIVE_ACTIVE_MOE_BACKEND=spark_fp8` and re-bench expected
> 60-75 TPS via FP8 MMA + Lynn per-16 scale epilogue.
