# Stage 6 R6000 Grouped-MoE FP4-MMA POC Contract

Date: 2026-06-04

Verdict: **contract only; no Lynn kernel, runtime default, or speed claim is
promoted by this document.**

The R6000 census is now banked:
[r6000_fp4_mma_census_20260604_164457](r6000_fp4_mma_census_20260604_164457/summary.md)
reports `PASS_R6000_FP4_MMA_BRINGUP` on AutoDL host 727 / RTX PRO 6000 96GB,
CUDA capability `[12, 0]`, 880 GiB data workspace, vLLM source candidates 200,
and P76/P79/P85/P87/P103 all passing.

This POC starts **after** that census. It is not another Spark decode-speed
experiment. Spark owns 35B serving/memory/MTP/compiled-loop ROI and RC quality;
R6000 owns native FP4-MMA/CUTLASS/CuTe grouped-kernel evidence.

## Non-Negotiable Boundary

The POC target is a Lynn NVFP4 grouped active-MoE kernel that can eventually use
Blackwell FP4 tensor cores. It must not report success merely because a generic
sm_120a MMA probe compiles.

Two layouts must be kept distinct:

| Layout | Meaning |
|---|---|
| Lynn resident NVFP4 | packed E2M1 weights with per-16 E4M3-style scales currently consumed by Lynn kernels |
| Blackwell/CUTLASS block-scaled FP4 | tensor-core-facing FP4 fragments and scale layout accepted by CuTe/CUTLASS block-scaled MMA paths |

The first POC must explicitly prove the bridge between those layouts: either a
zero-copy reinterpretation with exact scale semantics, or an explicit repack
step with measured memory/time cost. Silent reinterpretation is forbidden.

## Inputs

The active MoE dimensions remain:

```text
H = 2048
I = 512
E = 256
top_k = 8
```

The existing Lynn tensors are:

| Tensor | Current contract |
|---|---|
| `mlp.experts._gate_up_packed` | `[E, 2I, H/2]` packed E2M1 |
| `mlp.experts._gate_up_scale` | `[E, 2I, H/16]` scale |
| `mlp.experts._gate_up_global_scale` | scalar/global scale |
| `mlp.experts._down_packed` | `[E, H, I/2]` packed E2M1 |
| `mlp.experts._down_scale` | `[E, H, I/16]` scale |
| `mlp.experts._down_global_scale` | scalar/global scale |
| Router output | `expert_ids [M, top_k]`, `routing_weights [M, top_k]` |

## Reference Stack

Each POC run must compare against:

- BF16 active expert shadow reference for numeric sanity.
- Current packed no-shadow reference (`p2e_hybrid` / P3-A active-MoE probe).
- R6000 census artifact above for machine/toolchain/public-kernel readiness.
- CUTLASS/CuTe headers on the R6000 host.
- Public Marlin/Machete/vLLM source census as implementation blueprint, not a
  copied kernel.

## POC Ladder

| Gate | Scope | What can be banked |
|---|---|---|
| R5-A layout bridge | Single expert, gate/up only, synthetic M=1/16/64 | scale/layout bridge exactness, byte counts, repack cost if any |
| R5-B gate/up FP4-MMA | One layer, routed active experts, M=16/64 | gate/up numeric parity and speed vs BF16 active/P3-A, no full MoE claim |
| R5-C down FP4-MMA | One layer, consumes banked active scratch | down numeric parity, weighted-sum correctness, scratch/byte counts |
| R5-D grouped active MoE | gate/up + down one layer | active-MoE numeric parity, speed, no-shadow proof, no default promotion |
| R5-E selected-prefill smoke | layers 0-3 selected stack | residual-stack parity and memory profile only |

The first implementation target is **R5-A**, not a full fused MoE kernel.

## Required Evidence For Any PASS

Every artifact must include:

- exact git commit and working tree status;
- exact host/GPU/driver/CUDA/PyTorch/CUTLASS/vLLM-source inventory;
- exact launch command and environment;
- `nvidia-smi` before/after;
- input tensor shapes and layout/scale interpretation;
- numeric parity: cosine, max_abs/rel_l2, argmax or first-divergence where
  applicable;
- byte counts for packed weights, scales, repacked buffers, active scratch, and
  BF16-shadow equivalent;
- timing with warmup/repeats and a baseline;
- promotion boundary fields:
  - `banked_layout_bridge`
  - `banked_grouped_moe_fp4_mma_poc`
  - `banked_kernel_speed`
  - `banked_default_promotion`

For R5-A, only `banked_layout_bridge=true` may become true. For all gates before
server/RC, `banked_default_promotion=false`.

## Forbidden False Positives

Do not bank a POC if it:

- uses BF16 active expert resident shadows after the candidate starts;
- calls Spark-only evidence an FP4-MMA result;
- omits the per-16-to-block-scaled layout bridge decision;
- hides a full-weight BF16 dequant/repack inside the timing window;
- reports a two-stage scratch path as a final fused single-kernel win;
- widens RC after numeric drift is already observed;
- copies non-compatible code rather than using public kernels as design
  references.

## Local Static Gate

```bash
python3 scripts/test_stage6_r6000_grouped_moe_poc_contract_static.py
```

This static gate does not prove a kernel. It prevents the repo from starting an
R6000 FP4-MMA implementation before the layout bridge, evidence fields, and
promotion boundaries are explicit.
