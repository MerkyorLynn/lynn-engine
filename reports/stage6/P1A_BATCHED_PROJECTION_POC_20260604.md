# Stage 6 Phase 1-A — batched dense projection packed-NVFP4 PoC

**Date:** 2026-06-04
**Host:** Spark GB10 (`dgx-via-n5`)
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
**Projection:** `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`
**Runner:** `scripts/spark_stage6_p1a_batched_projection_poc.py`
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p1a_batched_projection_poc_20260604_001639`
**APEX safety:** APEX stayed online; `/health` returned `{"status":"ok"}` after the run and request metrics were idle.

## Verdict

**P1-A first batched bridge is NOT promoted.**

It passed the numeric and no-shadow gates, but failed the batched performance
gate. The kernel removes the Python row loop by launching over
`tokens x output-row-blocks`, but each program still computes one token row
independently, so it does not reuse activation tiles across tokens. BF16
`F.linear` amortizes the projection much better for M>1.

This is still useful evidence: the packed contract is correct for M>1, and the
next kernel shape is now clear. P1-A needs a true tiled
`BLOCK_T x BLOCK_OUT x BLOCK_K` kernel, not another row-loop wrapper.

## Numeric Gate

| batch | cosine vs FP32 dequant | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 1 | 1.000000000 | 1.387e-07 | 7.153e-07 | match |
| 4 | 1.000000000 | 1.294e-07 | 7.153e-07 | match |
| 16 | 1.000000000 | 1.422e-07 | 5.960e-07 | match |
| 64 | 1.000000000 | 3.161e-07 | 1.550e-06 | match |

Numeric gate: **PASS**.

## Microbench Gate

| batch | packed median | BF16 median | speedup | packed/token | BF16/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 159.27 us | 181.80 us | 1.141x | 159.27 us | 181.80 us |
| 4 | 599.17 us | 155.99 us | 0.260x | 149.79 us | 39.00 us |
| 16 | 2360.45 us | 158.57 us | 0.067x | 147.53 us | 9.91 us |
| 64 | 9420.65 us | 154.54 us | 0.016x | 147.20 us | 2.41 us |

Performance gate: **FAIL** for M>1.

## No-Shadow Evidence

The timed packed benchmark ran after deleting FP32/BF16 reference weights and
resetting peak memory.

| metric | value |
|---|---:|
| memory before packed bench | 0.0180 GiB |
| memory after packed bench | 0.0180 GiB |
| peak during packed bench | 0.0200 GiB |
| single projection BF16 shadow | 0.03125 GiB |

No-shadow gate: **PASS**.

## Decision

Keep `nvfp4_batched_matmul_packed` and its Spark harness as a correctness and
regression probe, but do **not** wire it into `resident_runner` or count it as a
serving performance win.

Next implementation target:

| item | requirement |
|---|---|
| kernel shape | `BLOCK_T x BLOCK_OUT x BLOCK_K` |
| activation reuse | load a token tile once and use it across output-row blocks |
| output | FP32 or BF16, reported explicitly |
| gate | same numeric/no-shadow evidence plus M=4/16/64 latency not worse than BF16 |
