# Stage 6 Phase 2-KB — gated-delta block-kernel PoC

Date: 2026-06-04

Verdict: **PASS as a core-kernel PoC; not yet promoted to serving.**

P2-J identified `chunk_gated_delta_with_state` as the linear-attention prefill
wall. P2-KA then proved that the existing single-token Triton decode recurrent
kernel is numerically reusable, but a host loop over tokens regresses badly
because it launches once per token. P2-KB moves that token loop inside one Triton
launch.

This is deliberately scoped to the gated-delta core only:

- no projection fusion;
- no conv fusion;
- no g/beta fusion;
- no packed-weight / zero-shadow integration;
- B=1 fixed for the first kernel.

## Implementation

Added `recurrent_gated_delta_block_gqa()` in `triton_kernels/gated_delta.py`.

The kernel keeps the existing decode recurrent math:

```text
q/k l2norm
q *= 1 / sqrt(128)
state *= exp(g_t)
kv_mem = state @ k_t
delta = (v_t - kv_mem) * beta_t
state += k_t outer delta
out_t = state @ q_t
```

The difference from P2-KA is launch granularity:

| Path | Launch shape |
|---|---|
| P2-KA host loop | one Triton recurrent launch per token |
| P2-KB block kernel | one Triton launch for the whole tested T block |

The kernel uses GQA directly: q/k stay `[1,T,16,128]`, each value head reads
`q/k[head//2]`, while v/g/beta/state/output stay on 32 value heads.

## Commands

Short run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2kb_gated_delta_block_kernel_poc.py \
    --layer 0 \
    --seq-lens 16,64,128 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2kb_gated_delta_block_kernel_20260604_024642/result.json
```

Long run:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2kb_gated_delta_block_kernel_poc.py \
    --layer 0 \
    --seq-lens 256,512 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2kb_gated_delta_block_kernel_long_20260604_024710/result.json
```

Artifacts:

- `reports/stage6/p2kb_gated_delta_block_kernel_20260604_024642/result.json`
- `reports/stage6/p2kb_gated_delta_block_kernel_20260604_024642/run.log`
- `reports/stage6/p2kb_gated_delta_block_kernel_long_20260604_024710/result.json`
- `reports/stage6/p2kb_gated_delta_block_kernel_long_20260604_024710/run.log`

## Speed

`block_vs_chunk = chunk_reference_ms / block_kernel_ms`; values above 1.0 are faster
than the current torch chunk reference.

| Seq len | Chunk reference | P2-KA host loop | P2-KB block kernel | Block vs host loop | Block vs chunk |
|---:|---:|---:|---:|---:|---:|
| 16 | 2.79 ms | 0.83 ms | 0.079 ms | 10.53x | 35.36x |
| 64 | 3.03 ms | 2.15 ms | 0.135 ms | 15.92x | 22.43x |
| 128 | 2.89 ms | 4.36 ms | 0.465 ms | 9.36x | 6.20x |
| 256 | 3.49 ms | 8.89 ms | 0.621 ms | 14.31x | 5.62x |
| 512 | 4.55 ms | 16.28 ms | 1.160 ms | 14.04x | 3.92x |

P2-KB therefore fixes the P2-KA failure mode: launch count drops from T launches
to one launch for the tested block.

## Numeric

All comparisons pass with cosine > 0.999 and argmax match.

| Run | Min cosine | Max cosine | Max rel_l2 | Argmax |
|---|---:|---:|---:|---|
| T16/T64/T128 | 0.999989555 | 1.000000000 | 0.004794770 | match |
| T256/T512 | 0.999993027 | 1.000000000 | 0.003773756 | match |

The strictest comparison is still vs `_chunk_gated_delta_with_state`; P2-KB also
matches the P2-KA host-loop oracle to near machine precision.

## Caveats

- This is a core-kernel PoC, not full `prefill_linear_attn` integration.
- The kernel computes recurrent math in sequential token order; it does not
  implement HF's triangular chunk algebra. The observed numeric agreement is
  sufficient for the current layer-level gate, but RC quality must still decide
  promotion.
- q/k l2norm is recomputed for each value-block program. This is acceptable for
  the first gate; later kernels can pre-normalize or stage q/k more efficiently.
- The first kernel is B=1 only and uses `T` as a Triton constexpr. Longer-context
  productization may need chunked launches, autotune, or a CUDA C++ version.

## Decision

Bank P2-KB as **core kernel passed**.

Next gate: P2-L should wire `recurrent_gated_delta_block_gqa()` into
`prefill_linear_attn` behind an opt-in flag, then rerun selected-layer/full
prefill smoke before any server promotion.
