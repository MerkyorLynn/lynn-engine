# Stage 6 Phase 2-M — selected-layer smoke with block linear-attn + P2E MoE

Date: 2026-06-04

Verdict: **PASS on selected-layer full prefill smoke; not yet server-promoted.**

P2-M combines the two opt-in paths banked earlier:

- MoE: `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`
- Linear attention: `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`

This smoke runs selected real transformer layers through `_prefill_layer`, including
RMSNorm, linear/full attention cache population, residuals, and MoE FFN. The active
expert BF16 shadows are deleted before the packed paths run.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2m_selected_layer_block_linear_smoke.py \
    --layers 0-3 \
    --seq-lens 16,64 \
    --warmup 0 \
    --iters 1 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2m_selected_layer_block_linear_20260604_030005/result.json
```

Artifacts:

- `reports/stage6/p2m_selected_layer_block_linear_20260604_030005/result.json`
- `reports/stage6/p2m_selected_layer_block_linear_20260604_030005/run.log`

## Speed

| Layers | Seq len | BF16 reference | P2E MoE only | P2M block-linear + P2E | P2M vs BF16 | P2M vs P2E |
|---|---:|---:|---:|---:|---:|---:|
| 0-3 | 16 | 57.40 ms | 45.11 ms | 39.12 ms | 1.467x | 1.153x |
| 0-3 | 64 | 93.59 ms | 90.59 ms | 84.26 ms | 1.111x | 1.075x |

## Numeric

All P2E/P2M comparisons pass with cosine > 0.999 and argmax match.

| Metric | Value |
|---|---:|
| Min cosine | 0.999949185 |
| Max cosine | 0.999983027 |
| Max rel_l2 | 0.010081087 |
| Argmax | match |

The lower cosine versus P2-L is expected: this is a multi-layer residual path
with MoE and linear attention both switched to packed/native opt-in modes.

## Memory

| Item | Value |
|---|---:|
| Deleted BF16 active expert shadow | 6.000 GiB |
| Packed active expert bytes | 2.250 GiB |
| Resident after deleting BF16 active shadows | 2.525 GiB |
| P2E peak T64 | 2.612 GiB |
| P2M peak T64 | 2.599 GiB |

## Decision

Bank P2-M as selected-layer full-prefill pass:

- numeric pass;
- no active BF16 expert shadow;
- speed pass vs both BF16 and P2E-only;
- default remains unchanged without opt-in flags.

Next gate: expand coverage beyond layers 0-3 (P2-N) and then run RC-quality /
server-promotion smoke before making any default-path decision.
