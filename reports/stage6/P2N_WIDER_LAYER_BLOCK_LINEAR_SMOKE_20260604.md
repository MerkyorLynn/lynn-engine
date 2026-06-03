# Stage 6 Phase 2-N — wider-layer block-linear smoke

Date: 2026-06-04

Verdict: **PASS on wider selected-layer coverage; not yet server-promoted.**

P2-N expands P2-M from layers 0-3 to layers 0-7 while keeping the same combined
opt-in path:

- MoE: `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`
- Linear attention: `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`

This covers six linear-attention layers plus two full-attention layers:

```text
0 linear_attention
1 linear_attention
2 linear_attention
3 full_attention
4 linear_attention
5 linear_attention
6 linear_attention
7 full_attention
```

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 scripts/spark_stage6_p2m_selected_layer_block_linear_smoke.py \
    --layers 0-7 \
    --seq-lens 16,64 \
    --warmup 0 \
    --iters 1 \
    --repeats 1 \
    --json-out /home/merkyor/lynn-engine/reports/stage6/p2n_wider_layer_block_linear_20260604_030926/result.json
```

Artifacts:

- `reports/stage6/p2n_wider_layer_block_linear_20260604_030926/result.json`
- `reports/stage6/p2n_wider_layer_block_linear_20260604_030926/run.log`

## Speed

| Layers | Seq len | BF16 reference | P2E MoE only | P2N block-linear + P2E | P2N vs BF16 | P2N vs P2E |
|---|---:|---:|---:|---:|---:|---:|
| 0-7 | 16 | 115.56 ms | 88.76 ms | 74.98 ms | 1.541x | 1.184x |
| 0-7 | 64 | 185.03 ms | 174.98 ms | 162.25 ms | 1.140x | 1.078x |

## Numeric

All P2E/P2N comparisons pass with cosine > 0.999 and argmax match.

| Metric | Value |
|---|---:|
| Min cosine | 0.999894142 |
| Max cosine | 0.999948277 |
| Max rel_l2 | 0.014593820 |
| Argmax | match |

The wider 8-layer residual stack accumulates more floating-order drift than P2-M,
but it remains inside the current selected-layer smoke gate.

## Memory

| Item | Value |
|---|---:|
| Deleted BF16 active expert shadow | 12.000 GiB |
| Packed active expert bytes | 4.500 GiB |
| Resident after deleting BF16 active shadows | 5.042 GiB |
| P2E peak T64 | 5.129 GiB |
| P2N peak T64 | 5.116 GiB |

## Decision

Bank P2-N as wider selected-layer coverage:

- numeric pass;
- no active BF16 expert shadow;
- speed pass vs both BF16 and P2E-only;
- default remains unchanged without opt-in flags.

Next gate: RC/server smoke on the combined opt-in path. This wider-layer smoke is
not a default-path promotion.
