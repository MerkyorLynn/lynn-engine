# Stage 6 Phase 1 — single dense projection packed-NVFP4 contract

**Date:** 2026-06-04
**Scope:** one real dense projection from the Lynn-native 35B W4A16 NVFP4 artifact.

## Target

Use `model.language_model.layers.0.linear_attn.in_proj_qkv.weight` as the first
dense projection PoC:

| property | value |
|---|---:|
| original shape | `8192 x 2048` |
| BF16 shadow bytes | 32.00 MiB |
| packed bytes | `8192 x 1024` uint8 = 8.00 MiB |
| scale bytes | `8192 x 128` FP16 = 2.00 MiB in the current 35B checkpoint |
| group size | 16 input columns per scale |
| global scale | scalar, effective scale = `scale / global_scale` |

The PoC kernel must read the packed E2M1 bytes and checkpoint-native scale
directly. It must not require the resident `.weight` BF16 shadow during the
timed packed path.

Earlier planning assumed E4M3 scale storage. The real 2026-06-04 Spark probe
found `torch.float16` scale for this Lynn-native 35B artifact, so byte-counts and
contracts use FP16 scale unless a future artifact proves otherwise.

## Numerical Contract

Reference order:

1. Unpack low nibble first, high nibble second.
2. Interpret E2M1 magnitude table as `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` with bit 3
   as sign.
3. Broadcast per-16 scale over input columns.
4. Apply `effective_scale = scale / global_scale`.
5. Accumulate in FP32.
6. Output FP32 for the PoC; integration may cast to BF16 at the caller boundary.

Gate against explicit FP32 dequant reference:

| metric | requirement |
|---|---:|
| cosine | `>= 0.99999` |
| relative L2 | `<= 2e-3` |
| argmax | match |

Also report comparison against the BF16-shadow `F.linear` path, but do not use it
as the primary correctness oracle because it includes BF16 weight rounding.

## Byte Contract

The harness must report:

- BF16 shadow bytes.
- packed weight bytes.
- scale bytes and dtype.
- global-scale bytes and dtype.
- timed packed path argument bytes.
- GPU memory after deleting all BF16/FP32 reference tensors.

The packed benchmark is valid only if it runs after deleting the BF16 reference
weights. A narrow code trace is not enough; the run must record allocated memory
below the single projection's BF16 shadow byte size during the packed benchmark.

## Harness

Run on Spark without constructing `LynnIncrementalRunner`:

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p1_dense_projection_poc.py \
    --json-out reports/stage6/p1_dense_projection_poc_$(date +%Y%m%d_%H%M%S).json
```

## Promotion Rule

Phase 1 is not banked until the run supplies all four evidence classes:

| evidence | required |
|---|---|
| byte-count | BF16 vs packed+scale reported from real checkpoint tensors |
| numeric parity | FP32 dequant oracle gate passes |
| microbench | packed Triton vs BF16 `F.linear` reported honestly |
| no hidden shadow | packed benchmark runs after deleting BF16/FP32 references |

If numeric and no-shadow pass but speed is slower, record it as a valid kernel
contract but **not** a Phase 1 performance win. Then tune kernel tiling or move
to a batched prefill kernel before promotion.
