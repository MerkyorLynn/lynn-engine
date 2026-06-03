# Stage 6 Phase 2 — grouped MoE packed-prefill census

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `model.language_model.layers.0.mlp.experts`  
**Runner:** `scripts/spark_stage6_p2_grouped_moe_prefill_census.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2_grouped_moe_prefill_census_20260604_003411`

## Verdict

**P2 census is banked. The next real kernel target is routed grouped MoE
prefill, not dense projection prefill and not server integration.**

The current P0.1 no-reload proof path (`stream_bf16`) is token/numeric exact,
but it dequants a whole layer's packed MoE experts into temporary wide tensors
on every prefill call. On one layer this costs **~0.49-0.51 s**, matching the
known **~20.75 s** 40-layer no-reload proof path.

The existing `smallm` grouped verifier avoids the full-layer temporary and is
memory-clean after the BF16 shadow is deleted, but it is still a Python /
selected-expert dequant verifier, not a serving path: **1.94-48.97x** faster
than `stream_bf16`, yet still only **0.082-0.418x** of resident BF16 prefill.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2_grouped_moe_prefill_census.py \
    --batches 1,4,16,64 \
    --warmup 1 --iters 3 --repeats 2 \
    --json-out reports/stage6/p2_grouped_moe_prefill_census_20260604_003411/result.json
```

## Weight Layout

| item | shape / bytes |
|---|---:|
| hidden | 2048 |
| experts / top-k | 256 / 8 |
| expert intermediate | 512 |
| BF16 grouped expert shadow | 1.500 GiB per layer |
| packed grouped expert tensors | 0.563 GiB per layer |
| BF16-to-packed ratio | 2.667x |
| memory after deleting BF16 shadow | 0.641 GiB |

The 40-layer grouped-MoE BF16 shadow is therefore the same **~60 GiB** shadow
released by Stage 6 option (b).

## Numeric Gate

`stream_bf16` is exact vs resident BF16 for this layer. `smallm` is not exact,
but remains tightly aligned and keeps argmax identical.

| batch | stream cosine | stream rel L2 | smallm cosine | smallm rel L2 | argmax |
|---:|---:|---:|---:|---:|---|
| 1 | 1.000000000 | 0.000e+00 | 0.999997667 | 2.160e-03 | match |
| 4 | 1.000000000 | 0.000e+00 | 0.999997927 | 2.039e-03 | match |
| 16 | 1.000000000 | 0.000e+00 | 0.999997493 | 2.239e-03 | match |
| 64 | 1.000000000 | 0.000e+00 | 0.999997654 | 2.166e-03 | match |

Numeric gate for census: **PASS**.

## Latency Gate

| batch | BF16 prefill | stream_bf16 | smallm verifier | stream vs BF16 | smallm vs BF16 | smallm vs stream |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.16 ms | 487.86 ms | 9.96 ms | 0.009x | 0.418x | 48.97x |
| 4 | 6.17 ms | 490.13 ms | 39.49 ms | 0.013x | 0.156x | 12.41x |
| 16 | 12.30 ms | 496.61 ms | 128.43 ms | 0.025x | 0.096x | 3.87x |
| 64 | 21.45 ms | 506.22 ms | 260.65 ms | 0.042x | 0.082x | 1.94x |

Performance gate for serving promotion: **FAIL** for both packed proof paths.

## Memory Gate

Packed proof paths ran after deleting `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj`.

| batch | stream peak | smallm peak |
|---:|---:|---:|
| 1 | 12.641 GiB | 0.699 GiB |
| 4 | 12.641 GiB | 0.699 GiB |
| 16 | 12.641 GiB | 0.700 GiB |
| 64 | 12.641 GiB | 0.700 GiB |

`stream_bf16` is correct but not a real packed-prefill kernel: its internal
full-layer dequant temporaries explain the high peak. `smallm` proves selected
expert grouping can stay memory-clean, but its selected-expert dequant +
PyTorch matmul loop is still too slow.

## Next P2 Kernel Contract

First real PoC should replace only the routed expert inner loop, preserving
current prefill router semantics.

| field | contract |
|---|---|
| input hidden | `h_flat: bf16[M, 2048]` |
| router | current prefill `F.linear -> torch.topk(sorted default) -> softmax` |
| route tensors | `expert_ids: int32[M, 8]`, `routing_weights: fp32[M, 8]` |
| gate/up weight | `uint8[256, 1024, 1024]`, scale `float[256, 1024, 128]`, scalar global |
| down weight | `uint8[256, 2048, 256]`, scale `float[256, 2048, 32]`, scalar global |
| output | `moe_out: bf16[M, 2048]`, routed expert only first |
| numeric gate | compare against `stream_bf16` and resident BF16; argmax must match, cosine reported |
| memory gate | no resident BF16 expert shadow; peak must stay near selected-expert scratch, not 12.64 GiB |
| latency gate | must beat `stream_bf16`; promotion requires approaching or beating resident BF16 prefill |

Do not wire P2 into `resident_runner` until this kernel-level contract passes.
