# Stage 6 Phase 2-B — routed gate/up grouping lower-bound

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Runner:** `scripts/spark_stage6_p2b_routed_gateup_grouping_poc.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2b_routed_gateup_grouping_poc_20260604_004923`

## Verdict

**P2-B passes as a routed gate/up lower-bound and keeps P2 alive as a no-reload
serving path. It is not a BF16-speed promotion.**

This harness preserves the current prefill router, precomputes token/slot groups
by unique expert, then calls the P2-A packed gate/up component once per unique
expert. It excludes down projection, route weighting, `index_add`, and shared
expert, so it is a lower-bound for routed MoE prefill.

Result: packed routed gate/up is **memory-clean** and **numeric-aligned**, and it
is far below the `stream_bf16` full-layer dequant cost. But it still loses to
BF16 grouped gate/up by ~2.3x.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2b_routed_gateup_grouping_poc.py \
    --batches 16,64 \
    --warmup 1 --iters 3 --repeats 2 \
    --json-out reports/stage6/p2b_routed_gateup_grouping_poc_20260604_004923/result.json
```

## Route Shape

| batch | route slots | unique experts | largest groups |
|---:|---:|---:|---|
| 16 | 128 | 95 | 5 experts with 3 rows; many 1-2 row groups |
| 64 | 512 | 207 | largest group 10 rows; top groups 7/7/7/6/6/6 rows |

This is a high-dispatch shape. Even with routes precomputed, packed grouping
launches once per unique expert.

## Numeric Gate

| batch | cosine vs BF16 grouped gate/up | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 16 | 0.999992234 | 3.961e-03 | 4.883e-04 | match |
| 64 | 0.999992104 | 3.983e-03 | 9.766e-04 | match |

Numeric gate: **PASS** for a component/lower-bound harness.

## Latency Gate

| batch | unique experts | packed grouped gate/up | BF16 grouped gate/up | speedup vs BF16 | packed/token | BF16/token |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 95 | 9.34 ms | 4.14 ms | 0.443x | 583.88 us | 258.85 us |
| 64 | 207 | 20.00 ms | 8.47 ms | 0.423x | 312.51 us | 132.34 us |

Performance vs BF16: **FAIL**.

But this is materially better than the P2 census `stream_bf16` proof path:

| batch | packed grouped gate/up | P2 census stream_bf16 full MoE |
|---:|---:|---:|
| 16 | 9.34 ms | 496.61 ms |
| 64 | 20.00 ms | 506.22 ms |

The M=64 routed gate/up lower-bound is already **~25x below** the
`stream_bf16` per-layer proof cost. Down projection will add substantial work,
but the no-reload serving path remains plausible as a replacement for the
~23-24 s per-request reload, even if it will not beat resident BF16 prefill.

## Memory Gate

Packed benchmarks ran after deleting `mlp.experts.gate_up_proj`.

| batch | allocated before | peak during packed |
|---:|---:|---:|
| 16 | 0.954 GiB | 0.954 GiB |
| 64 | 0.954 GiB | 0.955 GiB |

No-shadow gate: **PASS**.

## Decision

Do not wire P2-B into `resident_runner`; it is gate/up-only. Continue to P2-C
with routed down projection and full routed-output accounting.

Next useful work:

| path | requirement |
|---|---|
| P2-C routed down | Reuse `inter[M,K,512]`, packed down weights, routing weights, and emit `moe_out[M,2048]`; measure numeric/memory/latency. |
| Full P2 routed MoE | Compose routed gate/up + down and compare against `stream_bf16` and resident BF16 for one layer. |
| Server integration | Only after one-layer full P2 beats `stream_bf16` and remains memory-clean. |
