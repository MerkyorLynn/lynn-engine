# Stage 6 Phase 2-D — router/shared-inclusive one-layer MoE hybrid

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-spark`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Runner:** `scripts/spark_stage6_p2d_one_layer_moe_hybrid_poc.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2d_one_layer_moe_hybrid_poc_20260604_010633`

## Verdict

**P2-D is a mixed gate: numeric + no-active-shadow pass, performance vs resident
BF16 full MoE fails.** It keeps the no-reload service path alive because it is
far faster than the P2 `stream_bf16` proof, but it is not ready for
`resident_runner` integration.

The hybrid path includes:

- router linear + top-k + softmax + eager route grouping inside the timed path;
- packed NVFP4 active experts after deleting BF16 active shadows;
- existing BF16 shared expert path.

Shared expert is intentionally left BF16 in this gate. It is only **0.006 GiB**
for the layer and takes ~46-50us, so it is not the current bottleneck.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2d_one_layer_moe_hybrid_poc.py \
    --batches 16,64 \
    --warmup 1 --iters 2 --repeats 2 \
    --json-out reports/stage6/p2d_one_layer_moe_hybrid_poc_20260604_010633/result.json
```

## Memory

| item | bytes |
|---|---:|
| BF16 active experts | 1.500 GiB |
| packed active experts | 0.563 GiB |
| BF16 shared expert | 0.006 GiB |
| after deleting BF16 active shadows | 0.641 GiB |

Packed hybrid benchmarks ran after deleting `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj`.

| batch | allocated before | peak during hybrid |
|---:|---:|---:|
| 16 | 0.641 GiB | 0.641 GiB |
| 64 | 0.641 GiB | 0.642 GiB |

No-active-shadow gate: **PASS**.

## Numeric Gate

| batch | hybrid vs full BF16 cosine | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 16 | 0.999997512 | 2.231e-03 | 1.221e-04 | match |
| 64 | 0.999997426 | 2.269e-03 | 1.221e-04 | match |

Active packed vs BF16 active remained identical to P2-C-level tolerance:

| batch | active cosine | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 16 | 0.999981186 | 6.135e-03 | 3.052e-05 | match |
| 64 | 0.999981693 | 6.051e-03 | 3.052e-05 | match |

Numeric gate: **PASS** for a one-layer prefill PoC. This is not an RC quality
gate.

## Latency Gate

| batch | unique experts | hybrid full layer | BF16 full MoE | speedup vs BF16 | route/grouping | packed active precomputed | BF16 shared |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 95 | 12.43 ms | 11.68 ms | 0.940x | 2.46 ms | 10.14 ms | 0.050 ms |
| 64 | 207 | 29.21 ms | 21.65 ms | 0.741x | 5.41 ms | 23.99 ms | 0.046 ms |

Performance vs resident BF16 full MoE: **FAIL**.

Compared with the P2 no-reload proof path:

| batch | P2-D hybrid | `stream_bf16` proof |
|---:|---:|---:|
| 16 | 12.43 ms | 496.61 ms |
| 64 | 29.21 ms | 506.22 ms |

M=64 hybrid is ~**17x faster** than `stream_bf16`, so P2 still matters for
removing the **23-24 s reload**. But the timed hybrid also exposes why it should
not be promoted yet: the eager router/grouping cost is 5.41ms/layer at M=64, and
the packed active path is still slower than resident BF16 tensor-core GEMM.

## Decision

Do **not** wire P2-D into serving.

Next work should target the two measured blockers:

| next gate | purpose |
|---|---|
| P2-E grouped scheduler / packed active retune | Reduce route/grouping overhead and active packed latency; keep the no-BF16-active invariant. |
| P2-F opt-in one-layer replacement | Only if P2-E improves latency; replace one layer behind a flag and compare against `stream_bf16`, BF16 full MoE, and no-reload memory. |
| Native/CUDA path | If Spark Triton remains below BF16, move the active grouped kernel contract toward native CUDA / CUTLASS-style scheduling; Python remains control/verification only. |

P2-D banks correctness and memory evidence, not speed promotion.
