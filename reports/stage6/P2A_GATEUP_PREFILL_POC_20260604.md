# Stage 6 Phase 2-A — single-expert packed gate/up prefill PoC

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Kernel:** `triton_kernels.nvfp4_moe::nvfp4_prefill_gate_up_silu_one_expert`  
**Runner:** `scripts/spark_stage6_p2a_gateup_prefill_poc.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2a_gateup_prefill_poc_20260604_004226`  
**Sweep dir:** `/home/merkyor/lynn-engine/reports/stage6/p2a_gateup_prefill_sweep_20260604_004300`

## Verdict

**P2-A single-expert gate/up is a valid no-shadow component, but it is NOT a
performance win on Spark.**

The kernel reads one expert's packed NVFP4 gate/up weights, dequants in the
kernel, computes batched `gate/up -> silu(gate) * up`, and writes BF16
intermediate activations. It passes the component numeric/no-shadow gate in the
main run, but it loses badly to BF16 `F.linear` for M>1.

This is still useful as a component probe. It shows that a naive scalar-dequant
Triton gate/up is not enough; the next P2 question is routed grouping economics
and/or native FP4-MMA/CUTLASS-style kernels, not more tuning of this scalar
single-expert bridge.

## Main Run

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2a_gateup_prefill_poc.py \
    --batches 1,4,16,64 \
    --warmup 5 --iters 20 --repeats 3 \
    --json-out reports/stage6/p2a_gateup_prefill_poc_20260604_004226/result.json
```

Selected expert: `37` from the M=64 router sample.

| item | value |
|---|---:|
| hidden | 2048 |
| expert intermediate | 512 |
| one-expert BF16 gate/up shadow | 4.00 MiB |
| one-expert packed gate/up | 1.50 MiB |
| BF16-to-packed ratio | 2.667x |

## Numeric Gate

| batch | cosine vs BF16 gate/up | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 1 | 0.999990539 | 4.471e-03 | 2.441e-04 | match |
| 4 | 0.999992760 | 3.817e-03 | 2.441e-04 | match |
| 16 | 0.999992440 | 3.911e-03 | 2.441e-04 | match |
| 64 | 0.999992392 | 3.915e-03 | 4.883e-04 | match |

Component numeric gate: **PASS** for this run.

## Latency Gate

| batch | packed gate/up | BF16 gate/up | speedup vs BF16 | packed/token | BF16/token |
|---:|---:|---:|---:|---:|---:|
| 1 | 82.77 us | 16.29 us | 0.197x | 82.77 us | 16.29 us |
| 4 | 82.72 us | 18.81 us | 0.227x | 20.68 us | 4.70 us |
| 16 | 82.67 us | 19.22 us | 0.233x | 5.17 us | 1.20 us |
| 64 | 240.33 us | 19.10 us | 0.079x | 3.76 us | 0.30 us |

Performance gate vs BF16: **FAIL**.

## Memory Gate

Timed packed benchmarks ran after deleting `mlp.experts.gate_up_proj`.

| metric | value |
|---|---:|
| allocated after deleting BF16 gate/up | 0.953 GiB |
| peak during packed gate/up bench | 0.953 GiB |

No-shadow component gate: **PASS**.

## Tile Sweep

Sweep command covered M=16/64 with `BLOCK_T/BLOCK_INTER/BLOCK_HIDDEN` variants.

| tile | M=16 packed | M=16 BF16 | M=16 speedup | M=64 packed | M=64 BF16 | M=64 speedup | status |
|---|---:|---:|---:|---:|---:|---:|---|
| `16/16/128` | 83.67 us | 20.80 us | 0.249x | 240.96 us | 19.79 us | 0.082x | numeric M64 argmax mismatch in this sample |
| `32/16/128` | 83.60 us | 20.91 us | 0.250x | 174.96 us | 20.05 us | 0.115x | numeric M64 argmax mismatch in this sample |
| `16/32/128` | - | - | - | - | - | - | OOR, shared memory 106496 > 101376 |
| `32/32/128` | - | - | - | - | - | - | OOR, shared memory 114688 > 101376 |
| `16/16/256` | - | - | - | - | - | - | OOR, shared memory 114688 > 101376 |
| `32/16/256` | - | - | - | - | - | - | OOR, shared memory 131072 > 101376 |

The best M=64 tile (`32/16/128`) is still only **0.115x** of BF16.

## Decision

Keep `nvfp4_prefill_gate_up_silu_one_expert` and its harness as a component
probe. Do **not** wire it into `resident_runner`, and do not count it as a
serving win.

Next useful P2 step:

| path | requirement |
|---|---|
| P2-B routed gate/up grouping | Use current prefill router, group rows by expert, call the component kernel once per unique expert, and measure total gate/up-only latency/launch cost. |
| Native FP4-MMA/CUTLASS | Required if packed gate/up must beat BF16 tensor-core GEMM on Spark-like M>1 shapes. |
| Full P2 routed MoE | Only after gate/up grouping and down projection have separate numeric/memory/latency evidence. |
