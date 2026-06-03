# Stage 6 Phase 2-E — grouped scheduler / packed-active retune

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-spark`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Runner:** `scripts/spark_stage6_p2e_scheduler_active_retune.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2e_scheduler_active_retune_20260604_012046`

## Verdict

**P2-E passes as a one-layer scheduler/retune gate.** It does not solve the full
serving path yet, but it finds two concrete low-risk changes that turn the P2-D
hybrid from slower-than-BF16 into slightly faster-than-BF16 on this one-layer
prefill probe:

- Replace `unique + per-expert mask.nonzero` route grouping with
  `argsort + unique_consecutive`.
- Retune packed gate/up prefill from `block_inter=16` to `block_inter=8`.

Best P2-E hybrid at M=64: **20.25 ms/layer vs BF16 full MoE 21.41 ms/layer =
1.057x**, while preserving the no-active-shadow invariant.

This is still not a server integration gate. It is the first one-layer evidence
that packed no-reload MoE can beat the resident BF16 full-MoE layer when the
scheduler is less wasteful.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2e_scheduler_active_retune.py \
    --batches 16,64 \
    --warmup 1 --iters 2 --repeats 2 \
    --tile-sweep "block_t=32,block_inter=16,block_hidden=128,num_warps=4;block_t=32,block_inter=8,block_hidden=128,num_warps=4;block_t=16,block_inter=8,block_hidden=128,num_warps=4;block_t=64,block_inter=8,block_hidden=128,num_warps=4;block_t=32,block_inter=8,block_hidden=64,num_warps=4;block_t=32,block_inter=8,block_hidden=128,num_warps=8;block_t=32,block_inter=32,block_hidden=128,num_warps=4" \
    --json-out reports/stage6/p2e_scheduler_active_retune_20260604_012046/result.json
```

## Memory

| item | bytes |
|---|---:|
| BF16 active experts | 1.500 GiB |
| packed active experts | 0.563 GiB |
| after deleting BF16 active shadows | 0.641 GiB |

Hybrid peak stayed below **0.642 GiB** for M=16 and M=64 after deleting
`mlp.experts.gate_up_proj` and `mlp.experts.down_proj`.

No-active-shadow gate: **PASS**.

## Scheduler Result

| batch | route mode | route/grouping time | speedup vs current |
|---:|---|---:|---:|
| 16 | current `unique+mask` | 2.325 ms | 1.0x |
| 16 | sort + `unique_consecutive` | 0.341 ms | 6.83x |
| 64 | current `unique+mask` | 5.018 ms | 1.0x |
| 64 | sort + `unique_consecutive` | 0.614 ms | 8.17x |

This is the main win. It is still Python-side grouping, but it removes the
per-expert mask scan that P2-D exposed.

## Packed Active Retune

M=64 packed active sweep:

| tile | block_t | block_inter | block_hidden | warps | active time |
|---|---:|---:|---:|---:|---:|
| baseline | 32 | 16 | 128 | 4 | 24.48 ms |
| best | 32 | 8 | 128 | 4 | 20.12 ms |
| variant | 16 | 8 | 128 | 4 | 20.15 ms |
| variant | 64 | 8 | 128 | 4 | 23.11 ms |
| variant | 32 | 8 | 64 | 4 | 20.47 ms |
| variant | 32 | 8 | 128 | 8 | 23.60 ms |
| rejected | 32 | 32 | 128 | 4 | OOR: shared memory 114688 > 101376 |

Tile gate: **PASS** for `block_t=32, block_inter=8, block_hidden=128,
num_warps=4`.

Scratch-buffer reuse did **not** help:

| batch | allocation baseline | scratch mode | scratch speedup |
|---:|---:|---:|---:|
| 16 | 9.98 ms | 10.37 ms | 0.963x |
| 64 | 24.06 ms | 24.69 ms | 0.974x |

Scratch reuse is rejected for now.

## Hybrid Gate

Using the best tile (`block_inter=8`) and the two route modes:

| batch | route mode | hybrid | BF16 full MoE | speedup vs BF16 | route/grouping |
|---:|---|---:|---:|---:|---:|
| 16 | current | 10.59 ms | 11.58 ms | 1.093x | 2.325 ms |
| 16 | sort | 8.87 ms | 11.58 ms | 1.306x | 0.341 ms |
| 64 | current | 25.26 ms | 21.41 ms | 0.848x | 5.018 ms |
| 64 | sort | 20.25 ms | 21.41 ms | 1.057x | 0.614 ms |

Hybrid speed gate: **PASS only for the sort scheduler**. The current grouping
path still fails at M=64.

Numeric gate:

| comparison | cosine | rel L2 | max abs | argmax |
|---|---:|---:|---:|---|
| scratch vs allocation, M=16 | 1.000000000 | 8.05e-06 | 9.54e-07 | match |
| scratch vs allocation, M=64 | 1.000000000 | 2.41e-05 | 3.81e-06 | match |
| hybrid sort vs BF16, M=16 | 0.999997512 | 2.23e-03 | 1.22e-04 | match |
| hybrid sort vs BF16, M=64 | 0.999997426 | 2.27e-03 | 1.22e-04 | match |

Numeric gate: **PASS** for the one-layer prefill probe.

## Decision

Bank P2-E and move to P2-F:

| next gate | requirement |
|---|---|
| P2-F one-layer opt-in replacement | Implement the sort scheduler + `block_inter=8` packed active path behind a flag and run it as a one-layer replacement, not just a harness composition. |
| P2-G multi-layer no-reload smoke | If P2-F holds, replace selected layers in prefill and compare against `stream_bf16`, BF16 full prefill, memory, and token/numeric agreement. |
| Native/C++ runtime path | P2-E is useful, but it is still Python/Triton scheduling. The long-term llama.cpp chase remains CUDA C++ / CUTLASS-style grouped kernels plus a C++ hot-path runtime when paired with FP4-MMA hardware. |

Do not promote to server default from P2-E alone.
