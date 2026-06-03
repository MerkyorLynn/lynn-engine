# Stage 6 Phase 2-F — opt-in one-layer packed-prefill replacement

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-spark`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Runner:** `scripts/spark_stage6_p2f_one_layer_replacement_verify.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2f_one_layer_replacement_verify_20260604_013401`

## Verdict

**P2-F passes.** The P2-E scheduler/retune path has now moved from harness
composition into the engine dispatch path:

```bash
LYNN_PACKED_PREFILL_SLOW=1
LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid
LYNN_PACKED_PREFILL_P2E_LAYERS=0
```

After deleting `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`, the
engine `_moe_forward()` path runs packed active experts directly from resident
NVFP4 and keeps the shared expert BF16.

This is still opt-in and layer-filtered. Default serving is unchanged.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2f_one_layer_replacement_verify.py \
    --batches 16,64 \
    --warmup 1 --iters 2 --repeats 2 \
    --json-out reports/stage6/p2f_one_layer_replacement_verify_20260604_013401/result.json
```

## Engine Change

`engine/full_forward.py` now accepts:

| flag | meaning |
|---|---|
| `LYNN_PACKED_PREFILL_SLOW=1` | Enable packed-prefill proof/replacement modes. |
| `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid` | Use sort scheduler + retuned packed active MoE. |
| `LYNN_PACKED_PREFILL_P2E_LAYERS=0` | Restrict P2E replacement to selected layers; unset/`all` means all layers. |
| `LYNN_PACKED_PREFILL_P2E_BLOCK_INTER=8` | P2-E retuned gate/up tile. |

If a layer is not selected, `p2e_hybrid` falls back to `stream_bf16` so the
no-reload proof remains functional for non-selected layers.

## Numeric Gate

| comparison | cosine | rel L2 | max abs | argmax |
|---|---:|---:|---:|---|
| stream vs BF16, M=16 | 1.000000000 | 0 | 0 | match |
| stream vs BF16, M=64 | 1.000000000 | 0 | 0 | match |
| P2E vs BF16, M=16 | 0.999997512 | 2.231e-03 | 1.221e-04 | match |
| P2E vs BF16, M=64 | 0.999997426 | 2.269e-03 | 1.221e-04 | match |

Numeric gate: **PASS** for a one-layer prefill replacement probe.

## Latency Gate

| batch | BF16 full MoE | `stream_bf16` proof | P2E hybrid engine mode | P2E vs BF16 | P2E vs stream |
|---:|---:|---:|---:|---:|---:|
| 16 | 11.63 ms | 494.66 ms | 8.23 ms | 1.412x | 60.08x |
| 64 | 21.10 ms | 504.93 ms | 20.23 ms | 1.043x | 24.96x |

Speed gate: **PASS** for this one-layer opt-in path.

## Memory Gate

| batch | P2E before | P2E peak | `stream_bf16` peak |
|---:|---:|---:|---:|
| 16 | 0.641 GiB | 0.641 GiB | 12.641 GiB |
| 64 | 0.641 GiB | 0.643 GiB | 12.641 GiB |

No-active-shadow gate: **PASS**. P2E avoids the temporary full-layer BF16
dequant peak that `stream_bf16` pays.

## Decision

Bank P2-F and continue to P2-G:

| next gate | requirement |
|---|---|
| P2-G multi-layer replacement smoke | Select several/all MoE layers with `p2e_hybrid`, run no-reload prefill, and compare against BF16/stream for numeric, memory, and latency. |
| P2-H server no-reload A/B | Only after P2-G: use the p2e path to eliminate per-request reload and measure multi-request behavior. |
| Native runtime | P2-F is still Python/Triton dispatch. Long-term parity with llama.cpp still requires CUDA C++ / CUTLASS grouped kernels and eventually a C++ hot-path runtime, especially on FP4-MMA hardware. |

Do not make this default until multi-layer and RC gates pass.
