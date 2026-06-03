# Stage 6 Phase 2-C — active routed MoE lower-bound

**Date:** 2026-06-04  
**Host:** Spark GB10 (`dgx-via-n5`)  
**Model:** `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`  
**Layer:** `0`  
**Runner:** `scripts/spark_stage6_p2c_active_moe_lower_bound_poc.py`  
**Remote run dir:** `/home/merkyor/lynn-engine/reports/stage6/p2c_active_moe_lower_bound_poc_20260604_005645`

## Verdict

**P2-C passes as a full active routed expert lower-bound and keeps the no-reload
serving path alive. It is not a BF16 prefill-speed promotion.**

This composes:

- P2-B routed gate/up grouping from packed NVFP4.
- Existing `nvfp4_grouped_down_weighted_sum` per token.
- Current prefill router results, precomputed outside the timed lower-bound.

It excludes shared expert, router latency, layernorm, residual, and server
integration. The result is a one-layer active expert lower-bound, not a full
serving path.

## Command

```bash
docker run --rm --gpus all --ipc=host \
  -e PYTHONNOUSERSITE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/merkyor:/home/merkyor \
  -w /home/merkyor/lynn-engine \
  lynn-eval-base:cu13 \
  python3 -u scripts/spark_stage6_p2c_active_moe_lower_bound_poc.py \
    --batches 16,64 \
    --warmup 1 --iters 2 --repeats 2 \
    --json-out reports/stage6/p2c_active_moe_lower_bound_poc_20260604_005645/result.json
```

## Route Shape

| batch | route slots | unique experts | largest groups |
|---:|---:|---:|---|
| 16 | 128 | 95 | 5 experts with 3 rows; many 1-2 row groups |
| 64 | 512 | 207 | largest group 10 rows; top groups 7/7/7/6/6/6 rows |

The shape remains high-dispatch, but now includes both gate/up and down.

## Numeric Gate

| batch | cosine vs BF16 active MoE | rel L2 | max abs | argmax |
|---:|---:|---:|---:|---|
| 16 | 0.999981186 | 6.135e-03 | 3.052e-05 | match |
| 64 | 0.999981693 | 6.051e-03 | 3.052e-05 | match |

Numeric gate: **PASS** for the active-routed lower-bound. This is not yet an RC
quality gate; it is one-layer active expert parity evidence.

## Latency Gate

| batch | unique experts | packed active MoE | BF16 active MoE | speedup vs BF16 | packed/token | BF16/token |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 95 | 10.13 ms | 6.42 ms | 0.633x | 633.36 us | 401.11 us |
| 64 | 207 | 23.83 ms | 13.34 ms | 0.560x | 372.33 us | 208.41 us |

Performance vs BF16 active MoE: **FAIL**.

But compared to the no-reload proof path from P2 census:

| batch | packed active lower-bound | `stream_bf16` full MoE proof |
|---:|---:|---:|
| 16 | 10.13 ms | 496.61 ms |
| 64 | 23.83 ms | 506.22 ms |

The M=64 active routed lower-bound is **~21x faster** than the current
`stream_bf16` no-reload proof. This is the key service-path signal: packed
prefill may remove the **23-24 s reload** even if it does not beat resident BF16
prefill speed.

## Memory Gate

Packed benchmarks ran after deleting both `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj`.

| batch | allocated before | peak during packed active |
|---:|---:|---:|
| 16 | 0.641 GiB | 0.641 GiB |
| 64 | 0.641 GiB | 0.642 GiB |

No-shadow gate: **PASS**.

## Decision

Continue P2 toward a one-layer full-MoE no-reload harness:

| path | requirement |
|---|---|
| P2-D shared/router-inclusive one-layer | Add router timing and shared expert accounting, compare against `stream_bf16` and BF16 full `_moe_forward`. |
| P2-E full prefill smoke | Replace `stream_bf16` for one or more layers behind an opt-in flag; prove no reload, stable memory, token/numeric agreement. |
| Server integration | Only after P2-D/E prove the one-layer path beats `stream_bf16` and stays memory-clean. |

Do not promote P2-C into `resident_runner`; it is active-routed expert only.
