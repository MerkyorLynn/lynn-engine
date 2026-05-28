# llama.cpp APEX-MTP Service A/B (2026-05-28)

## Result

Request-level `speculative.n_max` is worth adding, but **not because a smaller
draft depth is faster**. In this short Spark run, current `n_max=4` is the best
single-stream APEX-MTP setting. The real finding is concurrency:

- Single stream: `draft-mtp n_max=4` beats AR by about **27%**.
- 4-way concurrent: AR beats `draft-mtp n_max=4` by about **46%**.

So the immediate production opportunity is **dynamic MTP admission**:

```text
single / low queue depth  -> enable draft-mtp, n_max=4
multi-slot / high queue   -> disable MTP or route to AR
```

## Environment

- Host: `dgx-spark`
- Model: `/home/merkyor/models/Qwen3.6-35B-A3B-APEX-MTP-GGUF/Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf`
- Branch: `/home/merkyor/build/llama.cpp` `codex/apex-mtp-service-ab`
- Temp binary: `/home/merkyor/build/llama.cpp/build-cuda-sm121-codex-ab/bin/llama-server`
- Temp port: `18099`
- Production fallback was runtime-masked/stopped during the test, then restored.

The tested GGUF is already imatrix-quantized:

```text
general.file_type = 17
quantize.imatrix.entries_count = 510
quantize.imatrix.chunks_count = 2820
qwen35moe.nextn_predict_layers = 1
tensor mix: Q6_K 218, Q8_0 131, Q5_K 94, F32 308, BF16 2
```

So a future Q4_K_M imatrix package would be a lower-bit package than this
`I-Balanced` build, not simply "the imatrix version" of it.

Production restore check:

```text
lynn-apex-mtp-llamacpp.service active
llama-server restored on 18098
```

## Patch Under Test

The llama.cpp research branch exposes only request-level `speculative.n_max`:

- `tools/server/server-task.cpp` parses `speculative.n_max`.
- `tools/server/server-context.cpp::get_n_draft_max()` clamps the slot draft
  budget to that request value.
- `speculative.type`, `n_min`, and `p_min` remain fixed by server startup,
  because the current implementation treats them as speculative-context config,
  not clean per-request knobs.

## Numbers

Each row used 2 sequential requests plus one 4-way concurrent round,
`max_tokens=96`, `temperature=0`, OpenAI-compatible non-streaming HTTP.

| Mode | Single Wall TPS | Single Server TPS | Single Accept | 4-Way Wall TPS | 4-Way Server TPS | 4-Way Accept |
|---|---:|---:|---:|---:|---:|---:|
| AR baseline | 60.65 | 66.05 | n/a | **124.80** | **33.56** | n/a |
| MTP `n_max=1` | 48.55 | 54.15 | 92.86% | 59.45 | 15.97 | 90.45% |
| MTP `n_max=2` | 64.41 | 69.74 | 86.33% | 73.67 | 20.83 | 84.70% |
| MTP `n_max=4` | **77.01** | **84.40** | 70.92% | 85.62 | 23.86 | 68.24% |

Artifacts:

```text
reports/mtp/service_ab_20260528_115837/
  ar_baseline.json
  mtp_nmax1.json
  mtp_nmax2.json
  mtp_nmax4.json
  summary.json
  run.log
```

## Conclusion

For current llama.cpp APEX-MTP, `n_max=4` is still the right single-stream
default. Lowering to `n_max=1/2` does not improve production TPS.

However, MTP is a poor fit for the current 4-slot concurrent server loop. It
adds draft-context work per slot and does not aggregate as well as plain AR.
The next high-ROI patch is not another static draft depth change; it is
request/queue-aware MTP admission:

1. Keep `--spec-type draft-mtp --spec-draft-n-max 4` available.
2. Add request-level `speculative.n_max`.
3. Route or clamp:
   - `n_max=4` when the slot is alone or queue depth is low.
   - `n_max=0` when multiple slots are active.

This gives Lynn a clean serving policy while deeper work continues on the
serial `draft-mtp` loop and packed verifier kernels.

## Q4_K_M imatrix Follow-Up

A Q4_K_M imatrix GGUF with the same embedded MTP tensors could be faster for
single-stream decode because Spark decode is heavily weight-bandwidth bound.
The likely gain is incremental rather than magical, because the tested model is
already imatrix-quantized and MTP adds draft-context overhead.

Expected direction:

| Package | Expected TPS | Risk |
|---|---:|---|
| Current `I-Balanced` imatrix | measured 77.01 single-stream wall TPS | current quality baseline |
| Q4_K_M imatrix + embedded MTP | likely faster, needs measurement | quality drop and possible MTP accept-rate shift |
| Plain Q4_K_M imatrix without MTP tensors | cannot run `draft-mtp` | no APEX-MTP speed path |

Release implication: for HF/ModelScope, upload the MTP-enabled GGUF or a clearly
paired base + MTP sidecar. Do not publish a Q4_K_M base-only package as
"APEX-MTP capable" unless `qwen35moe.nextn_predict_layers=1` and the trailing
MTP block tensors are present.
