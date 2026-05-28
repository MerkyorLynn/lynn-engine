---
library_name: gguf
pipeline_tag: text-generation
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.5-9B
base_model_relation: quantized
language:
- zh
- en
tags:
- qwen3.5
- gguf
- q4_k_m
- imatrix
- mtp
- draft-mtp
- llama.cpp
- lynn
quantized_by: Lynn (nerkyor)
---

# Qwen3.5-9B Q4_K_M imatrix MTP GGUF

This repository contains a local experimental GGUF artifact based on
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). The file is packaged
for llama.cpp and includes an embedded MTP-capable block that can be used with
`draft-mtp` serving.

## What Was Produced

The experiment produced one GGUF file:

| File | Size | SHA256 | Role |
|---|---:|---|---|
| `Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf` | 5.4 GB | `0f292ba0d1058065a6624883a76a2adf00b266d07b9396ed67b155ff522e18d4` | llama.cpp GGUF with Q4_K_M imatrix quantization and embedded MTP-capable tensors |

## Release Links

- Hugging Face MTP release: <https://huggingface.co/nerkyor/Qwen3.5-9B-GGUF-imatrix-MTP>
- ModelScope MTP release: <https://modelscope.cn/models/Merkyor/Qwen3.5-9B-GGUF-imatrix-MTP>
- Stable non-MTP imatrix release: <https://modelscope.cn/models/Merkyor/Qwen3.5-9B-GGUF-imatrix>

Objective facts observed during the Spark run:

- The base model family is Qwen3.5 9B.
- The GGUF contains MTP metadata/tensors: `qwen35.nextn_predict_layers` and
  `blk.32.nextn.*`.
- llama.cpp initialized the embedded MTP path with
  `common_speculative_impl_draft_mtp`.
- Request-level `speculative.n_max` produced non-zero draft and accepted-draft
  counters in llama.cpp timings.

## Provenance Note

Base-model attribution and license remain with Qwen. The quantized GGUF
packaging and Spark serving benchmark metadata were produced by Lynn.

The local manifest used for this release does not encode the training dataset
for the embedded MTP head. This model card therefore does not claim a specific
MTP-head training dataset and does not redistribute private training data.

## Spark Benchmark

Host: DGX Spark / GB10 / sm_121, llama.cpp branch `codex/apex-mtp-service-ab`,
OpenAI-compatible HTTP requests, `max_tokens=96`, `temperature=0`.

| Mode | Single Wall TPS | Single Server TPS | Single Accept | 4-Way Wall TPS | 4-Way Server TPS | 4-Way Accept |
|---|---:|---:|---:|---:|---:|---:|
| AR on non-MTP server | 36.61 | 38.18 | n/a | 120.58 | 31.83 | n/a |
| AR on MTP-capable server (`n_max=0`) | 32.45 | 34.12 | n/a | 32.13 | 8.25 | n/a |
| MTP `n_max=1` | 35.76 | 36.92 | 91.84% | 46.67 | 12.45 | 86.63% |
| MTP `n_max=2` | 46.58 | 48.51 | 79.45% | 62.72 | 16.91 | 76.85% |
| MTP `n_max=4` | **60.95** | **64.20** | 64.15% | 76.52 | 21.55 | 60.54% |

In this short benchmark, `draft-mtp n_max=4` improved single-stream wall TPS by
about 66% versus the non-MTP AR server: 36.61 -> 60.95 TPS.

For 4-way concurrent serving, the non-MTP AR server remained faster. Treat this
artifact as a low-queue-depth / single-stream acceleration experiment, not as a
high-concurrency default.

## Run With llama.cpp

Use a llama.cpp build that supports Qwen3.5 MTP / `draft-mtp`.

```bash
llama-server \
  -m Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf \
  --host 127.0.0.1 \
  --port 18099 \
  -a qwen35-9b-q4km-imatrix-mtp \
  --ctx-size 32768 \
  --parallel 4 \
  --threads 4 \
  --n-gpu-layers 999 \
  -fa on \
  --jinja \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --reasoning auto \
  --reasoning-budget -1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --metrics
```

Important: do not start the tested llama.cpp branch with
`--spec-type none,draft-mtp`. In that branch, `none` disables speculative
implementations. Start with `--spec-type draft-mtp` and use request-level
`"speculative.n_max": 0` when an AR control request is needed.

## License

Apache 2.0, inherited from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
