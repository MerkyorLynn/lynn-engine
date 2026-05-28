---
license: apache-2.0
license_name: apache-2.0
language:
- zh
- en
pipeline_tag: text-generation
tags:
- qwen3.5
- gguf
- q4_k_m
- imatrix
- mtp
- draft-mtp
- llama.cpp
- lynn
frameworks:
- llama.cpp
base_model:
- Qwen/Qwen3.5-9B
base_model_relation: quantized
quantized_by: Lynn (Merkyor)
---

# Qwen3.5-9B Q4_K_M imatrix MTP GGUF

本仓库提供一个基于 [Qwen/Qwen3.5-9B](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)
的本地实验 GGUF artifact。该文件面向 llama.cpp，包含可用于 `draft-mtp`
服务路径的内嵌 MTP-capable block。

## 本次实验产物

| 文件 | 大小 | SHA256 | 用途 |
|---|---:|---|---|
| `Qwen3.5-9B-Q4_K_M-imatrix-mtp.gguf` | 5.4 GB | `0f292ba0d1058065a6624883a76a2adf00b266d07b9396ed67b155ff522e18d4` | llama.cpp GGUF，包含 Q4_K_M imatrix 量化和内嵌 MTP-capable tensors |

## 发布地址

- ModelScope MTP 新版：<https://modelscope.cn/models/Merkyor/Qwen3.5-9B-GGUF-imatrix-MTP>
- Hugging Face MTP 新版：<https://huggingface.co/nerkyor/Qwen3.5-9B-GGUF-imatrix-MTP>
- 稳定非 MTP imatrix 旧版：<https://modelscope.cn/models/Merkyor/Qwen3.5-9B-GGUF-imatrix>

本次 Spark 实验能客观确认的事实：

- base model family 是 Qwen3.5 9B。
- GGUF 中存在 MTP metadata/tensors：`qwen35.nextn_predict_layers` 和
  `blk.32.nextn.*`。
- llama.cpp 能初始化该 GGUF 的 MTP 路径，日志包含
  `common_speculative_impl_draft_mtp`。
- 请求级 `speculative.n_max` 会产生非零 draft token 和 accepted draft token
  计数。

## Provenance 说明

base model attribution 和 license 归属 Qwen。Lynn 侧提供的是 GGUF 包装、
量化/发布元信息以及 Spark 服务 benchmark 记录。

当前本地 manifest 没有记录内嵌 MTP head 的训练数据来源，因此本 model card
不声明某个具体 MTP-head training dataset，也不分发任何私有训练数据。

## Spark Benchmark

测试环境：DGX Spark / GB10 / sm_121，llama.cpp 分支
`codex/apex-mtp-service-ab`，OpenAI-compatible HTTP，`max_tokens=96`，
`temperature=0`。

| 模式 | 单流 Wall TPS | 单流 Server TPS | 单流 Accept | 4 路 Wall TPS | 4 路 Server TPS | 4 路 Accept |
|---|---:|---:|---:|---:|---:|---:|
| 非 MTP server AR | 36.61 | 38.18 | n/a | 120.58 | 31.83 | n/a |
| MTP-capable server AR (`n_max=0`) | 32.45 | 34.12 | n/a | 32.13 | 8.25 | n/a |
| MTP `n_max=1` | 35.76 | 36.92 | 91.84% | 46.67 | 12.45 | 86.63% |
| MTP `n_max=2` | 46.58 | 48.51 | 79.45% | 62.72 | 16.91 | 76.85% |
| MTP `n_max=4` | **60.95** | **64.20** | 64.15% | 76.52 | 21.55 | 60.54% |

在这个短测里，`draft-mtp n_max=4` 将单流 wall TPS 从 36.61 提到 60.95，
约 +66%。但 4 路并发下，非 MTP AR server 仍然更快。因此这个 artifact
更适合作为低队列深度/单流加速实验，不建议直接作为高并发默认服务配置。

## llama.cpp 启动方式

需要使用支持 Qwen3.5 MTP / `draft-mtp` 的 llama.cpp build。

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

注意：不要在本次测试的 llama.cpp 分支上使用
`--spec-type none,draft-mtp`。该分支中 `none` 会关闭 speculative
implementations。需要 AR 对照时，保持 server 以 `--spec-type draft-mtp`
启动，并在请求级使用 `"speculative.n_max": 0`。

## 许可证

Apache 2.0，继承自上游 [Qwen/Qwen3.5-9B](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)。
