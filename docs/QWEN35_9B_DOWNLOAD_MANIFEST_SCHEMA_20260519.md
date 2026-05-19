# Qwen3.5-9B 下载 Manifest Schema · 2026-05-19

## 目标

定义首发下载页使用的 release manifest JSON schema。Manifest 用来让 `engine.merkyorlynn.com` 展示模型下载卡片、让 `dl.merkyorlynn.com` 承载大文件、让 GitHub 保存 schema 与小型 JSON 元数据。

口径与 Mac runbook / site-copy 保持一致：

- Mac 稳定轨：Qwen3.5-9B Q4_K_M imatrix GGUF + llama.cpp / LM Studio / CLI。
- NVIDIA 稳定轨：Lynn Engine + NVIDIA NVFP4 compatibility artifact。
- BF16 仅作为 reference，不作为普通用户默认下载。
- 不声明未知最终指标；未知 sha256 使用占位，并标记为 public publish 前 release-blocking。

## Manifest 文件位置

建议发布：

```text
https://engine.merkyorlynn.com/docs/qwen35-9b/download-manifest.json
https://dl.merkyorlynn.com/models/qwen35-9b/download-manifest.json
```

GitHub 保存：

```text
docs/schemas/qwen35_9b_download_manifest.schema.json
release/qwen35-9b/download-manifest.json
```

其中 GitHub 只存 schema 和小 JSON，不存大模型文件。

## Manifest 顶层结构

```json
{
  "schema_version": "qwen35-9b-download-manifest-v1",
  "model_family": "Qwen3.5-9B",
  "created_at": "2026-05-19T00:00:00Z",
  "updated_at": "2026-05-19T00:00:00Z",
  "release_channel": "first_release",
  "artifacts": []
}
```

### 顶层字段

| Field | Type | Required | Notes |
|---|---|---:|---|
| `schema_version` | string | yes | Manifest 版本号；breaking change 时递增。 |
| `model_family` | string | yes | 固定为 `Qwen3.5-9B`。 |
| `created_at` | string | yes | ISO-8601 UTC。 |
| `updated_at` | string | yes | ISO-8601 UTC。 |
| `release_channel` | string | yes | 例如 `first_release`、`candidate`、`archive`。 |
| `artifacts` | array | yes | Artifact entries。 |

## Artifact entry schema

每个 artifact entry 覆盖一个可下载对象或一个目录包。

| Field | Type | Required | Notes |
|---|---|---:|---|
| `artifact_id` | string | yes | 稳定 ID，例如 `qwen35-9b-q4km-imatrix-gguf`。 |
| `model_id` | string | yes | 例如 `qwen35-9b`。 |
| `variant` | string | yes | 例如 `mac_q4km_imatrix`、`bf16_reference`、`nvidia_nvfp4_compatibility`。 |
| `quant` | string | yes | 例如 `Q4_K_M_imatrix`、`BF16`、`NVFP4_W4A16`。 |
| `runtime_track` | enum | yes | `mac_llamacpp`、`nvidia_lynn_engine`、`reference`。 |
| `filename` | string | yes | 文件名或包名。 |
| `size_bytes` | integer/string | yes | 正式发布必须是 integer；未知时用 `TODO_SIZE_BYTES_RELEASE_BLOCKING`。 |
| `sha256` | string | yes | 正式发布必须是 64 位 hex；未知时用 `TODO_SHA256_RELEASE_BLOCKING`。 |
| `sources` | object | yes | 三源下载信息：`dl.merkyorlynn.com`、`hf`、`modelscope`。 |
| `recommended` | boolean | yes | 是否推荐给对应用户默认下载。 |
| `status` | enum | yes | `promote_ready`、`conditional`、`research`、`reference`。 |
| `quality_metrics` | object | yes | MMLU/GPQA 等公开指标；未知字段用 `null`，不要编造。 |
| `speed_metrics` | object | yes | TPS/smoke 等公开指标；未知字段用 `null`，不要编造。 |
| `license` | string | yes | 模型和分发许可摘要。 |
| `notice` | string | yes | 用户可见限制和说明。 |
| `created_at` | string | yes | Artifact entry 创建时间，ISO-8601 UTC。 |

## JSON Schema 草案

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://engine.merkyorlynn.com/schemas/qwen35_9b_download_manifest.schema.json",
  "title": "Qwen3.5-9B Download Manifest",
  "type": "object",
  "required": ["schema_version", "model_family", "created_at", "updated_at", "release_channel", "artifacts"],
  "properties": {
    "schema_version": {"type": "string", "pattern": "^qwen35-9b-download-manifest-v[0-9]+$"},
    "model_family": {"type": "string", "const": "Qwen3.5-9B"},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"},
    "release_channel": {"type": "string"},
    "artifacts": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": [
          "artifact_id",
          "model_id",
          "variant",
          "quant",
          "runtime_track",
          "filename",
          "size_bytes",
          "sha256",
          "sources",
          "recommended",
          "status",
          "quality_metrics",
          "speed_metrics",
          "license",
          "notice",
          "created_at"
        ],
        "properties": {
          "artifact_id": {"type": "string"},
          "model_id": {"type": "string"},
          "variant": {"type": "string"},
          "quant": {"type": "string"},
          "runtime_track": {"enum": ["mac_llamacpp", "nvidia_lynn_engine", "reference"]},
          "filename": {"type": "string"},
          "size_bytes": {"oneOf": [{"type": "integer", "minimum": 1}, {"type": "string", "pattern": "^TODO_.*_RELEASE_BLOCKING$"}]},
          "sha256": {"type": "string", "pattern": "^([a-f0-9]{64}|TODO_.*_RELEASE_BLOCKING)$"},
          "sources": {
            "type": "object",
            "required": ["dl.merkyorlynn.com", "hf", "modelscope"],
            "properties": {
              "dl.merkyorlynn.com": {"type": "string"},
              "hf": {"type": "string"},
              "modelscope": {"type": "string"}
            }
          },
          "recommended": {"type": "boolean"},
          "status": {"enum": ["promote_ready", "conditional", "research", "reference"]},
          "quality_metrics": {"type": "object"},
          "speed_metrics": {"type": "object"},
          "license": {"type": "string"},
          "notice": {"type": "string"},
          "created_at": {"type": "string", "format": "date-time"}
        }
      }
    }
  }
}
```

## 示例 manifest

> 注意：示例中的 sha256 和部分 size 是占位。任何 `TODO_*_RELEASE_BLOCKING` 字段都必须在 public publish 前替换，否则阻塞发布。

```json
{
  "schema_version": "qwen35-9b-download-manifest-v1",
  "model_family": "Qwen3.5-9B",
  "created_at": "2026-05-19T00:00:00Z",
  "updated_at": "2026-05-19T00:00:00Z",
  "release_channel": "first_release",
  "artifacts": [
    {
      "artifact_id": "qwen35-9b-q4km-imatrix-gguf",
      "model_id": "qwen35-9b",
      "variant": "mac_q4km_imatrix",
      "quant": "Q4_K_M_imatrix",
      "runtime_track": "mac_llamacpp",
      "filename": "Qwen3.5-9B-Q4_K_M-imatrix.gguf",
      "size_bytes": "TODO_Q4KM_SIZE_BYTES_RELEASE_BLOCKING",
      "sha256": "TODO_Q4KM_SHA256_RELEASE_BLOCKING",
      "sources": {
        "dl.merkyorlynn.com": "https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf",
        "hf": "TODO_HF_QWEN35_9B_Q4KM_REPO",
        "modelscope": "TODO_MODELSCOPE_QWEN35_9B_Q4KM_REPO"
      },
      "recommended": true,
      "status": "promote_ready",
      "quality_metrics": {
        "mmlu_500": 0.76,
        "gpqa_diamond": 0.3737,
        "thinking_on_32k_gpqa": "in_progress"
      },
      "speed_metrics": {
        "runtime": "llama.cpp",
        "smoke_required": ["/v1/models", "/v1/chat/completions"]
      },
      "license": "TODO_LICENSE_SUMMARY_RELEASE_BLOCKING",
      "notice": "Stable Mac path. Verify checksum before first run. 32K thinking-on GPQA is capability mode, not short-answer TPS benchmark.",
      "created_at": "2026-05-19T00:00:00Z"
    },
    {
      "artifact_id": "qwen35-9b-bf16-reference",
      "model_id": "qwen35-9b",
      "variant": "bf16_reference",
      "quant": "BF16",
      "runtime_track": "reference",
      "filename": "qwen35-9b-bf16-reference.tar.zst",
      "size_bytes": "TODO_BF16_SIZE_BYTES_RELEASE_BLOCKING",
      "sha256": "TODO_BF16_SHA256_RELEASE_BLOCKING",
      "sources": {
        "dl.merkyorlynn.com": "https://dl.merkyorlynn.com/models/qwen35-9b/bf16/qwen35-9b-bf16-reference.tar.zst",
        "hf": "TODO_HF_QWEN35_9B_BF16_REPO",
        "modelscope": "TODO_MODELSCOPE_QWEN35_9B_BF16_REPO"
      },
      "recommended": false,
      "status": "reference",
      "quality_metrics": {
        "mmlu_500": 0.772,
        "gpqa_diamond": 0.4495
      },
      "speed_metrics": {},
      "license": "TODO_LICENSE_SUMMARY_RELEASE_BLOCKING",
      "notice": "Reference artifact for calibration and quality ceiling; not the default user download.",
      "created_at": "2026-05-19T00:00:00Z"
    },
    {
      "artifact_id": "qwen35-9b-nvidia-nvfp4-compatibility",
      "model_id": "qwen35-9b",
      "variant": "nvidia_nvfp4_compatibility",
      "quant": "NVFP4_W4A16",
      "runtime_track": "nvidia_lynn_engine",
      "filename": "qwen35-9b-nvfp4-w4a16.tar.zst",
      "size_bytes": "TODO_NVFP4_SIZE_BYTES_RELEASE_BLOCKING",
      "sha256": "TODO_NVFP4_SHA256_RELEASE_BLOCKING",
      "sources": {
        "dl.merkyorlynn.com": "https://dl.merkyorlynn.com/models/qwen35-9b/nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.tar.zst",
        "hf": "TODO_HF_QWEN35_9B_NVFP4_W4A16_REPO",
        "modelscope": "TODO_MODELSCOPE_QWEN35_9B_NVFP4_W4A16_REPO"
      },
      "recommended": true,
      "status": "conditional",
      "quality_metrics": {
        "mmlu_500": "75.20-76.00%",
        "gpqa_diamond": 0.4293
      },
      "speed_metrics": {
        "runtime": "lynn_engine",
        "safe_profile_decode_tps": "60-62 class"
      },
      "license": "TODO_LICENSE_SUMMARY_RELEASE_BLOCKING",
      "notice": "Stable NVIDIA path through Lynn Engine NVFP4 W4A16. Verify checksum before first run.",
      "created_at": "2026-05-19T00:00:00Z"
    }
  ]
}
```

## 三源镜像一致性检查流程

### 1. Manifest 版本锁定

1. 先冻结 `schema_version` 和 artifact list。
2. 每次改 artifact 文件名、路径、size、sha256 或 status，都更新 `updated_at`。
3. breaking schema change 使用新版本，例如 `qwen35-9b-download-manifest-v2`。

### 2. HEAD / size 检查

对 `dl.merkyorlynn.com`、HF、ModelScope 三源分别检查：

```bash
curl -I <url>
```

记录：

- HTTP status；
- `Content-Length`；
- `ETag` / object version（如果源提供）；
- 是否支持 `Accept-Ranges: bytes`。

要求：

- 三源 `Content-Length` 必须等于 manifest `size_bytes`；
- 不允许把未知 size 标成通过；
- `TODO_*_RELEASE_BLOCKING` 必须在 public publish 前清零。

### 3. 断点续传检查

对大文件下载推荐：

```bash
curl -L --fail --continue-at - --output <filename>.partial <url>
```

完成后只在 sha256 通过时改名：

```bash
mv <filename>.partial <filename>
```

要求：

- `.partial` 文件不得被页面展示为可用 artifact；
- 失败的 partial 文件必须重新续传或删除重下；
- sha256 通过前不得写入 release-ready 状态。

### 4. sha256 检查

下载每个源的 artifact 后计算：

```bash
shasum -a 256 <filename>
```

要求：

- 三源 sha256 必须完全一致；
- sha256 必须等于 manifest `sha256`；
- 任一源 mismatch 都阻塞 public publish；
- checksum 文件自身也应有 GitHub tag 或 release asset 记录。

### 5. Partial 文件处理

| Case | Action |
|---|---|
| `.partial` size 小于 manifest size | 继续断点续传或删除重下 |
| `.partial` size 等于 manifest size 但 sha256 mismatch | 删除并重下，不得改名 |
| final filename sha256 mismatch | 移回 `.bad` 或删除，不得作为可用下载 |
| source HEAD 不支持 range | 页面可提示不支持续传，但仍必须 sha256 校验 |

## 页面消费方式

### `engine.merkyorlynn.com`

- 读取 manifest JSON；
- 按 `runtime_track` 分组展示 Mac / NVIDIA / reference；
- 用 `recommended` 决定默认下载卡片；
- 用 `status` 控制标签：`promote_ready`、`conditional`、`research`、`reference`；
- 显示 size、sha256、license、notice；
- 如果存在 `TODO_*_RELEASE_BLOCKING`，页面必须显示「未发布」或隐藏下载按钮。

### `dl.merkyorlynn.com`

- 只负责大文件、manifest、checksums、release-gate JSON 的静态分发；
- 不承载复杂页面逻辑；
- 目录布局应与 manifest `filename` 和 `sources.dl.merkyorlynn.com` 一致。

### GitHub

- 存 schema、小型 manifest JSON、release gate JSON 的可审查版本；
- 不存大模型文件；
- PR review 关注 schema 变更、status、notice、license、checksum 是否 release-blocking。

## 发布阻塞规则

Public publish 前必须满足：

- 所有推荐 artifact 的 `size_bytes` 是整数，不是 TODO；
- 所有推荐 artifact 的 `sha256` 是 64 位 hex，不是 TODO；
- 三源 size 和 sha256 一致；
- Mac Q4_K_M imatrix artifact 标为 `promote_ready` 前 smoke 通过；
- NVIDIA NVFP4 artifact 的 status 不得夸大；
- BF16 reference 不得被推荐给普通首跑用户；
- 9B thinking-on 32K GPQA 未完成时只能写 `in_progress`；
- 35B 不进入 9B 首发下载主路径；
- MTP 不计入 TPS credit。
