# Qwen3.5-9B Mac 首跑 Runbook · 2026-05-19

## 目标

让 Mac 用户从下载 Qwen3.5-9B Q4_K_M imatrix GGUF 到本地智能体接入，用最短路径跑通首个 OpenAI-compatible endpoint。

口径与首发官网文案保持一致：Mac 稳定轨是 Qwen3.5-9B Q4_K_M imatrix GGUF + llama.cpp / LM Studio；NVIDIA 轨另走 Lynn Engine + NVFP4 W4A16。

## 1. 推荐下载

推荐 Mac 用户下载：

```text
Qwen3.5-9B Q4_K_M imatrix GGUF
```

建议本地目录：

```text
~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf
```

三源下载布局：

| Source | Purpose | Layout |
|---|---|---|
| `dl.merkyorlynn.com` | 官方大文件下载源 | `https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/` |
| Hugging Face | 海外镜像 / 社区下载 | `TODO_HF_QWEN35_9B_Q4KM_REPO` |
| ModelScope | 国内镜像 / 备用下载 | `TODO_MODELSCOPE_QWEN35_9B_Q4KM_REPO` |

占位命令：

```bash
mkdir -p ~/Models/Lynn/Qwen3.5-9B/q4_k_m

# dl.merkyorlynn.com
curl -L --fail --continue-at - \
  --output ~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf

# Hugging Face
huggingface-cli download TODO_HF_QWEN35_9B_Q4KM_REPO \
  Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  --local-dir ~/Models/Lynn/Qwen3.5-9B/q4_k_m \
  --local-dir-use-symlinks False

# ModelScope
modelscope download --model TODO_MODELSCOPE_QWEN35_9B_Q4KM_REPO \
  Qwen3.5-9B-Q4_K_M-imatrix.gguf \
  --local_dir ~/Models/Lynn/Qwen3.5-9B/q4_k_m
```

## 2. 校验下载

下载完成后先校验，再启动 runtime。

占位元数据：

```text
file: Qwen3.5-9B-Q4_K_M-imatrix.gguf
size_bytes: TODO_Q4KM_IMATRIX_SIZE_BYTES
sha256: TODO_Q4KM_IMATRIX_SHA256
```

手动校验：

```bash
GGUF=~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf
ls -lh "$GGUF"
shasum -a 256 "$GGUF"
```

使用 `checksums.sha256`：

```bash
cd ~/Models/Lynn/Qwen3.5-9B
shasum -a 256 -c checksums.sha256
```

或使用仓库里的校验脚本：

```bash
bash scripts/local_qwen35_9b_verify_checksums.sh \
  --root ~/Models/Lynn/Qwen3.5-9B \
  --manifest ~/Models/Lynn/Qwen3.5-9B/checksums.sha256 \
  --out reports/qwen35_9b/local_checksum_verify.json
```

常见损坏排查：

| Symptom | Likely cause | Fix |
|---|---|---|
| 文件大小明显小于发布值 | 下载中断 | 使用 `curl --continue-at -` 或重新下载 |
| `shasum` 不匹配 | 下载损坏或文件版本不一致 | 删除本地文件，重新从同一源下载 |
| `llama-server` 报 GGUF parse error | 文件不完整或不是 GGUF | 检查扩展名、大小、sha256 |
| 校验 manifest 缺文件 | 目录结构不一致 | 确认文件在 `q4_k_m/` 下，manifest 相对路径匹配 |

## 3. llama.cpp 启动

Apple Silicon 推荐使用带 Metal 的 llama.cpp build。NVIDIA / CPU 路径不是本 runbook 的主轨：NVIDIA 用户首发推荐 Lynn Engine + NVFP4，CPU 只作为 fallback。

安装：

```bash
brew install llama.cpp
```

或自行构建 Metal 版本：

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_METAL=ON
cmake --build ~/llama.cpp/build -j
export PATH="$HOME/llama.cpp/build/bin:$PATH"
```

启动 OpenAI-compatible server：

```bash
GGUF=~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf

llama-server \
  -m "$GGUF" \
  --host 127.0.0.1 \
  --port 8080 \
  -c 32768 \
  -ngl 99 \
  --jinja \
  --reasoning auto \
  -a qwen35-9b-q4km
```

默认 endpoint：

```text
base_url: http://127.0.0.1:8080/v1
model: qwen35-9b-q4km
```

## 4. OpenAI-compatible smoke

检查模型列表：

```bash
curl -fsS http://127.0.0.1:8080/v1/models
```

检查 chat completion：

```bash
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-9b-q4km",
    "messages": [
      {"role": "user", "content": "Say OK in one short sentence."}
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

通过标准：

- `/v1/models` 返回 HTTP 200；
- chat 返回非空文本；
- 不把这一步当作正式 TPS benchmark。

## 5. LM Studio 用户步骤

1. 下载 Qwen3.5-9B Q4_K_M imatrix GGUF。
2. 校验文件大小和 sha256。
3. 在 LM Studio 中导入本地 GGUF。
4. 选择合适 context size；需要长上下文时确认内存足够。
5. 启动 Local Server。
6. 在客户端中使用 LM Studio 显示的 OpenAI-compatible base URL。
7. 运行 `/v1/models` 和 `/v1/chat/completions` smoke。

推荐 model name：

```text
qwen35-9b-q4km
```

## 6. 本地智能体接入

### Claude Code

示例配置：

```text
base_url = http://127.0.0.1:8080/v1
api_key = local
model = qwen35-9b-q4km
```

### Cline

选择 OpenAI-compatible provider：

```text
Base URL: http://127.0.0.1:8080/v1
API Key: local
Model ID: qwen35-9b-q4km
```

### OpenCode

示例：

```text
provider: openai-compatible
base_url: http://127.0.0.1:8080/v1
api_key: local
model: qwen35-9b-q4km
```

不同客户端字段名可能略有差异；保持 base URL 指向 `/v1`，model name 与 `llama-server -a` 一致即可。

## 7. thinking-on 32K GPQA 定位

thinking-on 32K 是能力模式，用来观察长上下文和推理能力，不等于短答 TPS benchmark。

不要把 thinking-on 32K GPQA 的进行中结果写成最终完整分数；也不要用它替代短答、工具调用、JSON 输出等本地智能体 smoke。

## 8. Troubleshooting

### Context 太小

Symptom：长任务提前截断、提示上下文不足。

Fix：启动时使用 `-c 32768`，并确认 Mac 内存足够。LM Studio 中也需要设置足够 context。

### 模型输出太长

Symptom：回答不断续写、延迟变高。

Fix：在客户端设置 `max_tokens`，先用 32 到 256 tokens 做 smoke，再增加生成长度。

### 端口占用

Symptom：`llama-server` 启动失败，提示 bind error。

Fix：

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
kill <pid>
```

或换端口：

```bash
llama-server -m "$GGUF" --host 127.0.0.1 --port 18099 -c 32768 -ngl 99 --jinja --reasoning auto -a qwen35-9b-q4km
```

### 内存不够

Symptom：进程被系统杀掉、加载失败、系统明显卡顿。

Fix：关闭其他大内存应用；降低 context；确认下载的是 Q4_K_M imatrix GGUF，而不是 BF16 权重。

### 速度慢

Symptom：首 token 慢或 decode 低于预期。

Fix：确认使用 Apple Silicon Metal build；确认 `-ngl 99`；关闭其他占用 GPU/CPU 的任务；先用短 prompt smoke，不要直接跑 thinking-on 32K。

### GGUF 损坏

Symptom：加载时报 GGUF parse error 或 checksum mismatch。

Fix：重新下载，重新校验 `size_bytes` 和 `sha256`，不要混用不同来源的 partial 文件。
