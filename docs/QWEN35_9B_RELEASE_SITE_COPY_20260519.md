# Qwen3.5-9B 首发官网文案草案 · 2026-05-19

## 首屏定位

### Hero

**Lynn Engine 本地推理，从 9B 开始。**

面向个人设备和本地智能体的首发模型线：Mac 用户走稳定的
Qwen3.5-9B Q4_K_M imatrix GGUF + llama.cpp；NVIDIA 用户走 Lynn Engine +
Lynn-native NVFP4 W4A16。

### Sub-hero

- **Mac 稳定轨**：下载 GGUF，使用 llama.cpp、LM Studio 或 CLI 启动本地 OpenAI-compatible endpoint。
- **NVIDIA 性能轨**：下载 NVFP4 W4A16 包，使用 Lynn Engine 一键脚本启动本地服务。
- **统一校验**：下载后先验证 `checksums.sha256`，再运行 smoke。

### Primary CTAs

| CTA | Target |
|---|---|
| Download Qwen3.5-9B | `https://dl.merkyorlynn.com/models/qwen35-9b/` |
| Read install guide | `https://engine.merkyorlynn.com/docs/qwen35-9b/quickstart` |
| View model card | `https://engine.merkyorlynn.com/docs/qwen35-9b/model-card` |

## 安装矩阵

| 用户 | 推荐路径 | 文案 |
|---|---|---|
| Mac / Apple Silicon | Q4_K_M imatrix GGUF + llama.cpp | 最稳的本地首发路径。下载 GGUF，用 llama.cpp、LM Studio 或 CLI 启动。 |
| NVIDIA Linux / Blackwell | Lynn Engine + NVFP4 W4A16 | 性能轨。下载 Lynn-native NVFP4 W4A16，用 Lynn Engine 一键脚本启动。 |
| Windows NVIDIA | WSL2 first / native roadmap | 首发优先支持 WSL2 或 Docker；native Windows 作为 roadmap，不承诺首发。 |

## Mac 安装文案

### llama.cpp

```bash
brew install llama.cpp
llama-server \
  --model ~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 18099 \
  --ctx-size 32768 \
  --jinja \
  --reasoning auto
```

推荐给希望自己控制 endpoint、端口和上下文长度的用户。

### LM Studio

1. 下载 Q4_K_M imatrix GGUF。
2. 在 LM Studio 中导入本地 GGUF。
3. 启动 Local Server。
4. 在客户端中使用 LM Studio 暴露的 OpenAI-compatible base URL。

推荐给希望图形界面启动和测试的 Mac 用户。

### CLI

```bash
MODEL=~/Models/Lynn/Qwen3.5-9B/q4_k_m/Qwen3.5-9B-Q4_K_M.gguf \
bash scripts/local_qwen35_9b_first_run.sh
```

推荐给希望一条命令完成发现、启动和 smoke 的用户。

## NVIDIA Linux 安装文案

首发 NVIDIA 路径使用 Lynn-native NVFP4 W4A16。

```bash
curl -fsSL https://engine.merkyorlynn.com/install/qwen35-9b-nvidia.sh | bash
```

脚本应做的事：

1. 检查 Linux + NVIDIA 环境。
2. 提示下载 NVFP4 W4A16 包。
3. 验证 `checksums.sha256`。
4. 启动 Lynn Engine OpenAI-compatible endpoint。
5. 运行 `/health` 和 chat smoke。

> 发布前需要把脚本地址替换为真实安装入口；页面文案不应暗示脚本已经覆盖所有发行版。

## Windows 文案

Windows NVIDIA 首发标注为 roadmap。当前推荐：

- WSL2 Ubuntu + NVIDIA 驱动，按 NVIDIA Linux 路径安装；
- 或 Docker + NVIDIA Container Toolkit；
- native Windows binary 不作为首发承诺。

## 不夸大声明

- 9B thinking-on 32K GPQA 仍在长跑中；不要声明最终 198 题完整分数。
- 35B 是支线，不是本次 9B 首发主路径。
- MTP 不计入首发 TPS credit；首发吞吐只引用主模型解码路径。
- W4A8 / FP4xFP8 resident 是实验轨，不是默认安装路径。
- NVIDIA compact NVFP4 只能标注为 candidate，不能标注 stable。

## 页脚说明

- 官网 / 文档入口：`engine.merkyorlynn.com`
- 大文件下载源：`dl.merkyorlynn.com`
- 所有下载文件必须配套 `checksums.sha256`。
- 模型下载完成后先校验，再启动 runtime。
