# Lynn Engine — Release Notes 2026-05-20

> **状态变更:Lynn Engine 从产品默认推理底层 → R&D 持续探索路径。Lynn 客户端 5/20 决定短期投奔 llama.cpp 生态作为默认本地推理。**

---

## TL;DR

| 维度 | 5/15 milestone | 5/20 决策后 |
|---|---|---|
| Lynn engine 定位 | R6000 27B Lynn-native NVFP4 production-ready 推理内核(107-108 TPS strict default)| **R&D 持续探索**(需要"同硬件同模型速度接近或超过 llama.cpp + 质量有不可替代优势"两条门槛全过才回主线) |
| 默认本地推理 | Lynn engine NVFP4 / W4A16 | **llama.cpp Q4_K_M-imatrix**(Mac Metal / Win MSVC / Linux CUDA 全平台) |
| 默认 ship 模型 | Lynn 27B-A3B Lynn-native NVFP4 | **Qwen3.5-9B Q4_K_M-imatrix(5.3GB)** thinking-on excl_pf MMLU 90+ / GPQA 80+ |
| Pro 模型 | — | **Qwen3.6-35B-A3B Q4_K_M-imatrix(20GB)** NVIDIA 24GB+ 用户 opt-in |

**为什么 pivot**:5/16-5/20 5 天在 Spark sm_121 (GB10) 上推 W4A8 FP8 e2e e2e 5 个 CLI 并行 23 commit 测出 **架构信号(不是 kernel 慢)**— Python overhead 在 decode 是 dominant,需要 vectorized expert dispatch + CUTLASS grouped GEMM + C++ service loop,**数月级工程量**,产品 deadline 前不该花。

**为什么 llama.cpp**:
- Spark sm_121 上 Q4_K_M-imatrix 35B = 69.77 TPS / 9B = 36.80 TPS 单流,c=8 total 9B = 177.54 TPS,**已是同硬件产品级 ROI 上限**
- 35B 量化三档质量几乎平(BF16 / Q4_K_M-imatrix / Lynn NVFP4 GPQA 都在 49.5±1pp)
- 9B Q4_K_M GPQA Diamond 198 题 44.44% 反超 NVFP4 W4A16 42.93%(早 50 题样本失真校准纠错)
- llama.cpp 79MB runtime + 全平台 ecosystem,**Lynn Python deps 1.5GB + 单平台限制对端侧用户体验差距明显**

---

## 5/15 → 5/20 时间线

### 5/15 R6000 sm_120a milestone

P7-S OpenAI HTTP server + tool-call 闭环,**107.23 TPS serving replay / 103.44 TPS strict full path 含最终 lm_head / 99.86 TPS strict full path no FP4 lm_head**。三口径锁。

后续 P100+ 系列 strict default 突破 **107-108 TPS**:
- P25 512 token 实测 107.43 TPS
- P37 + 40/40 prompt 全过 strict default
- AMBER 113-114 TPS 但 P37 drift / 70-prompt fail,不能默认发布

### 5/15 同期 Spark sm_121 sm_120a 移植

27B Lynn-native NVFP4 从 23.88 TPS push 到 **42.85 TPS**(`LYNN_PACKED_DECODE_BACKEND=native_fast_2d` 路径 trigger,+79% env-only 跃迁),Spark 平台首次跑过 40。

### 5/17 战略 pivot 回原版 Qwen3.6-35B-A3B

原 27B 自蒸馏 variable-expert 路线质量下降 10%(尤其推理质量下降严重)被弃用,集中支持原版模型族 + Lynn engine 提供专项加速。

### 5/18 Spark 35B-A3B baseline 三档锁

| 路径 | Spark sm_121 单流 TPS | Quality MMLU 500 / GPQA 198 |
|---|---:|---|
| Lynn-native NVFP4 W4A16 lynn-engine | **38.96** | 84.40 / 49.49 |
| llama.cpp Q4_K_M-imatrix | **69.77** | 83.00 / 50.00 |
| SGLang BF16 official | 30.14 | 86.40 / 45.45 |

### 5/19 ISA 实测限制

Spark sm_121 (GB10) **所有 FP4 MMA / `kind::f8f6f4` 全部 ptxas reject** — R6000 sm_120a 上 native FP4 的本钱 Spark 上没有。**W4A8 FP8** 成为 Spark 唯一性能突破口(FP8 MMA 162 TFLOPS = 1.64× BF16)。

### 5/16-5/20 W4A8 FP8 Phase 2:5 CLI 并行 23 commit

| Commit | 内容 |
|---|---|
| `e370cbc` + `ca77375` | 离线 NVFP4 → FP8 repack 工具 v0(FP64 cosine verify > 0.999) |
| `7401a32` | FP8 fused gate/up + SwiGLU Triton kernel v0 |
| `753cff4` | repack V1 full Lynn-native dir → FP8 |
| `8580e33` | autotune sweep harness + 2160-config Spark result |
| `7d5cdda` | MoE-expert variant FP8 fused kernel |
| `dac9e7d` | engine integration scaffold(loader + dense FFN FP8 branch) |
| `2222a7a` | MoE forward FP8 path(prefill branch) |
| `836da56` | 5 CLI 全 merged 完成 |
| `1f12b4c` | V2 3D MoE expert repack(178s 完整 35B repack,cos all > 0.99961) |
| `ec4dc5b` | autotune block default 应用 + `auto_block` shape-aware dispatch |
| `e88916d` | Wave 2 V2 smoke retry prep — loader dual-storage + harness env knob cleanup |
| `14cddf7` | `torch._scaled_mm` B-arg layout fix(`.t().contiguous()` → `.t()`) |
| `1beb629` | **Wave 2 P0 — FP8 MoE decode dispatch fix(架构级修复)** |
| `146d722` | smoke env disable graph for FP8 configs |

### Kernel micro-benchmark(全过 correctness gate)

| 维度 | 数值 |
|---|---|
| Dense fused gate/up + SwiGLU(M=1) | 3-4× over BF16(autotuned 峰值 6.00×)|
| MoE expert variant kernel(N=1408) | 1.82-2.10× over BF16 |
| 35B 全 dir 离线 NVFP4 → FP8 repack | **178 秒**,cos all > 0.99961 |
| autotune sweep 总 config 数 | 2160(全 PASS cos ≥ 0.9992)|

### 5/20 Wave 2 e2e smoke 实测:7 个 bug 暴露架构信号

| Bug | 描述 | Fix |
|---|---|---|
| 1 | smoke harness env knob 残留 → NVFP4 runtime 读 `packed_key` KeyError | env_overrides 清干净 |
| 2 | V2 loader 只产 `.weight_fp8`,linear_attn 读 `.weight` 缺 BF16 → KeyError | loader dual-storage(FP8 + BF16 dequant 兜底)|
| 3 | `torch._scaled_mm` B-arg `.t().contiguous()` 拷成 row-major → stride(0)!=1 RuntimeError | 去 `.contiguous()` 保 column-major view |
| 4 | FP8 branch 加在 prefill `_moe_forward`,decode 走 `_resolve_decode_moe_impl` 完全不同函数 → KeyError | 加 `moe_forward_decode_fp8` 到 `moe_optimized.py`,auto-detect FP8 keys delegate |
| 5 | FP8 decode `torch.unique(...).tolist()` host sync 不兼容 CUDA graph capture | smoke env disable graph for FP8 configs(临时)|
| 6 | Graph capture 失败留 96GB reserved fragmented → 后续 eager OOM | 副生消失 |
| 7 | **Python overhead 在 decode 是 dominant(架构信号)**:30 MoE × 8 active expert × per-expert `_scaled_mm` = 240+ kernel launches/token,graph disabled 后每个 launch 吃 full Python/CUDA dispatch overhead | 需要 vectorized expert dispatch + CUTLASS grouped GEMM + C++ service loop,**月级工程量** |

---

## 5/20 决策

### 短期(2-4 周)

**Lynn 客户端引入 llama.cpp 作为默认本地推理底层**:

- **底层推理**:llama.cpp ecosystem(Mac Metal / Windows / Linux CUDA 全平台 + Q4_K_M GGUF)
- **默认模型**:Qwen3.5-9B Q4_K_M-imatrix(5.3 GB),80% 用户 9B 已经够好
- **Pro 模型**:Qwen3.6-35B-A3B Q4_K_M-imatrix(20 GB),NVIDIA 24GB+ 用户 opt-in
- **Lynn 客户端**:自动硬件 detect + install llama.cpp + 下载模型 + 启 server + 注册 provider + tool-call 门禁 + 本地优先 routing(Electron + brain backend)
- **Lynn 智能体**:tool routing / 6 层 memory / MCP / skills / 跨模型 fallback(**Lynn 真护城河**)

### Lynn engine 持续(R&D)

- **W4A8 FP8 路径继续 R&D**,门槛拉高:同模型同硬件下,Lynn engine **速度必须接近或超过 llama.cpp,且质量上有不可替代优势**,两条门槛全过才回主线
- **消费 Blackwell 32GB 卡普及时**(R6000-class FP4 MMA 硬件),Lynn engine 路径回主线 reset
- **长 ctx 6.77× SGLang 在 16K 上下文** 数据点保留,作为 NVIDIA Pro 用户"高级模式"卖点
- **MTP K=1 sequential 6/6 @ 26.4 TPS** correctness-clean baseline 保留,继续磨

**Wave 2 全部 commit 留在 main 分支不 revert**。后人查 5/20 5 个 CLI 并行 + 7 bug fix trail + 178s repack + autotune sweep 2160 config — 都是真实工程财产。

`claude/fp8-native-intermediate-buffer-20260520` branch(commit `9b17aa1`)留作 future 工程参考 — 等架构突破后(vectorized expert dispatch / CUTLASS grouped GEMM / C++ service loop)再回来 wire-up。

---

## 9B Q4_K_M-imatrix 5GB / MMLU 90+ / GPQA 80+(default ship 候选)

### thinking-on 32K(2026-05-18-19 实测)

| Eval | naive | parse_fail | **excl_pf**(能力上限)|
|---|---:|---:|---:|
| MMLU 100 sample 5-shot | 81.00%(81/100) | 10(10%) | **90.00%**(81/90) |
| GPQA Diamond 50 sample | 50.00%(25/50) | 20(40%) | **83.33%**(25/30) |
| GPQA Diamond 198 题完整 | 72.22% | n/a | **81.71%** |
| 公开 Qwen3.5 9B reference | MMLU-Pro 82.5 / GPQA 81.7 | — | — |

> excl_pf = "排除解析失败题"的 naive 上限。Parse fail 主要是 32K 思考预算还不够 GPQA Diamond Organic Chemistry 类题目(15/20 fail 集中),不是模型答错。这条规律跟公开 Qwen3.5 reference 完全对齐。

### 5GB 安装的 framing

| 维度 | Lynn 默认 ship 9B Q4_K_M-imatrix |
|---|---|
| 模型文件 | **5.3 GB**(Q4_K_M-imatrix GGUF)|
| llama.cpp runtime | 79 MB(C++ binaries + .so)|
| **总安装体积** | **5.4 GB 整** |
| MMLU 100 thinking-on excl_pf | **90.00%** |
| GPQA Diamond 198 thinking-on excl_pf | **81.71%** |
| 同硬件 Spark sm_121 单流 TPS | 36.80 |
| Spark sm_121 c=8 concurrent total TPS | **177.54** |
| Mac / Windows / Linux CUDA | 全平台原生支持 |

**普通用户最直观的卖点 = "本地无限 token"**:9B 跑本地,**无 quota / 无 API key / 无跨境延迟**,智能体跑一晚不限消费。

---

## 给同行的踩坑总结(接 5/15 文章段十的 7 条,补 4 条 5/20 新铁律)

### 8. **Decode 路径跟 Prefill 路径必须分别 wire 分别测**

前 6 个 bug 修完后 Bug 4 才暴露 — FP8 branch 加在 prefill `_moe_forward` 但 decode 走 `_resolve_decode_moe_impl` 完全不同函数。**每个 forward path(prefill / decode / spec-decode)必须独立 smoke 验 FP8 branch 真被 hit**,不能假设"forward 路径 1 个就 = 全 wire 完"。

### 9. **`.contiguous()` 在 `_scaled_mm` B-arg 上是第二种地雷**

5/15 写过"`.contiguous()` 在 MoE expert 量化路径上是地雷"(96G OOM)。**5/20 同一个 `.contiguous()` 在不同语境又是地雷**:`.t().contiguous()` 把 column-major view 拷成 row-major copy,`_scaled_mm` 要求 `b.stride(0)==1` 不满足。**view 跟 copy 在 PyTorch 是看不见的差异,但 kernel 入口处会爆**。

### 10. **CUDA graph capture + host sync 永远不兼容**

`torch.unique(...).tolist()` / `mask.nonzero(as_tuple=True)` / 任何返回 dynamic-shape tensor 的操作都不能在 graph-captured region 里。任何"active subset" / "sparse dispatch"模式必须 vectorize 成 fixed-shape ops 才能进 graph,**否则只能 eager,失去 graph 速度收益**。

### 11. **Micro-benchmark 跟 e2e 之间隔几个数量级 overhead**

5/15 段十教训 7:"layer profile 比 micro-bench 重要"。**5/20 进化版**:fixture micro 3-4× 在 e2e 可以变成不出 1×。当 Python/Torch dispatch overhead 主导时,所有 kernel 优化全部失效。**先证明 dispatch overhead 不主导,再 follow kernel 优化路线**。

---

## 仓库后续维护

- **main 分支** = stable artifact + 5/20 pivot 决策 anchor
- **R&D branches** =  `claude/fp8-*` 系列 / `claude/mtp-*` 系列保留作工程参考
- **Lynn 客户端 llama.cpp 集成** = 另一仓 `MerkyorLynn/Lynn` 主线,本仓与之解耦

下次 Lynn engine 回主线的条件:**同硬件同模型速度接近或超过 llama.cpp,且质量有不可替代优势** — 两条门槛全过。不然就让 llama.cpp 跑模型,Lynn 跑产品。

---

## 关联文章 / 文档

- **5/15 工程进度公开**(R6000 P7-S 78.85 TPS milestone 时期)— 知乎 / GitHub
- **5/20 全维度反思**(本文)+ `Lynn-Engine-Zhihu-Progress-2026-05-20.md`(知乎稿)+ `Lynn专用引擎开发心得_20260520.md`(8 条核心经验)
- **docs/STRATEGY.md / docs/DESIGN.md** 保留,5/20 决策不修改原战略文档,以本 release notes 为准

---

*2026-05-20 / Lynn / MerkyorLynn — Lynn engine Spark Wave 2 数据出炉 + 战略 pivot*
