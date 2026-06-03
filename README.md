# Lynn Engine

> **🚀 战略校正(2026-06-03):Lynn engine 重启为并行主线,目标「对标 llama.cpp」,不再是"降级为 R&D / 只用 llama.cpp"。** 客户端短期仍以 llama.cpp/GGUF 为**务实默认后端**;引擎并行推进——同模型同硬件**对标 llama.cpp**,终局 = 自己啃下**融合 4-bit / 零-shadow 内核**(单投影 PoC → 全 dense + 删 shadow → MoE grouped 专家 → 融合减 launch,每阶段 gate + RC),在 FP4-MMA 卡(R6000 一代)上**逼近乃至超过**。**做引擎,要做就自己把内核啃下来。**
>
> **✅ 自有核心已 bank 的最新突破:** 35B NVFP4 服务已把 decode 阶段 BF16 dequant-shadow 从常驻内存中释放,**resident 88→28 GiB(省 ~60 GiB)**,token-exact,TPS 0.998× 无回归;`server/openai_http.py` 已接入 `reload→prefill→release→decode` 服务循环。**P0.1/P0.2 packed-prefill gates 已过:**释放 60GiB shadow 后不调用 reload,`stream_bf16` 证明 token-exact no-reload prefill(peak **40.28 GiB**,proof prefill **20.75s**);P0.2 resident inventory 显示 after-release BF16 只剩 **4.72 GiB**。**2026-06-04 P1 单 dense projection PoC 已过:**真实 `linear_attn.in_proj_qkv` 从 packed E2M1 + FP16 scale 直接跑 Triton matvec,不读 BF16 shadow,numeric PASS,32.00→10.00 MiB(3.20×),**160.29us vs BF16 190.03us = 1.186×**。**P1-A naive + tiled scalar batched bridges 均已反证:**数值/no-shadow 过,但 M>1 仍输 BF16 tensor-core GEMM。**P2 grouped MoE 已推进到 P2-I:**P2-F 已接入 engine opt-in mode `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`;P2-H 进入完整 `_prefill_layer` 链路(RMSNorm + linear/full attention cache + MoE),mixed L0-3 T16 **45.97ms vs BF16 58.57ms = 1.274x**,比 `stream_bf16` 快 **52.09x**;P2-I 扩到 mixed L0-7 T16 **88.96ms vs BF16 113.82ms = 1.279x**,比 `stream_bf16` 快 **46.70x**,numeric pass,peak **5.123GiB vs stream 17.102GiB**。下一关是 P2-J linear-attn prefill trace。Python 只做控制面和验证,追赶 llama.cpp 靠 CUDA C++/CUTLASS/native kernel。

> **🆕 2026-06-03 Decode 内核启动开销战役 — Spark NVFP4 35B-A3B 单流 38.96 → ~45 TPS,质量 RC 等价。**
> 实测 decode 是 launch-bound(census:**~1527 CUDA launch/token**,~40% 时间耗在 CPU 端 dispatch)。逐簇融合 launch + 消拷贝:**fused RMSNorm(最大头)/ shared-expert / linear-attn g/beta-fold / full-attn(token-exact)/ NVFP4 `_scaled_mm` bf16-out copy-elision**,**5 个 RC-validated launch-cut**——在 structured/V9/GPQA/tool-call/long-form 上 **40/40 greedy 输出与 baseline 逐字一致**,继承 **MMLU 84.40 / GPQA-Diamond 49.49**。全部 gated、默认安全、可回滚。
>
> **🎯 关于对标 llama.cpp(口径已据 6/3 evidence-lock 校正)。** 同硬件 Q4_K_M **69.77** 领先 ~1.5×。**起初以为根因是 BF16 dequant-shadow 的 ~2× 带宽墙、写个零-shadow 内核就能把墙从 ~40 推到 ~140;6/3 实测(2 个无头 CLI 代码 trace + 4 个 Spark 探针)把这个前提证伪了**:read-4bit 其实已做(MoE 专家走「packed-4bit→寄存器反量化→bf16 GEMV」Triton 核)、那 60 GiB BF16 是 **prefill 专用**(decode 整块删掉照跑、TPS 不降 42.4→43.7)、把 attn 改读 FP4 **无收益甚至更慢**(full-attn 0.999× / linear out_proj 0.775×)、reusable decode CUDA graph **净负 0.75×**。→ **decode 是 launch-bound,且 Spark sm_121(无 FP4 MMA)结构性卡 ~45。** 与 69.77 的差距是 llama.cpp 手写、低-dispatch、成熟度极高的 ggml CUDA —— 需 ground-up 内核重写 + 最终 FP4-MMA 硅(R6000 已退租),**不是 Spark 的交付目标**。本轮 bankable 红利:**decode-only 删 shadow → 常驻 87→27 GiB**(腾 60 GiB 给 KV/长上下文/batch)。跨设备内核 moat(同套 NVFP4 权重挪 FP4-MMA 卡变 native)逻辑仍在,有那张卡时才兑现。详见 [decode launch-overhead campaign](reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md)。

> **🆕 2026-05-28 Qwen3.6-35B-A3B update — Spark 已切到最快 APEX-MTP I-Balanced 单流路线。**
> 当前 `lynn-apex-mtp-llamacpp.service` 使用
> `Qwen3.6-35B-A3B-APEX-MTP-I-Balanced.gguf` + `--spec-type draft-mtp --spec-draft-n-max 4`。
> 短 A/B:单流 **77.01 tok/s** vs AR **60.65 tok/s**(+27%);生产 sanity 中位 **76.19 tok/s**。
> 既有 32K thinking-on 质量锚点:**MMLU500 90.00% / GPQA198 78.79% naive,83.87% excl parse fail / tool-call 12/15**。
> 详见
> [APEX-MTP service A/B](reports/mtp/LLAMA_CPP_APEX_MTP_SERVICE_AB_20260528.md),
> [32K quality refresh status](reports/qwen36_35b/APEX_MTP_QUALITY32K_REFRESH_STATUS_20260528.md),以及
> [知乎/公众号草稿](reports/articles/QWEN36_35B_A3B_QUALITY_SPEED_OPTIMAL_PATH_20260528.md)。

> **🆕 2026-05-27 Active R&D update — Lynn engine 没有放弃,正在把 Nemotron-style self-spec 的可移植部分落到 Qwen35 APEX-MTP。**
> (历史记录;最新口径以顶部 6/3 重启校正为准)当时客户端默认走 llama.cpp/GGUF,engine 研发线持续推进:
> **Qwen3.6-35B-A3B W4A16 + official APEX/MTP sidecar 已跑通 K=2 verify/accept/crop/full-accept/prefix-repair token-exact smoke**。
> 当前 blocker 不是算法控制流,而是 K2 verifier 的 T=1-equivalent attention / `o_proj` kernel 成本。详见
> [5/27 active status](reports/mtp/LYNN_ENGINE_ACTIVE_RESEARCH_STATUS_20260527.md) 和
> [Qwen35 MTP block verifier log](reports/mtp/QWEN35_MTP_BLOCK_VERIFY_OVERNIGHT_20260526.md)。
>
> 简单说:产品 fallback 先用 llama.cpp,但 Lynn engine 主线正在继续吃 APEX-MTP / K=N / self-spec 这条硬路线。

> **🆕 2026-05-20 状态变更(⚠️ 已被顶部 6/3 重启校正取代)— 当时把 Lynn engine 降级为 R&D 持续探索路径;现引擎已重启为对标 llama.cpp 的并行主线。**
> Lynn 客户端短期投奔 llama.cpp 生态作为默认本地推理底层(Mac Metal / Win MSVC / Linux CUDA 全平台 + Q4_K_M GGUF)。
> 默认 ship 模型 = **Qwen3.5-9B Q4_K_M-imatrix(5.3GB)** thinking-on excl_pf MMLU 90+ / GPQA 80+。
> 历史决策见 [5/20 Release Notes](./RELEASE_NOTES_20260520.md);当前权威状态见 [6/3 Restart Notes](./RELEASE_NOTES_20260603.md)。
>
> Lynn engine 工程财产全留(5 CLI 并行 + 7 bug fix trail + 178s repack + autotune sweep 2160 config 都是真东西)。回主线门槛:**同硬件同模型速度接近或超过 llama.cpp,且质量有不可替代优势**。
>
> ⚠️ 以下 5/16 状态文档保留作历史进度记录,**最新状态以 RELEASE_NOTES_20260603 为准**。

---

> **为 NVIDIA Blackwell 写的 Lynn 27B-A3B NVFP4 单模型推理引擎。**
> 从零写,锁定 Lynn 自家的 variable-pruned MoE + NVFP4 格式,目标很窄也很硬:在 R6000 / Spark 这类 Blackwell 机器上,把 Lynn 27B A3B MoE 基座跑成可生产、可优化、可长期接管的推理内核。

[Read in English](README_EN.md) · [📝 6月知乎连载:从零开始 Qwen 3.6 35B-A3B 写专用推理引擎踩坑心得分享](https://zhuanlan.zhihu.com/p/2045562329396400486) · [战略文档](docs/STRATEGY.md) · [架构设计](docs/DESIGN.md) · **[🆕 6/3 Restart Notes](RELEASE_NOTES_20260603.md)** · [P1 dense projection PoC](reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md) · [P1-A tiled sweep](reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md) · [P2 grouped MoE census](reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md) · [P2-G multi-layer MoE smoke](reports/stage6/P2G_MULTILAYER_MOE_SMOKE_20260604.md) · [P2-H selected-layer prefill smoke](reports/stage6/P2H_SELECTED_LAYER_PREFILL_SMOKE_20260604.md) · [P2-I selected-MoE expansion](reports/stage6/P2I_SELECTED_MOE_EXPANSION_SMOKE_20260604.md) · [5/20 历史 Release Notes](RELEASE_NOTES_20260520.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## 当前状态(2026-06-03)

**6/3 战略校正(取代 5/20 pivot)**:Lynn engine **重启为并行主线,目标对标 llama.cpp** —— 不再是"降级为 R&D / 客户端投奔 llama.cpp"。客户端短期仍以 llama.cpp/GGUF 作**务实默认后端**,但引擎同步推进:同模型同硬件对标 llama.cpp,终局是自己啃下融合 4-bit / 零-shadow 内核,在 FP4-MMA 卡(R6000 一代)上逼近乃至超过。6/3 实测口径(Spark sm_121 无 FP4 MMA,decode 结构性卡 ~45,parity 是 ggml 重写 / FP4-MMA 硅目标)见顶部 6/3 战役与 [6/3 Restart Notes](RELEASE_NOTES_20260603.md);5/20 决策背景仅作历史记录。

### 5/16-5/20 Spark sm_121 W4A8 FP8 Phase 2 实测信号

5 个 CLI 并行 + 23 commit 推 Wave 2 W4A8 FP8 e2e。最终 retry #4 暴露架构信号:**Python overhead 在 decode dominant**,30 MoE × 8 active expert × per-expert `_scaled_mm` = 240+ kernel launches/token,graph disabled 后每个 launch 吃 full Python/CUDA dispatch overhead。修不掉的 7 号 bug,**需要 vectorized expert dispatch + CUTLASS grouped GEMM + C++ service loop,月级工程量**。

### 35B 横向对比(Spark sm_121 GB10 单流;baseline 2026-05-18,lynn-engine 行 6/3 战役更新)

| 路径 | 模型大小 | 单流 TPS | MMLU 500 | GPQA Diamond 198 | 备注 |
|---|---:|---:|---:|---:|---|
| Lynn-native NVFP4 W4A16 / lynn-engine | 23 GB | **38.96 → ~45** | 84.40% | 49.49% | 5/18 base → **6/3 launch-overhead 战役(5 cut,RC-validated)** |
| **llama.cpp Q4_K_M-imatrix** | **20 GB** | **69.77** | 83.00% | **50.00%** | 同硬件 ~1.55× lynn-engine — 6/3 实测 Spark NVFP4 decode 结构性卡 ~45(带宽 + dispatch 杠杆均已否决);parity 需 ggml 级重写 / FP4-MMA 硅,**非 Spark 交付目标** |
| **llama.cpp APEX-MTP I-Balanced** | **25 GB** | **77.01** | **90.00% thinking32** | **78.79% / 83.87% excl_pf thinking32** | 当前 Spark 单流最快;高并发仍需 AR admission |
| SGLang BF16 official | 67 GB | 30.14 | 86.40% | 45.45% | reference |
| Lynn W4A8 FP8(工程探索期) | 35 GB | — | — | — | 架构未完成,见 RELEASE_NOTES |

**关键发现**:35B 量化三档 GPQA 几乎平(BF16 / Q4_K_M-imatrix / Lynn NVFP4 都在 49.5±1pp)。我们以前期待的 "NVFP4 GPQA 优势"在足量样本上不成立。

### 9B Q4_K_M-imatrix 默认 ship 候选(thinking-on excl_pf MMLU 90+ / GPQA 80+ / 5GB)

| 维度 | Lynn 默认 ship 9B Q4_K_M-imatrix |
|---|---|
| 模型文件 | **5.3 GB**(Q4_K_M-imatrix GGUF) |
| llama.cpp runtime | 79 MB(C++ binaries + .so) |
| **总安装体积** | **5.4 GB 整** |
| MMLU 100 thinking-on excl_pf | **90.00%**(81/90) |
| GPQA Diamond 198 thinking-on excl_pf | **81.71%** |
| Spark sm_121 单流 TPS | 36.80 |
| Spark sm_121 c=8 concurrent total TPS | **177.54** |
| Mac / Windows / Linux CUDA | 全平台原生支持 |

**普通用户最直观的卖点 = "本地无限 token"**:9B 跑本地,无 quota / 无 API key / 无跨境延迟,智能体跑一晚不限消费。

### Lynn engine 重启主线

- **Lynn engine 已重启为并行主线**,目标是同模型同硬件对标 llama.cpp,不是停在 R&D 降级状态。
- **已 bank:** 35B NVFP4 decode-only shadow release,常驻 88→28 GiB,token-exact,TPS 0.998×。
- **已过 P0.2:** release 后剩余 BF16 resident 只有 4.72 GiB,下一刀优先 projection / embed / lm_head,router 不是第一杠杆。
- **已过 P1 单投影:** `linear_attn.in_proj_qkv` packed Triton matvec numeric/no-shadow/microbench 全过,1.186× vs BF16 shadow。
- **P1-A 反证完成:** naive 与 tiled scalar batched bridge 都 numeric/no-shadow 过,但 M>1 都输 BF16 tensor-core GEMM;不进入 serving。
- **P2 census 已过:** 单层 MoE 证实 `stream_bf16` 是 20s no-reload proof 的主耗时,small-M verifier 内存干净但慢。
- **P2-A/B/C/D/E/F/G/H/I 证据:** 单 expert gate/up 输 BF16;routed active path M64=23.83ms/layer;P2-F engine opt-in mode M64=20.23ms/layer;P2-G 4 层 residual-scale MoE smoke M64=80.97ms;P2-H mixed L0-3 full prefill T16=45.97ms;P2-I mixed L0-7 full prefill T16=88.96ms,1.279x vs BF16,46.70x vs `stream_bf16`,peak 5.123GiB,numeric pass。
- **下一关:** P2-J linear-attn prefill trace;dense/MoE M>1 若要追平并超过 BF16,最终要 native FP4-MMA/CUTLASS-style bridge。
- **Spark sm_121 诚实口径:** decode 结构性卡 ~45,追 69.77 需要 native runtime + fused kernels + 最终 FP4-MMA 硅。
- **Wave 2 全部 commit 留在 main 分支不 revert** — 5 CLI 并行 + 7 bug fix trail + 178s repack + autotune sweep 2160 config 全是真实工程财产。

---

> 以下 5/15-5/16 R6000 27B Lynn-native NVFP4 工程记录保留作历史进度参考。当前公开数据 + 引擎重启口径以本节(2026-06-03)和 [RELEASE_NOTES_20260603](RELEASE_NOTES_20260603.md) 为准。

## 5/15-5/16 R6000 27B Lynn-native NVFP4 工程记录(历史)

5/15 R6000 sm_120a 上 Lynn-native NVFP4 milestone:**107-108 TPS strict default**(P25 512 实测 107.43,P37 + 40/40 prompt 全过 strict),三口径:

- **107.23 TPS** serving replay
- **103.44 TPS** strict full path 含最终 lm_head(真过 100 的硬数)
- **99.86 TPS** strict full path no FP4 lm_head

详细 P3 → P108 系列时间线(118.73 TPS R6000 → A100 W4A8 Recovery → SM120a FP4 contract → W4A8 hardware route):

| 项目 | 状态 |
|---|---|
| **27B A3B final BF16** | ✅ Recovery step5000 final 已 merge,structural validation PASS,greedy sanity PASS |
| **27B A3B Lynn-native NVFP4** | ✅ 20G artifact 已生成并传到 R6000,manifest integrity PASS |
| **独立加载** | ✅ 不依赖 vLLM / SGLang / TRT-LLM / llama.cpp,直接读 safetensors + Lynn quant manifest |
| **6-prompt coherent smoke** | ✅ 中文解释 / Python / RoPE-ALiBi / 英文算术 / tool JSON / longctx 全通过 |
| **R6000 strict full path(P23)** | ✅ **118.73 tok/s**(packed NVFP4 MoE + native FP4 lm_head + active MoE retune) |
| **serving replay ceiling** | ✅ **123.78 tok/s**(40-layer body graph) |
| **R6000 strict default(P25-P37 后)** | ✅ **107-108 tok/s**(P25 512=107.43,P37+40/40 全过 — 真正 ship 数) |
| **P10-P15 graph slot + state refresh** | ✅ runner graph-slot strict gate,packed-resident memory 56.47GiB shadow release,state-refresh 0.79ms |
| **P85-P92 SM120a FP4 contract** | ✅ blockscaled FP4 MMA / E2M1 / CuTe tile / 真实 gate-up expert 全 PASS,**max_abs ~e-7** |
| **P95-P97 active MoE composition** | ✅ active-MoE 1.113× baseline,native_down_tile1 vs Triton 2.14× |
| **P103 W4A8 hardware route** | ✅ FP8(E4M3/E5M2) × E2M1 raw/blockscaled atoms 全 PASS;W4A8+MTP 升主线 |
| **P104-P108 A100 W4A8 Recovery** | ✅ alpha overlay folded → 3.67% → 1.79% worst drift;generation AMBER 6-prompt mean prefix 38.67/48 |
| **5/17 战略 pivot** | 27B 自蒸馏 variable-expert 路线 quality 下降 10%(尤其推理质量)→ pivot 回原版 Qwen3.6-35B-A3B |
| **5/18-5/20 Spark FP8 Phase 2** | 5 CLI 并行 23 commit;Wave 2 retry #4 暴露 Python overhead 架构信号 → 5/20 pivot 决策 |

**注:5/15 27B 主力 artifact** 为 Lynn 27B A3B variable-pruned MoE Recovery step5000(BF16 ~60G / NVFP4 ~20G,Lynn-native variable-expert format,通用框架不能直接加载)。5/17 路线 pivot 回原版 Qwen3.6-35B-A3B 后,**Lynn 27B 自蒸馏整线弃** — Lynn engine 收敛到为原版模型族提供专项加速(Spark W4A8 FP8 / R6000 NVFP4 native FP4)。

## 性能进度

**当前 production 配置**:resident dequant→BF16 slow path,no MTP(V Pro Distill 没 MTP head),单流。性能起飞要等 P4/P5 native FP4 GEMM 替掉 scalar bridge。

| 阶段 | 单 token 延迟 | t/s | 状态 |
|---|---|---|---|
| Phase 2 brute-force | ~300 ms | 2-3 | 历史基线 |
| Phase 3.1 incremental decode | ~200 ms | 5 | 历史基线 |
| P5/P6 eager Triton path | ~30-33 ms | 30-33 | P6-K/N/O |
| **P6-S resident graph smoke** | **~15-16 ms** | **63-66** | ✅ 50TPS 目标突破 |
| **P7 current serving env** | **~14.6-15.0 ms** | **66-68** | ✅ 6-prompt generate PASS |
| **P7/P8 CUDA graph ceiling** | **12.68 ms** | **78.8** | ✅ 稳定可复现 ceiling |
| P8 torch.compile spike | 12.33 ms | 81.1 | 实验信号,非产品路径 |
| **P10-M packed NVFP4 MoE** | **10.01 ms** | **99.86** | ✅ strict full path,BF16 lm_head |
| **P10-P native FP4 lm_head** | **9.67 ms** | **103.44** | ✅ strict full path,opt-in |
| **serving replay/body graph** | **9.33 ms** | **107.23** | ✅ 40-layer graph ceiling |
| **P10-U runner graph slot** | **9.69-11.26 ms** | **88.8-103.1** | ✅ 6 prompts × 3 prefixes strict PASS |
| **OpenAI server stable path(pre-P25)** | **~11.2 ms** | **88-89** | ✅ 历史稳定基线 |
| **P11 session-scoped packed resident** | — | — | ✅ BF16 shadow 81.06→24.59 GiB,decode ids exact match |
| **P12 one-shot + graph after release** | — | **79.5-83.8** | ✅ release 56.47GiB 后 graph/eager exact match |
| **P13 graph-slot generate/window** | **12.6-12.8ms replay / 60-105ms capture** | **78-79 replay / 8-14 e2e** | ⚠️ current-position strict;future window 多 prompt FAIL |
| **P14 state refresh** | **0.79ms roundtrip + 12.6ms replay** | **~70-80 projected** | ✅ copy cost green-light,implementation pending |
| **P15 correct runtime config** | **9.66ms strict / 9.33ms replay** | **103.48 / 107.23** | ✅ `LYNN_PACKED_DECODE=0`,shared BF16 保留 |
| **P16 active-MoE boundary** | **skip-active 5.75ms replay / non-MoE 4.79ms replay** | **173.8 / 208.8 upper bound** | 🔬 155TPS 需要新 grouped native-FP4 active expert kernel |
| **P17 Triton FP4 dot_scaled** | **raw gate/up shape 0.0125ms; e8m0 neutral byte=127** | compute headroom ✅ | 🔬 layout/scale contract 已定位,下一关是 per-16→group32 bridge |
| **P18 scale-contract decision** | **dot_scaled raw 0.018ms vs scalar 0.050ms** | speed ✅ / quality ❌ | 🔬 简单 e8m0 bridge 不可 ship,转 custom per-16 kernel |
| **P19 active block retune** | **8.66ms strict / 8.32ms replay** | **115.4 / 120.3** | ✅ quality-safe scheduling gain |
| **P20 unsorted router top-k** | **8.51ms strict / 8.17ms replay** | **117.6 / 122.4** | ✅ same expert set,MoE parity PASS |
| **P21 shared gate/up fusion** | **8.50ms strict / 8.15ms replay** | **117.7 / 122.7** | ✅ BF16 shared exact,small gain |
| **P22 MoE warp retune** | **8.46ms strict / 8.11ms replay** | **118.3 / 123.3** | ✅ down kernel 8 warps |
| **P23 active MoE accounting** | **8.42ms strict / 8.08ms replay** | **118.7 / 123.8** | ✅ expert-id int32 cleanup;router/topk branch ruled out |
| **P25 OpenAI server graph path** | **~9.95ms decode / 0.65s prefill** | **~99-100 decode / 87.7 wall @512 tok** | ✅ 服务态跨过 100 decode TPS |
| **P24/P26 Triton dead ends** | `tl.dot` gate/up / merged-topk gate/up | — | ❌ 质量可过但都更慢,不进默认 |
| **P27 native CUDA extension smoke** | build/load/launch | add-one 0.0047ms | ✅ R6000 sm_120 CUDA extension 地基打通 |
| **P28 native gate/up contract** | CUDA scalar gate/up | 0.035ms/layer | ✅ cosine≈1.0,契约通过;速度不 promoted |
| **P29 native down contract** | CUDA scalar down | 0.030ms/layer | ✅ cosine=1.0,active MoE 两半契约齐 |
| **P30 native active-MoE contract** | CUDA scalar gate/up + down | 0.064-0.068ms/layer | ✅ 完整 active routed expert path exact |
| **P31 native active-MoE runtime gate** | `LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar` | 1.48-1.59× MoE function speedup | ✅ opt-in exact,默认仍关 |
| **P32-P34 generate gate** | cuda_scalar full generate / graph / allowlist | 121.7TPS 但 `!` loop;graph-off/allowlist 仍 greedy drift | ❌ 不 promoted,已加 fail-loud guard |
| **P35 sorted-router graph slot** | `LYNN_ROUTER_TOPK_SORTED=1` + full-token graph slot | 12/12 strict PASS,97.7-111.5TPS replay | ✅ graph-slot 线恢复 parity |
| **P36 decode dispatch cleanup** | runner-fixed MoE/backend dispatch | 100.53 vs 100.55TPS | ✅ exact,默认保留;非 155 突破 |
| **P37 MoE block retune closed** | layer28 profile + full generate gate | candidate 94.94TPS 且 greedy drift | ❌ block-size 线收口,转 native grouped FP4 |
| **P38 multi-layer MoE wall** | layers 2/8/14/20/28/36 | full MoE mean 0.193ms/layer,active 0.112ms/shared 0.060ms | 🔬 无异常慢层,直接打 active expert kernel |
| **P39-P40 fast fixed MoE** | 固定 R6000 best MoE config | layer-level 1.079×,generate exact,100.94TPS median | ✅ 小幅默认提速,非 155 突破 |
| **P42 cuda_scalar retest** | full-attention-only allowlist | 0/3 greedy match,mean 82.49TPS | ❌ scalar bridge 不作为生产捷径 |
| **P43 shared expert triage** | fused shared BF16 expert | 0.0609→0.0556ms/layer | ✅ 保留小优化,但收益太小,不是 155 主线 |
| **P44 shortcut triage** | merged-topk / cross-expert `_scaled_mm` | 0.48× / 0.39× vs Triton | ❌ 包装级捷径关闭,不能偷 PyTorch `_scaled_mm` |
| **P45 native active-MoE ABI** | one-call CUDA contract | 0.0658ms vs Triton 0.0583ms,cosine≥0.99999988 | ✅ ABI 地基完成,不 promote |
| **P46 fused atomic probe** | one-kernel atomic accumulation | 0.1768ms vs Triton 0.0592ms | ❌ atomics 太慢且有轻微 drift,P47 转非 atomic grouped kernel |
| **P48-P50 tile-hidden down** | non-atomic CUDA down projection | isolated/decode-state 1.25-1.27×;完整 decode step 5 top1 漂移 | 🔬 kernel-level win confirmed,但 tiny accumulation drift 会翻 greedy,不 promote |
| **P51 MoE budget ladder** | top-k limit / skip shared expert | best 124.39TPS 但输出崩;top6 coherent only +1.6% | ❌ 少算专家不到 155,质量先坏,关闭捷径 |
| **P52-A/B native FP4 sensitivity** | selected gate/up `_scaled_mm` + scale decomposition | active cosine 0.976;FP8 scale contract min cosine 0.972 | ❌ PyTorch `_scaled_mm` 组合不可 ship;瓶颈锁到 per-16 scale contract |
| **P53 Triton retune review** | E2M1 decode simplification / scale hoist | LUT variant exact but avg 0.936×;scale-hoist JIT 过重 | 🔬 有局部信号,无免费 15TPS;不 promote |
| **P54 vendor-layout feasibility** | e8m0/group32 scale search | best upper-bound inter cosine 0.9869-0.9918,all fail | ❌ 直接转 ModelOpt-like scale contract 不够,主线锁 per-16 grouped kernel |
| **P55 gate/up tile-inter** | CUDA scalar `tile_inter=2` | 1.09-1.14× vs Triton gate/up,max_abs=0 | 🔬 正向 tile 形态信号,作为 P56 grouped kernel 设计点 |
| **P56 gate/up tile runtime** | `LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter` | 114.43TPS median(+15.4%)但 `!` loop / greedy mismatch | ❌ runtime 不 promote,保留 tile shape 信号 |
| **P57 vendor route inventory** | ModelOpt / CT / variable-expert audit | 当前 env 无 ModelOpt/llmcompressor;CT 只有 converter;27B 非 HF-vanilla | 🔬 官方路线可走,但需 BF16→vendor NVFP4 v2 + padding/mask |
| **P58 graph-off retest** | `cuda_tile_inter` + block graph 全关 | 28.43TPS median,greedy mismatch 仍在 | ❌ 不是 graph-only bug;scalar tile-inter 不做生产桥 |
| **P59 dual NVFP4 dispatch** | metadata-only layout classifier | Lynn-native / vendor / BF16 / unknown packed FP4 fail-loud | ✅ 一个 engine,两类 NVFP4 artifact 显式分流 |
| **P85-P87 SM120a FP4 tile contract** | CUTLASS/CuTe blockscaled FP4 MMA | E2M1 `<<2` shift + CuTe fragment layout | ✅ 非均匀 synthetic tile `max_abs=0` |
| **P88 real gate/up packed-code tile** | real Lynn 27B layer28 expert116 | production activation FP4 codes + real packed weights,neutral scale | ✅ `max_abs=0`,证明当前 packed tensor 可进 SM120 MMA |
| **P89 per-16 scale tile contract** | split16 neutral-scale MMA + 显式 per-16 scale accumulate | `rel_l2=1.0e-7`,tolerance PASS;K32 fold best rel_l2 0.0227 | ✅ 先消费当前 Lynn-native artifact;不等官方重重量化 |
| **P90 split16 gate/up kernel** | real expert116,8 gate + 8 up rows,K=2048 | median 0.0621ms,max_abs `2.38e-7`,rel_l2 `1.53e-7` | ✅ 第一版真实 full-K native FP4 gate/up row tile PASS |
| **P91 split16 row-tile sweep** | row_count 8/16/32/64 | 64 rows median `0.0443ms`,rows/ms `2892`,all tolerance PASS | ✅ 下一个设计点锁定 64-row tile |
| **P92 full gate/up expert** | 512 gate + 512 up rows,K=2048 | median `0.0502ms`,rows/ms `20401.7`,max_abs `4.77e-7` | ✅ 完整 active expert gate/up 子算子 PASS |
| **P93 top-k gate/up backend** | top-k=8,one CUDA launch,output `[8,512]` | median `0.0602ms`,quantized-ref cosine `0.9999986`,rel_l2 `0.00167` | ✅ production-shaped gate/up backend contract PASS;slightly slower than Triton so not promoted |
| **P94 active MoE composition** | P93 gate/up + packed down weighted-sum | median `0.0823ms` vs Triton `0.0818ms`,quantized-ref cosine `0.9999986` | ✅ full active-MoE end-to-end contract PASS;speed now needs fused/non-atomic scheduling |
| **P95 down backend sweep** | fixed P93 inter + down variants | native_down_tile1 median `0.0243ms` vs Triton `0.0521ms` | ✅ down half has real 2.14× headroom;P96 should compose native gate/up + native down |
| **P96 native-down composition** | P93 gate/up + native_down_tile1 | median `0.0830ms` vs Triton active `0.0810ms`,quantized-ref cosine `0.9999986` | ✅ numeric PASS / ❌ no speed win;next work must reduce gate/up or fuse scheduling |
| **P97 interval decomposition** | CUDA-event gate/down intervals | best `0.0800ms` vs baseline `0.0891ms`,speedup `1.113×` | ✅ first full active-MoE composition speed win;next gate is full-generate parity |
| **P98 split16 runtime gate** | runtime `split16_fp4 + native_down_tile1` | graph-on capture fails;graph-off greedy mismatch and `0.80×` median TPS | ❌ P93/P97 quantized-activation contract is not production BF16-activation semantics |
| **P99 activation quant strategy** | after P98 redirect | production keeps BF16 activation semantics;activation quant moves to A100 MTP/requant path | ✅ stops the wrong shortcut early,keeps W4A4 as explicit model contract |
| **P100 native-down runtime gate** | graph-off native-down-only retest | median `1.049×`,but `new_ids_all_match=false` on all prompts | ❌ not graph-only;native-down-only runtime replacement rejected |
| **P101 graph-owned state gate** | P14-C/P35/P101 authoritative state sequence | replay-only 75-85TPS class,but `same_ids=false`,min cosine `0.6536` | ❌ graph-owned mutable state drifts;do not promote |
| **P102 mixed-MMA probe** | SM120a CuTe atom compile matrix | E2M1×E2M1 controls PASS;BF16/FP16×E2M1 raw/blockscaled all FAIL | ❌ no BF16-activation + FP4-weight shortcut;W4A4 model adaptation is required for 155+ |
| **P103 W4A8 probe** | SM120a FP8 activation × E2M1 weight compile matrix | E4M3/E5M2 × E2M1 raw/blockscaled all PASS | ✅ W4A8 is a viable hardware route;near-term mainline is W4A8+MTP |
| **P104 W4A8 sensitivity** | active-MoE fake-quant E4M3/per16 | gate/up clean;full-active near gate,max rel_l2 ~3.30% on R6000 sample | ⚠️ W4A8 is trainable,not direct-promote |
| **P105 W4A8 generate gate** | 6-prompt generation fake-quant | 64-token gate: gateup 4/6 exact,full 5/6 exact,late/local drift | ⚠️ A100 Recovery required before runtime promotion |
| **P106 A100 Recovery** | expert-wise foldable intermediate alpha overlay | real-prompt 40-layer worst active-MoE rel_l2 **3.67% → 1.79%**;folded artifact W4A8-vs-original BF16 **1.7836%** | ✅ folded overlay artifact GREEN;W4A8 artifact contract,not BF16 fallback |
| Long target | <5 ms | >200 | native FP4 / larger fused blocks |

当前 R6000 推荐环境:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_MOE_GATE_BLOCK_INTER=8
export LYNN_MOE_GATE_BLOCK_HIDDEN=256
export LYNN_MOE_DOWN_BLOCK_HIDDEN=8
export LYNN_MOE_DOWN_BLOCK_INTER=512
export LYNN_MOE_GATE_NUM_WARPS=4
export LYNN_MOE_DOWN_NUM_WARPS=8
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
export LYNN_PACKED_SHARED_EXPERT=0
```

实测 final step5000 NVFP4:

```text
strict full path:      118.73 tok/s  (P23 int32 expert-id cleanup + P22/P21/P20/P19)
serving replay/body:   123.78 tok/s  (40-layer graph ceiling)
OpenAI server decode:   ~99-100 tok/s (P25 reusable block graph env,512-token wall TPS 87.7)
BF16 lm_head path:     99.86 tok/s
quality smoke:         6/6 coherent + strict tool-call + no-think loop guard PASS
```

Native FP4 lm_head 是当前的 deterministic/greedy opt-in 优化:6/6 prompt top-1 match,top-20 overlap 最低 15/20,logits cosine 最低 0.9924。采样型生产流量默认仍保留 BF16 lm_head fallback,直到更大规模 parity gate 完成。

CUDA graph 说明:P10-S 已记录 full-token graph family 的跨 token 漂移边界,所以 future-window full-token graph 仍不进默认生产。P25 采用的是更保守的 reusable linear-block graph 服务路径,已经把 OpenAI server decode 推到 **~99-100 TPS**;155TPS 剩余差距仍来自 active expert kernel,不是 HTTP wrapper。详见 [`docs/LYNN_ENGINE_P10S_GRAPH_BOUNDARY_20260515.md`](docs/LYNN_ENGINE_P10S_GRAPH_BOUNDARY_20260515.md) 和 [`docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md`](docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md)。

P15 配置说明:不要把 `LYNN_PACKED_DECODE=1` 当成“更 packed 就更快”。R6000 实测它会把 full graph path 从 **103.48 tok/s** 拉低到 **88.15 tok/s**,因为 Q/K/V/O 等小 decode linear 走 generic packed native path 反而慢。当前正确配置是 MoE active experts packed、linear-attn fused native、lm_head native,但 **generic packed decode 关闭**;shared expert 也保留 BF16,因为 packed scalar/native shared 都更慢。详见 [`docs/LYNN_ENGINE_P15_RUNTIME_CONFIG_20260516.md`](docs/LYNN_ENGINE_P15_RUNTIME_CONFIG_20260516.md)。

P16 155TPS 说明:profiling 证明非-MoE 路径 replay-only 可到 **208.8 tok/s**,skip active routed experts 可到 **173.8 tok/s**,但 top-k 近似和 block-size sweep 都无法安全到 155。结论是 155 不是再调 env var,而是要写新的 **grouped native-FP4 active expert kernel**。详见 [`docs/LYNN_ENGINE_P16_155TPS_ACTIVE_MOE_20260516.md`](docs/LYNN_ENGINE_P16_155TPS_ACTIVE_MOE_20260516.md)。

P17 说明:Triton 3.6 的 `tl.dot_scaled(e2m1)` 已在 R6000 上通过 raw packed FP4 layout probe,真实 gate/up 形状 `1x2048 @ 2048x8192` 只需 **0.0125ms**。scale probe 进一步确认 `rhs_scale=[N,K//32]` 且 synthetic two-sided neutral byte 为 **127**。这证明 FP4 tensor-core compute 不是瓶颈;下一关是把 Lynn per-16/e4m3 scale contract 桥接到 e8m0/group32 grouped native dot。详见 [`docs/LYNN_ENGINE_P17_TRITON_DOT_SCALED_20260516.md`](docs/LYNN_ENGINE_P17_TRITON_DOT_SCALED_20260516.md)。

P18 说明:三条 scale bridge 都已实测,`dot_scaled` raw gate/up 可到 **0.018ms**(当前 scalar bridge 约 **0.050ms**),但质量不过线:per-16→group32 fold 最佳 inter cosine **0.894**,BF16→e8m0/group32 re-quant **0.980**,padded per-16 **0.936**。结论:155TPS 仍有硬件 headroom,但不能用简单 e8m0 bridge 牺牲质量,下一步转 **custom per-16 grouped native FP4 kernel / CUTLASS** 或更强 engine-native quant artifact。详见 [`docs/LYNN_ENGINE_P18_SCALE_CONTRACT_DECISION_20260516.md`](docs/LYNN_ENGINE_P18_SCALE_CONTRACT_DECISION_20260516.md)。

P25 说明:补齐 `LYNN_LINEAR_BLOCK_GRAPH=1 / REUSE=1 / PREWARM=1` 后,OpenAI-compatible server 不再停在 88-89 TPS 旧路径,而是稳定到 **~99-100 decode tok/s**;512 token 请求端到端 **87.7 tok/s**。这证明服务层不是 155 的主 blocker,剩余差距仍在 active expert kernel。详见 [`docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md`](docs/LYNN_ENGINE_P25_SERVER_100TPS_20260516.md)。

P24/P26 说明:两个 Triton-only 捷径都被关掉。P24 的 per-16 dequant→`tl.dot` 数值过但最快 **0.0807ms**,比生产 scalar gate/up **0.0335ms** 慢;P26 的 merged-topk 调度 cosine=1.0,但四个代表层都 **2.06-2.09× 更慢**。结论进一步明确:155TPS 需要 CUDA/CUTLASS 级 custom per-16 grouped native-FP4 expert kernel,不是继续 rearrange Triton program grid。详见 [`docs/LYNN_ENGINE_P24_TL_DOT_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P24_TL_DOT_NEGATIVE_20260516.md) 和 [`docs/LYNN_ENGINE_P26_MERGED_TOPK_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P26_MERGED_TOPK_NEGATIVE_20260516.md)。

P27 说明:R6000 native CUDA extension build/load/launch gate 已通过。当前环境为 PyTorch **2.10.0+cu128** + CUDA toolkit **12.8** + `sm_120`;`torch.utils.cpp_extension.load` 可成功编译并加载 Lynn 自有 CUDA extension,1M float `add_one` smoke kernel `max_abs=0`,平均 **0.0047ms**。这不是 TPS 提升本身,但它把下一步 custom per-16 grouped native-FP4 active expert kernel 的工程地基打通。详见 [`docs/LYNN_ENGINE_P27_CUDA_EXTENSION_SMOKE_20260516.md`](docs/LYNN_ENGINE_P27_CUDA_EXTENSION_SMOKE_20260516.md)。

P28 说明:第一个真实 active-MoE CUDA extension 契约已打通。`gate_up_silu_scalar` 直接消费 Lynn 27B final 的 grouped packed NVFP4 tensor、per-16 scale、top-k expert ids,输出 `[top_k,512]` intermediate。四个代表层对 Triton reference `cosine≈1.0 / max_abs≈0`,但速度 **0.035ms** 略慢于 Triton **0.034ms**,所以这是 contract PASS,不是 speed promotion。下一步是在同一 C++/CUDA 入口内部替换 scalar inner loop 为真正 grouped native-FP4 math。详见 [`docs/LYNN_ENGINE_P28_NATIVE_GATEUP_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P28_NATIVE_GATEUP_CONTRACT_20260516.md)。

P29 说明:active MoE 的第二半 down weighted-sum CUDA extension 契约也已打通。`down_weighted_sum_scalar` 消费 `[top_k,512]` intermediate、routing weights、down packed/scale/global,输出 `[2048]` hidden。四个代表层对 Triton reference `cosine=1.0`,速度 **0.030ms** 慢于 Triton **0.026-0.027ms**,所以同样不 promoted,但 P28+P29 已经组成完整 native active-MoE 数据契约。详见 [`docs/LYNN_ENGINE_P29_NATIVE_DOWN_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P29_NATIVE_DOWN_CONTRACT_20260516.md)。

P30 说明:完整 active routed expert path 已经能由 Lynn 自有 CUDA extension 串起来:CUDA gate/up scalar → CUDA down weighted-sum scalar,四个代表层对 production Triton active path `cosine=1.0 / max_abs≈0`。速度 **0.064-0.068ms/layer** 慢于 Triton **0.058ms/layer**,所以不作为 runtime 默认;但 P27-P30 已经把 native extension build、gate/up、down、完整 active MoE 契约全部打通。接下来真正影响 155TPS 的工作只剩替换 scalar inner loop 为 grouped native-FP4 tensor-core math。详见 [`docs/LYNN_ENGINE_P30_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P30_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md)。

P31 说明:把 P30 native active-MoE scalar path 接入 production `moe_forward_decode_packed_nvfp4` 后,在函数边界反而比默认 Triton backend 快 **1.48-1.59×**,且四个代表层 exact/cosine=1.0。这说明 native extension 砍掉了一部分 Python/Triton wrapper 开销。但它仍是 opt-in,默认不开;promotion 前必须跑 full-token decode parity、full graph/server TPS、tool-call/no-think guards。详见 [`docs/LYNN_ENGINE_P31_NATIVE_ACTIVE_MOE_RUNTIME_GATE_20260516.md`](docs/LYNN_ENGINE_P31_NATIVE_ACTIVE_MOE_RUNTIME_GATE_20260516.md)。

P32-P34 说明:full-generate gate 明确否决了直接推广 `cuda_scalar`。全层 `cuda_scalar` + reusable linear-block graph 虽然能看到 **121.7 tok/s**,但从第 2 个 decode token 开始进入 token-0 / `!` loop;关掉 graph 后文本恢复连贯,但 greedy ids 仍不匹配;只替换 full-attention eager 层也仍有 top-1 drift。runner 已加 fail-loud guard,禁止 `cuda_scalar` 与 linear-block graph 的不安全组合静默上线。详见 [`docs/LYNN_ENGINE_P32_P34_NATIVE_ACTIVE_MOE_GENERATE_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P32_P34_NATIVE_ACTIVE_MOE_GENERATE_NEGATIVE_20260516.md)。

P35 说明:graph-slot 路线不是死路,但它比 eager/full-graph 路径更挑 router 顺序。把 `LYNN_ROUTER_TOPK_SORTED=1` 打开后,4 类 prompt × 3 prefix 的 full-token graph-slot gate **12/12 strict PASS**,所有行 `max_abs=0`,replay **97.7-111.5 tok/s**。结论:当前稳定 serving 可继续用 P20 unsorted top-k,但未来 graph-slot serving 模式必须强制 sorted-router contract。详见 [`docs/LYNN_ENGINE_P35_SORTED_ROUTER_GRAPH_SLOT_20260516.md`](docs/LYNN_ENGINE_P35_SORTED_ROUTER_GRAPH_SLOT_20260516.md)。

P36 说明:把 decode hot loop 里的 MoE 函数选择、linear-attn recurrent backend、state update 策略固定到 runner 初始化阶段,移除 per-layer/per-token env/import 分派。R6000 gate 显示 greedy ids **完全一致**,legacy median **100.55TPS**,fast-dispatch median **100.53TPS**。结论:这是安全工程清理,默认保留;但 100→155 的主瓶颈不在 Python 分派,仍在 active routed expert kernel / graph-owned serving path。详见 [`docs/LYNN_ENGINE_P36_DECODE_FAST_DISPATCH_20260516.md`](docs/LYNN_ENGINE_P36_DECODE_FAST_DISPATCH_20260516.md)。

P37 说明:layer28 分段 profile 显示当前 MoE 由 router **0.036ms**、active packed NVFP4 experts **0.107ms**、shared BF16 expert **0.060ms** 组成;孤立 block sweep 里 `down_block_hidden=16` 看似有 0.7% 小优势,但 full generate gate 跑到 **94.94TPS** 且 3/3 prompts greedy ids 漂移。结论:继续抠 Triton block-size 不是 155TPS 的正路,下一步必须转 **grouped native-FP4 active expert kernel** 或严格 parity 的 graph-owned serving。详见 [`docs/LYNN_ENGINE_P37_MOE_BLOCK_RETUNE_CLOSED_20260516.md`](docs/LYNN_ENGINE_P37_MOE_BLOCK_RETUNE_CLOSED_20260516.md)。

P38 说明:把 P37 的单层 profile 扩到 6 个采样层(2/8/14/20/28/36),结果高度一致:full MoE 平均 **0.193ms/layer**,其中 router **0.037ms**,active routed packed NVFP4 experts **0.112ms**,shared BF16 expert **0.060ms**。结论:没有异常慢层可捡,155TPS 的主攻方向就是 active expert grouped native-FP4 kernel,shared expert 是第二优先级。详见 [`docs/LYNN_ENGINE_P38_MOE_MULTILAYER_PROFILE_20260516.md`](docs/LYNN_ENGINE_P38_MOE_MULTILAYER_PROFILE_20260516.md)。

P39-P40 说明:active 内部拆分后,gate/up **0.033ms**、down **0.025ms**,split 结果与 combined active exact;随后把当前 R6000 best MoE config 固化为 `LYNN_MOE_FAST_FIXED` 路径。layer-level candidate 平均 **1.079×**,full-generate gate **greedy ids 全一致**,默认开启后复核 median **100.79TPS**。这是安全小刀,但不是 155TPS 的大突破;下一步仍然是 fused/grouped native-FP4 active expert kernel。详见 [`docs/LYNN_ENGINE_P39_P40_FAST_FIXED_MOE_20260516.md`](docs/LYNN_ENGINE_P39_P40_FAST_FIXED_MOE_20260516.md)。

P42 说明:重新测试 `cuda_scalar` native active-MoE backend,即使只 allowlist 到 full-attention 层、避开 linear-block graph capture,full-generate 仍然 **0/3 greedy match**,且有一题掉到 **48.50TPS**。结论:`cuda_scalar` 只保留为 native extension contract/diagnostic backend,不再作为生产提速捷径。详见 [`docs/LYNN_ENGINE_P42_CUDA_SCALAR_RETEST_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P42_CUDA_SCALAR_RETEST_NEGATIVE_20260516.md)。

P43-P44 说明:155TPS 路线继续收窄。P43 证明 shared expert fused path 只从 **0.0609ms** 降到 **0.0556ms/layer**,绝对收益太小;P44 重新关掉 merged-topk 调度(最快约 production 的 **0.48×**)和 cross-expert `torch._scaled_mm` 拼接(mm-only 约 **0.39×**,quant+mm 约 **0.15×**,且 min cosine **0.97698**)。结论:不能靠包装/调度偷过 155,必须写真 grouped/block-diagonal FP4 active expert kernel。详见 [`docs/LYNN_ENGINE_P43_P44_ACTIVE_MOE_NATIVE_FP4_TRIAGE_20260516.md`](docs/LYNN_ENGINE_P43_P44_ACTIVE_MOE_NATIVE_FP4_TRIAGE_20260516.md)。

P45-P46 说明:P45 把 native active-MoE 固定成 one-call CUDA ABI:`active_moe(hidden,expert_ids,routing_weights,gate_up*,down*) -> out[2048]`,数值与 two-call scalar reference 对齐(min cosine **0.99999988**),但仍慢于 Triton(**0.0658ms vs 0.0583ms**)所以只作为 kernel 地基。P46 试了单 kernel fused-atomic bridge,速度 **0.1768ms** 远慢于 Triton **0.0592ms**,并有轻微 accumulation drift。结论:P47 必须做非 atomic grouped/block-diagonal kernel,不是继续叠 wrapper。详见 [`docs/LYNN_ENGINE_P45_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P45_NATIVE_ACTIVE_MOE_CONTRACT_20260516.md) 和 [`docs/LYNN_ENGINE_P46_FUSED_ATOMIC_NEGATIVE_20260516.md`](docs/LYNN_ENGINE_P46_FUSED_ATOMIC_NEGATIVE_20260516.md)。

P48-P50 说明:第一版非 atomic 路线选择了 down projection 子段,新增 tile-hidden CUDA kernel。isolated microbench 在 6 个代表层上从 Triton down **0.02586ms** 降到 **0.02067ms**(**1.25×**),P49 进一步证明真实 decode-state 下仍有 **1.27×** 局部收益(max rel_l2 **8.74e-05**)。但 P50 关闭 CUDA graph 后跑完整 decode,仍在 step 1/layer 27 出现微小漂移,step 5 翻 top1。结论:P48 是 kernel 级正信号,但 tiny accumulation drift 足以影响 greedy,不能进默认路径。详见 [`docs/LYNN_ENGINE_P48_DOWN_TILE_NONATOMIC_20260516.md`](docs/LYNN_ENGINE_P48_DOWN_TILE_NONATOMIC_20260516.md)。

P51 说明:为了排除“少算专家就能到 155TPS”的捷径,新增 opt-in `LYNN_MOE_TOPK_LIMIT` / `LYNN_MOE_SKIP_SHARED` 预算阶梯。结果很明确:top6+shared 仍 coherent 但只提升 **1.6%**,top4+shared 提升 **4.5%** 已出现 `<think>` 污染;skip shared 最快 **124.39TPS** 但输出崩坏。结论:不能靠裁专家到 155,必须继续 grouped native-FP4 active expert FFN 或 exact graph-owned route。详见 [`docs/LYNN_ENGINE_P51_ACTIVE_MOE_BUDGET_LADDER_20260516.md`](docs/LYNN_ENGINE_P51_ACTIVE_MOE_BUDGET_LADDER_20260516.md)。

P52 说明:路线正式分叉为两条:一条是 exact-owned serving,保持当前 Triton active MoE 数学不变,继续压 orchestration / graph overhead;另一条是真 grouped native-FP4 active expert FFN,用 CUTLASS/CuTe 或 custom CUDA 表达 block-diagonal selected experts,不再用 `_scaled_mm` 包装或 top-k 近似。P52-A/B 进一步证明:只把 selected gate/up 换成 `torch._scaled_mm` 会更慢且 active cosine 只有 **0.976**,主要误差来自把 Lynn 的 FP32 per-16 scale 压进 FP8 `scale_b` contract(min cosine **0.972**),不是神秘 tensor-core 累加问题。MTP/spec decode 不是近期主线,它是之后的 serving multiplier,不是底座 kernel。详见 [`docs/LYNN_ENGINE_P52_GROUPED_NATIVE_FP4_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P52_GROUPED_NATIVE_FP4_CONTRACT_20260516.md)。

P53 说明:针对外部 review 提出的 Triton 免费优化假设,新增 scale-hoist 与 E2M1 decode 简化两个 opt-in probe。scale-hoist 朴素版 JIT/body 过重,不构成免费收益;轻量 LUT 表达式数值 exact,前三个代表层快 **6-8%**,但 layer36 慢到 **0.536×**,平均 **0.936×**。结论:这个方向保留为 allowlist/变体研究,不替换默认 Triton active MoE。详见 [`docs/LYNN_ENGINE_P53_TRITON_RETUNE_REVIEW_20260516.md`](docs/LYNN_ENGINE_P53_TRITON_RETUNE_REVIEW_20260516.md)。

P54 说明:为了回应“能否直接做一份 NVIDIA/ModelOpt 友好的 e8m0/group32 NVFP4 artifact 然后吃官方 kernel”,新增离线 scale-search 上界 probe。`search_recon_mse` 代表可部署式静态重构搜索,`search_dot_upper_bound` 是激活感知点积上界;四个代表层 best upper-bound inter cosine 仍只有 **0.9869-0.9918**,全部低于 0.995 safety gate。结论:官方 NVFP4/Blackwell 路线证明方向正确,但 Lynn 当前 per-16 FP32 scale artifact 不能靠简单离线 exponent search 转成 vendor scale contract;主线继续写 **Lynn-native per-16 grouped active expert kernel**。详见 [`docs/LYNN_ENGINE_P54_VENDOR_LAYOUT_FEASIBILITY_20260516.md`](docs/LYNN_ENGINE_P54_VENDOR_LAYOUT_FEASIBILITY_20260516.md)。

量化 v2 说明:imatrix / 层策略 / activation-aware scale search 对 Lynn 有价值,但它是**量化质量线**,不是当前 100→155TPS runtime blocker 的替代品。近期最适合先用于 GGUF/Q4_K_M 公开版,降低 PPL/KLD 并提升 V8/V9/tool/longctx retention;下一阶段再从 BF16 final 重新探索 vendor-compatible NVFP4 v2 artifact。详见 [`docs/LYNN_ENGINE_QUANTIZATION_V2_ROADMAP_20260516.md`](docs/LYNN_ENGINE_QUANTIZATION_V2_ROADMAP_20260516.md)。

P55 说明:新增 gate/up 侧 `tile_inter` CUDA scalar probe,让一个 block 同时计算多个 intermediate row 并复用 hidden 读取。四个代表层上 `tile_inter=2` 全部 max_abs=0,局部比 Triton gate/up 快 **1.09-1.14×**;`tile_inter=4/8` 反而慢。结论:这是 P56 grouped per-16 kernel 的正向 tile 形态信号,但仍只是局部 scalar probe,不会单独进默认生产。详见 [`docs/LYNN_ENGINE_P55_GATEUP_TILE_INTER_20260516.md`](docs/LYNN_ENGINE_P55_GATEUP_TILE_INTER_20260516.md)。

P56 说明:把 P55 `tile_inter=2` 接进 runtime opt-in gate 后,中位 decode TPS 从约 **99.15** 提到 **114.43**(+15.4%),但 full-generate greedy 不匹配并出现 `!` loop。结论:tile shape 有真实速度信号,但 scalar tile-inter runtime 不安全,不 promote;下一步直接把 `tile_inter=2` 作为真 grouped per-16 native-FP4 kernel 的形态提示。详见 [`docs/LYNN_ENGINE_P56_GATEUP_TILE_RUNTIME_REJECTED_20260516.md`](docs/LYNN_ENGINE_P56_GATEUP_TILE_RUNTIME_REJECTED_20260516.md)。

P57 说明:官方/ModelOpt 路线不丢人,但它需要一份从 BF16 final 重新量化的 **vendor-friendly NVFP4 v2 artifact**,而不是把当前 Lynn-native per-16 artifact 硬转过去。R6000 当前 env 无 `modelopt/llmcompressor`,只有 `compressed_tensors` 的 `modelopt_nvfp4` converter,且 27B 是物理 variable-expert、`hf_vanilla_compatible=false`,所以官方路线还需要 padding/mask 回固定 256 experts 或 vendor 侧 variable-expert 支持。详见 [`docs/LYNN_ENGINE_P57_VENDOR_ROUTE_INVENTORY_20260516.md`](docs/LYNN_ENGINE_P57_VENDOR_ROUTE_INVENTORY_20260516.md)。

双 artifact 策略:允许同时保留 **Lynn-native NVFP4 final** 和 **vendor-friendly NVFP4 v2**。前者继续服务 Lynn engine / per-16 grouped kernel;后者从 BF16 final 重新量化,用于官方生态兼容。两份产物必须不同目录、不同 manifest、不同验证门,不得互相覆盖。详见 [`docs/LYNN_ENGINE_DUAL_NVFP4_ARTIFACT_POLICY_20260516.md`](docs/LYNN_ENGINE_DUAL_NVFP4_ARTIFACT_POLICY_20260516.md) 和 [`docs/LYNN_ENGINE_VENDOR_VS_LYNN_NATIVE_TRADEOFF_20260516.md`](docs/LYNN_ENGINE_VENDOR_VS_LYNN_NATIVE_TRADEOFF_20260516.md)。

P58 说明:为排除“P56 只是 CUDA extension 与 linear-block graph 交互坏掉”的可能,重跑 `cuda_tile_inter` 并关闭 `LYNN_LINEAR_BLOCK_GRAPH*`。结果候选中位 **28.43TPS**,且 greedy mismatch 仍存在。结论:这不是 graph-only bug,而是 scalar tile-inter 累加/调度顺序本身足以扰动 greedy decode;该 runtime 线正式关闭,只保留 tile 形态信号。详见 [`docs/LYNN_ENGINE_P58_GATEUP_TILE_GRAPH_OFF_REJECTED_20260516.md`](docs/LYNN_ENGINE_P58_GATEUP_TILE_GRAPH_OFF_REJECTED_20260516.md)。

P59 说明:新增 metadata-only NVFP4 layout classifier,在加载 tensor 之前区分 `lynn_native_per16_variable`、`compressed_tensors_nvfp4`、`modelopt_nvfp4`、`bf16_or_unquantized` 和 unknown packed FP4。目标是一个 Lynn engine 同时支持我们的原生 NVFP4 与未来官方/vendor-friendly NVFP4 v2,但任何交叉误加载都 fail-loud。详见 [`docs/LYNN_ENGINE_P59_DUAL_NVFP4_LAYOUT_DISPATCH_20260516.md`](docs/LYNN_ENGINE_P59_DUAL_NVFP4_LAYOUT_DISPATCH_20260516.md)。

P85-P89 说明:官方/ModelOpt 路线继续保留,但 runtime 主线不再等待重重量化。P85-P87 已证明 SM120a blockscaled FP4 MMA、E2M1 `<<2` shift 和 CuTe fragment layout 全部正确;P88 把真实 Lynn 27B gate/up packed-code tile 跑到 `max_abs=0`;P89 进一步证明当前 Lynn-native per-16 scale artifact 可以用 **split16 neutral-scale MMA + 显式 per-group scale accumulate** 直接消费,误差仅 `rel_l2=1.0e-7`。简单把两个 K16 scale 折成一个 K32 scale 会漂移(best rel_l2 仍 0.0227),所以官方/vendor-friendly NVFP4 v2 应放到 MTP/retrain + re-quant 批次,不阻塞 P90 真 active expert kernel。详见 [`docs/LYNN_ENGINE_P85_BLOCKSCALED_FP4_MMA_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P85_BLOCKSCALED_FP4_MMA_CONTRACT_20260516.md)、[`docs/LYNN_ENGINE_P86_FP4_SHIFT_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P86_FP4_SHIFT_CONTRACT_20260516.md)、[`docs/LYNN_ENGINE_P87_FP4_LAYOUT_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P87_FP4_LAYOUT_TILE_CONTRACT_20260516.md)、[`docs/LYNN_ENGINE_P88_REAL_GATEUP_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P88_REAL_GATEUP_TILE_CONTRACT_20260516.md) 和 [`docs/LYNN_ENGINE_P89_PER16_SCALE_TILE_CONTRACT_20260516.md`](docs/LYNN_ENGINE_P89_PER16_SCALE_TILE_CONTRACT_20260516.md)。

P90 说明:第一版真实 split16 gate/up kernel 已经把 P89 从 tile proof 推到 full-K row tile。输入是真实 Lynn 27B layer28 expert116、真实 activation、真实 packed gate/up weight,输出 8 个 gate rows + 8 个 up rows,K=2048 全维度。结果 `max_abs=2.38e-7`, `rel_l2=1.53e-7`,tolerance PASS。结论:当前 Lynn-native artifact 可以直接喂 native FP4 gate/up kernel,无需等待官方/vendor re-quant;P91 开始扩大 row tile 并减少 atomic/launch overhead。详见 [`docs/LYNN_ENGINE_P90_SPLIT16_GATEUP_KERNEL_20260516.md`](docs/LYNN_ENGINE_P90_SPLIT16_GATEUP_KERNEL_20260516.md)。

P91 说明:row-tile sweep 直接给出下一步形态。8/16/32/64 行全部在 `1e-5` tolerance 下 PASS;64 行 median **0.0443ms**,rows/ms 从 8 行的 **350** 提到 **2892**。这说明扩大 row tile 没有破坏 P90 数学,还显著摊薄 launch/atomic overhead。P92 应直接尝试 full 512-row gate/up expert shape。详见 [`docs/LYNN_ENGINE_P91_SPLIT16_ROWTILE_SWEEP_20260516.md`](docs/LYNN_ENGINE_P91_SPLIT16_ROWTILE_SWEEP_20260516.md)。

P92 说明:完整 gate/up expert 子算子已经通过。真实 Lynn 27B layer28 expert116,512 个 gate rows + 512 个 up rows,K=2048 全维度,median **0.0502ms**,`max_abs=4.77e-7`,`rel_l2=1.53e-7`。这回答了核心路线问题:当前 Lynn-native artifact 可以直接驱动完整 native-FP4 gate/up expert 子算子。后续工作从“能不能吃当前 artifact”转为“怎么把 launch/atomic/row ownership 做成生产 backend”。详见 [`docs/LYNN_ENGINE_P92_FULL_GATEUP_EXPERT_20260516.md`](docs/LYNN_ENGINE_P92_FULL_GATEUP_EXPERT_20260516.md)。

P19 说明:在不改变数值路径的前提下,active MoE kernel block retune 把 R6000 full graph 从 **103.40/107.13 TPS** 提到 **115.41/120.25 TPS**。推荐配置已成为默认:`gate_hidden=256,down_inter=512`,并保留 env override 方便后续设备差异调参。详见 [`docs/LYNN_ENGINE_P19_ACTIVE_BLOCK_RETUNE_20260516.md`](docs/LYNN_ENGINE_P19_ACTIVE_BLOCK_RETUNE_20260516.md)。

P20 说明:router `topk(sorted=False)` 已验证同 expert set、同配对权重,代表层 MoE 输出 max_abs=0,把 full graph 进一步推到 **117.55/122.43 TPS**。详见 [`docs/LYNN_ENGINE_P20_ROUTER_TOPK_UNSORTED_20260516.md`](docs/LYNN_ENGINE_P20_ROUTER_TOPK_UNSORTED_20260516.md)。

P21 说明:shared expert 保持 BF16,只把 gate/up 两个小 GEMM 融成一个 BF16 GEMM,代表层 max_abs=0。full graph 小幅提升到 **117.71/122.71 TPS**。详见 [`docs/LYNN_ENGINE_P21_SHARED_GATEUP_FUSION_20260516.md`](docs/LYNN_ENGINE_P21_SHARED_GATEUP_FUSION_20260516.md)。

P22 说明:MoE active kernel 暴露 `num_warps` 调参后,R6000 最优为 gate/up 4 warps、down 8 warps,full graph 小幅提升到 **118.25/123.25 TPS**。详见 [`docs/LYNN_ENGINE_P22_MOE_WARP_RETUNE_20260516.md`](docs/LYNN_ENGINE_P22_MOE_WARP_RETUNE_20260516.md)。

P23 说明:active MoE 全层画像确认没有异常慢层,40 层 active routed experts 几乎均匀在 **~0.069ms/layer**,router 约 **0.049ms/layer**,shared expert 约 **0.057ms/layer**。1D top-k、手写 softmax、Triton fused top-k+softmax 都没有进入生产价值;唯一 promoted 的安全小刀是 expert id 在 router 后一次性转 int32,避免 gate/up 和 down 重复 cast,full graph 提到 **118.73/123.78 TPS**。详见 [`docs/LYNN_ENGINE_P23_ACTIVE_MOE_ACCOUNTING_20260516.md`](docs/LYNN_ENGINE_P23_ACTIVE_MOE_ACCOUNTING_20260516.md)。

Packed-resident memory 说明:当前默认 server 仍保留 BF16 shadow 以支持多请求 prefill。P11 已证明 session-scoped 生命周期中,prefill 后可以释放 56.47 GiB BF16 shadow,显存从 81.06 GiB 降到 24.59 GiB 且 greedy decode ids 完全一致。P12 进一步把这个能力接到 OpenAI server 的 opt-in one-shot 模式:首请求释放 56.47 GiB,第二请求明确 HTTP 409 fail-loud;并验证 release 后 current-position graph slot 仍与 eager decode exact match。详见 [`docs/LYNN_ENGINE_P11_PACKED_RESIDENT_MEMORY_20260515.md`](docs/LYNN_ENGINE_P11_PACKED_RESIDENT_MEMORY_20260515.md) 和 [`docs/LYNN_ENGINE_P12_ONESHOT_SERVER_20260515.md`](docs/LYNN_ENGINE_P12_ONESHOT_SERVER_20260515.md)。

## 27B 质量与基座状态

27B 来自 Qwen 3.6 35B-A3B BASE 的 variable-expert 剪枝路线:

```text
BASE 35B-A3B
  → activation profile
  → variable-target expert pruning(1010 experts cut,front layers protected)
  → router fine-tune
  → Recovery LoRA
  → step5000 selected as final
  → BF16 merge
  → Lynn-native NVFP4 quantization
```

已知质量结论:

| Variant | 状态 | 结果 |
|---|---|---|
| 27B BF16 step5000 | ✅ full eval | V8 strict 33/34 = 97.06%, V9 adjusted 37/59 = 62.71% |
| 27B NVFP4 step5000 | ✅ runtime smoke | 6-prompt resident smoke PASS,2-token greedy sanity PASS |
| 27B Q4_K_M | ⏳ not primary | variable-expert GGUF 需要额外 padding/format work,不是当前 Lynn-native 主线 |

Recovery v1.1 targeted longctx/chem/sql 已试过,但未取代 step5000:它没有改善 longctx,并拉低总分。因此当前基座选择 **step5000 final**。

## 路线 — Lynn engine native-kernel 重启 + llama.cpp 务实 fallback

详见 [`RELEASE_NOTES_20260603.md`](RELEASE_NOTES_20260603.md) 当前重启口径。客户端短期仍可把 llama.cpp/GGUF 作为务实 fallback,但 Lynn engine 已回到并行主线:用自研 native kernels 追赶 llama.cpp。

### 短期(2-4 周):Lynn 客户端 llama.cpp 集成 ship 路径

| 层 | 角色 | 实现 |
|---|---|---|
| **底层推理** | llama.cpp ecosystem | Mac Metal / Windows / Linux CUDA 全平台 + Q4_K_M GGUF |
| **默认模型** | Qwen3.5-9B Q4_K_M-imatrix(5.3 GB)| 80% 用户 9B 已经够好 |
| **Pro 模型** | Qwen3.6-35B-A3B Q4_K_M-imatrix(20 GB)| NVIDIA 24GB+ 用户 opt-in,llama.cpp 同栈 |
| **Lynn 客户端** | 自动硬件 detect + install llama.cpp + 下载模型 + 启 server + 注册 provider + tool-call 门禁 + 本地优先 routing | Electron + brain backend(本仓姊妹仓 `MerkyorLynn/Lynn`)|
| **Lynn 智能体** | tool routing / 6 层 memory / MCP / skills / 跨模型 fallback | Lynn 真护城河 |

**llama.cpp 负责"跑得起来、跑得快、装得小"。Lynn 负责"会用模型、会调工具、会记忆、会自动配置"**。

### 长期 R&D:Lynn engine NVFP4 / W4A8 FP8 持续探索(本仓主仓)

**回主线门槛**(两条必过):
1. **速度**:同硬件同模型下,Lynn engine 接近或超过 llama.cpp
2. **质量**:有不可替代优势(GPQA / long-ctx / 稳定性 stddev / 工具调用等)

**R&D 主要方向**(都在 main 分支保留 commit trail,不 revert):

- **W4A8 FP8 vectorized expert dispatch + CUTLASS grouped GEMM**:绕开 Python decode overhead(Wave 2 retry #4 暴露的架构信号)
- **C++ service loop**:host gap > GPU compute gap 时再触发(memory `LYNN_ENGINE_CPP_RUST_REWRITE_ROI_20260517` 已分级 3 tier)
- **R6000-class FP4 MMA 硬件**:消费 Blackwell 32GB 卡上市后,Lynn engine native FP4 路径(5/15 R6000 107-108 TPS strict default)有竞争力
- **MTP K=1 sequential 6/6 @ 26.4 TPS** correctness-clean baseline 持续磨,等 W4A8 base 稳后叠加
- **长 ctx 6.77× SGLang at 16K** 数据点保留,作 NVIDIA Pro "高级模式"卖点

### 历史决策已变更(5/20 后失效)

5/15-5/16 锁定的以下决策已在 5/20 后失效:

- ~~推理主格式:Lynn-native NVFP4 only~~ → **生产默认 GGUF Q4_K_M-imatrix(llama.cpp ecosystem);NVFP4 / W4A8 FP8 留 R&D**
- ~~模型锁定:Lynn-27B-A3B variable-pruned family~~ → **5/17 战略 pivot 回原版 Qwen3.6-35B-A3B + Qwen3.5-9B(质量下降 10% 弃 27B 自蒸馏)**
- ~~vertical companion 给 Lynn LoRA + 剪枝训练流水线~~ → **Lynn engine 收敛到为原版模型族提供专项加速,LoRA 训练侧独立路线**
- **推理范围:单 prompt + batch=1** 保留(不做 PagedAttention)— Lynn 客户端用户场景就是单用户,无需 batched serving
- **推理硬件:Blackwell sm_12x** 保留(DGX Spark / 5090 / R6000 PRO),但优先级降到 R&D

## 教程 — 即使你不写自己的引擎也值得读

写 Lynn engine 时,我们挖出了 Qwen 3.6 35B-A3B **跟 Llama / Qwen 2 不一样、文档没写明的怪癖**。共 7 篇深度文章在 [`tutorials/`](tutorials/):

| # | 主题 | 一句话 |
|---|---|---|
| [01](tutorials/01_rmsnorm_one_plus_weight.md) | RMSNorm `(1.0 + w) × x` 不是 `w × x` | Qwen 3 系 RMSNorm 是 +1 偏移。照 Llama 抄数值偏 ~10x |
| [02](tutorials/02_rope_three_gotchas.md) | RoPE 三个连环坑 | theta 在 `rope_parameters`(不是 `rope_theta`)+ `partial_rotary_factor=0.25` + GPT-NeoX 半切(不是 Qwen 2 even/odd)|
| [03](tutorials/03_attn_output_gate.md) | q_proj 是 2× per-head 切分 | 必须先 view 成 (..., H_Q, 2*head_dim) 再 chunk,否则 head_i_gate 混进 head_i_q |
| [04](tutorials/04_gated_delta_net.md) | linear_attention = GatedDeltaNet | Mamba 风格 chunk 递推 + delta rule + l2norm Q/K |
| [05](tutorials/05_three_invisible_bugs.md) | 三个 self-consistent bug 复盘 | reference + lynn 同源同错 = 自一致测试假阳的教训 |
| [06](tutorials/06_moe_router_softmax_topk_order.md) | MoE router order + shared expert | Qwen 用 softmax-all → topK → renormalize,跟 naive 数学等价但精度路径不同 |
| [07](tutorials/07_lora_on_gated_delta_net.md) | 给 GatedDeltaNet 加 LoRA | 哪些线性层可加 / 哪些不能 / r=384 用于 Recovery 的理由 |

[`tutorials/posts/zhihu_qwen36_engine_postmortem.md`](tutorials/posts/zhihu_qwen36_engine_postmortem.md) 是知乎博客风格的合集长文。

## 快速上手(R6000 / Blackwell)

```bash
# 1. 准备 Lynn-native NVFP4 artifact
MODEL=/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final

# 2. 启用当前 R6000 best env
export PYTHONPATH=/root/autodl-tmp/lynn-engine
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
export LYNN_PACKED_SHARED_EXPERT=0

# 3. 运行 resident smoke
python benchmarks/resident_cli.py \
  --model "$MODEL" \
  --prompts-jsonl /root/autodl-tmp/reports/lynn-engine-p5/p7i_6prompt.jsonl \
  --max-new 32 \
  --chat-template \
  --out /tmp/lynn_27b_nvfp4_smoke.json

# 4. 可选:OpenAI-compatible server
python -m server.openai_http \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 18099
```

## 仓库结构

```
engine/
  loader.py                       FP8 e4m3 反量化 + 单层 safetensors 加载器
  qwen36_block.py                 早期完整 transformer block(P1.1,带 bug 的 reference)
  qwen36_linear_attn_block.py     GatedDeltaNet 移植 — 跟 HF bit-exact
  full_forward.py                 40 层端到端 forward(已修正)
  inference_state.py              单 request 的 KV cache + recurrent state
  incremental_decode.py           prefill / decode 原语(Phase 3.1)
  moe_optimized.py                MoE 三档优化(active / bmm / indexed bmm)
  convert_fp8_to_bf16.py          离线 FP8 → BF16 转换器(CPU)
  test_*.py                       逐层 alignment 测试 + 多 prompt 验证

triton_kernels/
  attention.py / rope.py / rmsnorm.py / moe.py   P1 spike 内核
  moe_expert_ffn.py               Phase 3.2.3 融合 MoE 内核(scaffold)

server/
  openai_http.py                  FastAPI OpenAI 兼容 server
  README.md                       brain 接入指引

docs/
  STRATEGY.md                     ⭐ 战略 + 长期路线图
  DESIGN.md                       架构 + 实施日志(长)
  PHASE3_KV_CACHE_DESIGN.md       Phase 3.1 设计文档
  NVFP4_QUANTIZATION.md           NVFP4 量化 pipeline(C 阶段末 + B 阶段输入)

pruning/
  calibration/
    seeds.jsonl                   102 手工策划 seed prompt(19 类别)
    expand.py                     DeepSeek API 扩到 1440 prompts
    calibration_set_v1.1.jsonl    生成的完整集
  profile_activations.py          Phase 1 W1: 激活画像脚本
  decide_pruning.py               Phase 1 W2: drop 30 expert 决策算法
  training/
    recovery_lora_qwen36.yaml     Recovery LoRA r=384 配置(LLaMA-Factory)

benchmarks/
  lynn_27b_vs_35b.py              Phase 1 W4 gate test(自动跑 V9 比较)

tutorials/
  README.md + 7 篇 tutorial + Zhihu 长文
```

## 部署目标硬件

| 硬件 | VRAM | 状态 |
|---|---|---|
| **DGX Spark**(GB10 sm_121,unified 119GB) | 119 GB | C 阶段开发主力 |
| **RTX 5090 笔记本**(sm_120,24GB) | 24 GB | **Lynn-27B-A3B-NVFP4 占 ~20 GB,4 GB 余量给 16K context** ✅ |
| **RTX 5090 台式机**(sm_120,32GB) | 32 GB | 预期 180-250 t/s,32K context 宽裕 |
| **RTX PRO 6000 Blackwell**(sm_120,96GB) | 96 GB | 多 LoRA 切换 + 长 context 扩展 |
| ~~4090 / Ada~~ | — | 不支持(没 FP4 tensor cores)|
| ~~A100 / H100 / Hopper~~ | — | 不支持(同上,FP4 emulation 不值)|
| ~~Ampere / Volta~~ | — | 不支持(老)|

## 为什么写这个

Lynn brain 用 Qwen 3.6 35B-A3B 当主链,每天处理几千个 agent 请求。vLLM(目前生产)能吃掉 Spark 内存带宽 ~80%,Lynn engine 的目标是**单模型锁定 + Blackwell sm_12x 特化** + agent prompt 缓存(99% 命中)拿到剩下 20% + 更低 tail latency。

但更重要的产出是**通过自己写引擎获得的理解** — 上面的 tutorials 就是这个理解的固化。

## 老实讲的取舍

```
❌ 锁定 Qwen 3.6 35B-A3B(及其剪枝衍生)+ Blackwell sm_12x
❌ Qwen 升 3.7/4 不兼容时,需要 4-6 周重写
❌ 不做批处理 / 多并发(单 prompt 焦点)
❌ 不做采样(只 greedy);不做 beam / speculative
❌ 不支持 FP8/INT4/AWQ/GPTQ 等(NVFP4 唯一生产格式)
✅ 完全契合 Lynn brain 的部署模式
✅ 跟 LoRA + 剪枝训练流水线 vertical 整合
✅ 所有架构细节都给后人写下来了
```

## 不打算做的事(也写明防止 scope creep)

```
❌ Continuous batching / PagedAttention(vLLM 主场)
❌ 多模型 loader(锁 Lynn-27B-A3B 家族)
❌ 多量化格式(NVFP4 唯一)
❌ Hopper / Ada / Ampere 支持(Blackwell 唯一)
❌ TP / PP / 多机分布式(单机单 GPU)
❌ Vision encoder 整合(Lynn 不接图像;后续看)
❌ 与 vLLM 通用 API 完全兼容(只兼容 brain 用的子集)
❌ 投 paper(Lynn 是产品,不是论文)
```

## License

TBD(大概率 MIT,B 阶段生产切换前定下来)。

---

## 相关链接

- [战略 / 路线图(STRATEGY.md)](docs/STRATEGY.md)
- [架构设计(DESIGN.md)](docs/DESIGN.md)
- [Phase 3 incremental decode 设计](docs/PHASE3_KV_CACHE_DESIGN.md)
- [NVFP4 量化 pipeline](docs/NVFP4_QUANTIZATION.md)
- [剪枝 pipeline(calibration / profile / decide / LoRA)](pruning/README.md)
- [HTTP server(brain 接入指引)](server/README.md)
- [English README](README_EN.md)
- [📝 知乎 2026年6月连载:从零开始 Qwen 3.6 35B-A3B 写专用推理引擎踩坑心得分享](https://zhuanlan.zhihu.com/p/2045562329396400486)
- [📝 历史工程复盘:从零开始 Qwen 3.6 35B-A3B 写专用推理引擎(2026-05-11 mega-post)](https://zhuanlan.zhihu.com/p/2036443846322680848) — Phase 2 + Phase 3.2 + NVFP4 路线决策三段合并
- [Toolabstain 论文(姊妹项目)](https://github.com/MerkyorLynn/toolabstain-paper)
