# Lynn Engine

> **为 NVIDIA Blackwell 写的 Lynn 27B-A3B NVFP4 单模型推理引擎。**
> 从零写,锁定 Lynn 自家的 variable-pruned MoE + NVFP4 格式,目标很窄也很硬:在 R6000 / Spark 这类 Blackwell 机器上,把 Lynn 27B 基座跑成可生产、可优化、可长期接管的推理内核。

[Read in English](README_EN.md) · [📝 知乎工程复盘(2026-05-11)](https://zhuanlan.zhihu.com/p/2036443846322680848) · [战略文档](docs/STRATEGY.md) · [架构设计](docs/DESIGN.md)

[![commits](https://img.shields.io/github/commit-activity/m/MerkyorLynn/lynn-engine)](https://github.com/MerkyorLynn/lynn-engine/commits/main)
[![license](https://img.shields.io/badge/license-TBD-orange)](.)

## 当前状态(2026-05-16)

Lynn engine 已经从“Qwen 35B 架构复刻”推进到 **Lynn 27B final 基座的独立 NVFP4 runtime**:

| 项目 | 状态 |
|---|---|
| **27B final BF16** | ✅ Recovery step5000 final 已 merge,structural validation PASS,greedy sanity PASS |
| **27B Lynn-native NVFP4** | ✅ 20G artifact 已生成并传到 R6000,manifest integrity PASS |
| **独立加载** | ✅ 不依赖 vLLM / SGLang / TRT-LLM / llama.cpp,直接读 safetensors + Lynn quant manifest |
| **6-prompt coherent smoke** | ✅ 中文解释 / Python / RoPE-ALiBi / 英文算术 / tool JSON / longctx 全通过 |
| **当前 R6000 strict full path** | ✅ **118.73 tok/s**(P23:packed NVFP4 MoE + native FP4 lm_head + active MoE retune) |
| **serving replay ceiling** | ✅ **123.78 tok/s**(40-layer body graph,可稳定复现) |
| **OpenAI server guard** | ✅ strict tool-call PASS,`<think>` loop fail-pattern guard PASS,补齐 reusable block graph 后 decode **~100 tok/s** |
| **P10 runner graph-slot gate** | ✅ 6 prompts × 3 prefixes = 18/18 strict PASS,runner graph slot 88.8-103.1 tok/s |
| **P11 packed-resident memory gate** | ✅ prefill 后释放 **56.47 GiB** BF16 shadow,allocated **81.06 → 24.59 GiB**,greedy ids exact match |
| **P12 one-shot + graph-after-release gate** | ✅ OpenAI server 首请求释放 **56.47 GiB**;释放后 graph slot 79.5-83.8 tok/s,max_abs=0 |
| **P13 graph-slot generate wiring** | ✅ `generate()` opt-in 接入 full-token graph slot;多 prompt 证明 future-window 不安全,下一步 state-refresh slot |
| **P14 state-refresh probe** | ✅ full mutable state roundtrip **0.79ms**,远低于 graph capture 60-105ms |
| **P15 runtime config audit** | ✅ 关闭全局 `LYNN_PACKED_DECODE`;恢复 **103.48 strict / 107.23 replay**;shared expert packed 路径证伪 |
| **下一目标** | 生产稳定 100+ TPS + custom per-16 grouped native-FP4 active expert kernel |

当前主力 artifact:

```text
Lynn 27B variable-pruned Recovery step5000
├── BF16 final      ~60G  (reference / eval / fallback)
└── NVFP4 final     ~20G  (Lynn-native runtime artifact)
```

> 注意:这里的 NVFP4 是 **Lynn-native variable-expert NVFP4**。它不是公开发布用的 compressed-tensors v8-RTN,也不是 GGUF Q4_K_M。通用框架通常不能直接加载这个 variable-pruned artifact,这正是 Lynn engine 要存在的原因。

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

## 路线 — R6000 first,Spark second

详见 [`docs/STRATEGY.md`](docs/STRATEGY.md)。简版:

| 阶段 | 状态 | 目标 |
|---|---|---|
| **P6** | done | 50TPS 突破:resident graph smoke 63-66TPS |
| **P7** | done | graph reuse / prewarm / RMSNormGated / serving env,66-68TPS |
| **P8** | done | 78.8TPS CUDA graph ceiling + 81TPS compile spike |
| **P9** | done | packed NVFP4 active expert path,逼近 100TPS |
| **P10** | done/current | native FP4 lm_head + full-path 103TPS,runner graph-slot strict gate |
| **P11** | done | packed-resident memory lifecycle,56GiB BF16 shadow release 已证明 |
| **P12** | done/current | one-shot server release gate + release 后 graph slot gate 已过 |
| **P13** | current | graph-slot generate 接线已过;下一步移除 capture hot path / 生产稳定 100+ |
| **P14** | current | state-refresh slot route 0.79ms copy-cost 已验证;实现 reusable graph-owned-state slot |

Spark sm_121 分支单独推进,当前质量 gate 已通过、scalar_bridge 约 24TPS;目标是在 Spark 上验证同一 native path 并冲 50+TPS。详见 [`docs/SPARK_OPTIMIZATION_BRANCH_PLAN_20260515.md`](docs/SPARK_OPTIMIZATION_BRANCH_PLAN_20260515.md)。

**已锁定决策**:
- 推理硬件:**Blackwell sm_12x**(DGX Spark / 5090 / RTX PRO 6000)
- 推理主格式:**Lynn-native NVFP4**(BF16 仅 reference;GGUF/FP8 是兼容/对外测试资产)
- 推理范围:**单 prompt + batch=1**(不做 PagedAttention)
- 模型锁定:**Lynn-27B-A3B variable-pruned family**
- 定位:**vertical companion** 给 Lynn LoRA + 剪枝训练流水线,**不替 vLLM 当通用引擎**

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
- [📝 知乎工程复盘:从零开始 Qwen 3.6 35B-A3B 写专用推理引擎(2026-05-11 mega-post)](https://zhuanlan.zhihu.com/p/2036443846322680848) — Phase 2 + Phase 3.2 + NVFP4 路线决策三段合并,~15-20k 中文字
- [Toolabstain 论文(姊妹项目)](https://github.com/MerkyorLynn/toolabstain-paper)
