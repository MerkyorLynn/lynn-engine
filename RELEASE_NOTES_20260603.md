# Lynn Engine — Restart Notes 2026-06-03

> **状态校正:** Lynn engine 已从 2026-05-20 的"产品默认后端降级为 R&D"状态重启为并行主线。客户端短期仍可用 llama.cpp/GGUF 作务实默认后端,但 Lynn engine 的目标重新明确为:同模型、同硬件对标 llama.cpp,用自研 native kernels 追赶并最终形成 Lynn 自有核心。

## 公开连载

- 知乎 2026 年 6 月连载: [从零开始 Qwen 3.6 35B-A3B 写专用推理引擎踩坑心得分享](https://zhuanlan.zhihu.com/p/2045562329396400486)
- GitHub 战役文档: [Decode launch-overhead campaign](reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md)
- Stage-6 服务能力: [Decode-only shadow-free serving recipe](reports/stage6/DECODE_ONLY_SHADOW_FREE_SERVING_RECIPE.md)
- 下一关 contract: [Stage 6 Phase-0 packed-prefill / zero-reload](reports/stage6/PHASE0_TRACE_SPEC.md)
- P0.2 resident inventory: [Stage 6 P0.2 resident BF16 inventory](reports/stage6/P02_RESIDENT_INVENTORY_20260603.md)
- P1 dense projection PoC: [Stage 6 Phase 1 single dense projection PoC](reports/stage6/P1_DENSE_PROJECTION_POC_20260604.md)
- P1-A batched projection result: [Stage 6 Phase 1-A batched projection PoC](reports/stage6/P1A_BATCHED_PROJECTION_POC_20260604.md)
- P1-A tiled projection sweep: [Stage 6 Phase 1-A tiled projection sweep](reports/stage6/P1A_TILED_PROJECTION_SWEEP_20260604.md)
- P2 grouped MoE prefill census: [Stage 6 Phase 2 grouped MoE prefill census](reports/stage6/P2_GROUPED_MOE_PREFILL_CENSUS_20260604.md)
- P2-A single-expert gate/up PoC: [Stage 6 Phase 2-A gate/up prefill PoC](reports/stage6/P2A_GATEUP_PREFILL_POC_20260604.md)
- P2-B routed gate/up grouping PoC: [Stage 6 Phase 2-B routed gate/up grouping](reports/stage6/P2B_ROUTED_GATEUP_GROUPING_POC_20260604.md)
- P2-C active routed MoE lower-bound: [Stage 6 Phase 2-C active MoE lower-bound](reports/stage6/P2C_ACTIVE_MOE_LOWER_BOUND_POC_20260604.md)
- P2-D one-layer MoE hybrid: [Stage 6 Phase 2-D router/shared-inclusive MoE hybrid](reports/stage6/P2D_ONE_LAYER_MOE_HYBRID_POC_20260604.md)
- P2-E scheduler/active retune: [Stage 6 Phase 2-E scheduler active retune](reports/stage6/P2E_SCHEDULER_ACTIVE_RETUNE_20260604.md)
- P2-F opt-in one-layer replacement: [Stage 6 Phase 2-F one-layer replacement verify](reports/stage6/P2F_ONE_LAYER_REPLACEMENT_VERIFY_20260604.md)
- P2-G multi-layer MoE smoke: [Stage 6 Phase 2-G multi-layer MoE smoke](reports/stage6/P2G_MULTILAYER_MOE_SMOKE_20260604.md)
- P2-H selected-layer full prefill smoke: [Stage 6 Phase 2-H selected-layer prefill smoke](reports/stage6/P2H_SELECTED_LAYER_PREFILL_SMOKE_20260604.md)
- P2-I selected-MoE expansion smoke: [Stage 6 Phase 2-I selected-MoE expansion smoke](reports/stage6/P2I_SELECTED_MOE_EXPANSION_SMOKE_20260604.md)
- P2-J linear-attn prefill trace: [Stage 6 Phase 2-J linear-attn prefill trace](reports/stage6/P2J_LINEAR_ATTN_PREFILL_TRACE_20260604.md)
- P2-KA gated-delta recurrent-loop PoC: [Stage 6 Phase 2-KA gated-delta native recurrent-loop PoC](reports/stage6/P2KA_GATED_DELTA_NATIVE_LOOP_POC_20260604.md)
- P2-KB gated-delta block-kernel PoC: [Stage 6 Phase 2-KB gated-delta block-kernel PoC](reports/stage6/P2KB_GATED_DELTA_BLOCK_KERNEL_POC_20260604.md)
- P2-L linear-attn block integration: [Stage 6 Phase 2-L linear-attn block integration smoke](reports/stage6/P2L_LINEAR_ATTN_BLOCK_INTEGRATION_SMOKE_20260604.md)
- P2-M selected-layer block-linear smoke: [Stage 6 Phase 2-M selected-layer block-linear smoke](reports/stage6/P2M_SELECTED_LAYER_BLOCK_LINEAR_SMOKE_20260604.md)
- P2-N wider-layer block-linear smoke: [Stage 6 Phase 2-N wider-layer block-linear smoke](reports/stage6/P2N_WIDER_LAYER_BLOCK_LINEAR_SMOKE_20260604.md)

## Banked Results

| 项目 | 结论 |
|---|---|
| Decode TPS | Spark NVFP4 35B-A3B `38.96 -> ~45 TPS`,5 个 launch-cut 已 RC 验证 |
| 质量 | 40/40 greedy 输出与 baseline token-identical,继承 MMLU 84.40 / GPQA-Diamond 49.49 |
| MTP | correctness / accept 成立,但 eager speculative runtime 净亏;下一步是去掉 runtime overhead / token-exact batched verify,不是重训 head |
| Reusable graph | A/B 为 `44.60 -> 33.46 TPS`,净负 0.75x,不进入默认 |
| 60GiB memory win | decode 阶段释放 BF16 dequant-shadow,常驻 `88 -> 28 GiB`,token-exact,TPS 0.998x |
| 服务集成 | `server/openai_http.py` 已接 `reload -> prefill -> release -> decode` cycle 与 health metrics |
| P0.1 no-reload proof | `stream_bf16` 模式 ALL_PASS:释放后不 reload,peak 40.28 GiB,token-exact,probe prefill 20.75s |
| P0.2 resident inventory | release 后 BF16 resident 只剩 4.72 GiB;最大项为 linear-attn projection 1.884 GiB、embed/lm_head 各 0.947 GiB |
| P1 single dense projection | 真实 `linear_attn.in_proj_qkv` packed Triton matvec ALL_PASS:32.00->10.00 MiB,cos=1.0,no BF16 shadow,160.29us vs BF16 190.03us(1.186x) |
| P1-A naive batched projection | numeric/no-shadow PASS,但 M>1 perf FAIL:M=4/16/64 仅 0.260x/0.067x/0.016x;不 promote |
| P1-A tiled scalar projection | numeric/no-shadow PASS,最多 25.93x 快于 naive,但仍输 BF16 GEMM:M=16 best 0.742x、M=64 best 0.359x;不 promote |
| P2 grouped MoE census | 单层 `stream_bf16` 精确但 0.49-0.51s/layer;small-M verifier peak 0.70GiB 但仍慢 BF16;下一步是真 routed grouped MoE prefill kernel |
| P2-A single-expert gate/up | packed no-shadow component numeric PASS,但 scalar-dequant gate/up 输 BF16:M=16 0.233x、M=64 0.079x;best sweep M=64 0.115x;不 promote |
| P2-B routed gate/up grouping | route-grouped packed gate/up numeric/no-shadow PASS;M64 207 unique experts=20.0ms/layer,比 BF16 gate/up 慢(0.423x)但远低于 stream_bf16 0.5s/layer |
| P2-C active routed MoE | packed gate/up+down active path numeric/no-shadow PASS;M64 23.83ms/layer,约 21x 快于 stream_bf16,但 0.560x vs BF16 active;继续 P2-D |
| P2-D one-layer hybrid | router/shared 加回后 numeric/no-shadow PASS;M64 29.21ms/layer,约 17x 快于 stream_bf16,但 0.741x vs BF16 full MoE;不接 serving,下一步降 router/grouping + packed-active 调度成本 |
| P2-E scheduler/active retune | sort scheduler 把 M64 route/grouping 5.02ms 降到 0.61ms;`block_inter=8` 把 active 24.48ms 降到 20.12ms;hybrid M64 20.25ms vs BF16 21.41ms=1.057x,numeric/no-shadow PASS |
| P2-F opt-in engine path | 新增 `LYNN_PACKED_PREFILL_SLOW_MODE=p2e_hybrid`;删除 active BF16 shadow 后,M64 20.23ms vs BF16 21.10ms=1.043x,24.96x vs stream_bf16,peak 0.643GiB |
| P2-G multi-layer smoke | 4 层 residual-scale MoE smoke PASS;M64 80.97ms vs BF16 85.12ms=1.051x,30.04x vs stream_bf16,numeric pass,peak 2.527GiB vs stream 14.526GiB |
| P2-H selected-layer full prefill | `_prefill_layer` 完整链路 PASS(RMSNorm + linear/full attention cache + MoE);mixed L0-3 T16 45.97ms vs BF16 58.57ms=1.274x,52.09x vs stream_bf16,numeric pass,peak 2.606GiB vs stream 14.585GiB |
| P2-I selected-MoE expansion | mixed L0-7 T16 PASS;88.96ms vs BF16 113.82ms=1.279x,46.70x vs stream_bf16,numeric pass,peak 5.123GiB vs stream 17.102GiB |
| P2-J linear-attn prefill trace | trace exact vs `prefill_linear_attn`;`chunk_gated_delta_with_state` 占 T16..512 traced wall **71-76%**,锁定下一 native kernel 目标 |
| P2-KA gated-delta recurrent loop | numeric PASS(min cosine 0.999989555,argmax match),speed FAIL:T512 native loop 15.62ms vs chunk 4.16ms=0.266x;反证逐 token 复用 decode kernel,下一步 P2-KB 真 chunk/block prefill kernel |
| P2-KB gated-delta block kernel | core kernel PASS:一个 Triton launch 内循环 T token;T512 1.16ms vs host loop 16.28ms=14.04x,vs chunk 4.55ms=3.92x;numeric pass(min cosine 0.999989555,argmax match);下一步 P2-L 接入 `prefill_linear_attn` |
| P2-L linear-attn block integration | opt-in `prefill_linear_attn` PASS:`LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1`;T512 2.18ms vs 5.58ms=2.56x,T16..512 output/state/conv numeric pass(min cosine 0.999983974,argmax match) |
| P2-M selected-layer full prefill | layers 0-3 PASS:P2E MoE + block linear-attn;T16 39.12ms vs BF16 57.40ms=1.467x,T64 84.26ms vs BF16 93.59ms=1.111x,numeric/no-shadow pass |
| P2-N wider selected-layer prefill | layers 0-7 PASS:P2E MoE + block linear-attn;T16 74.98ms vs BF16 115.56ms=1.541x,T64 162.25ms vs BF16 185.03ms=1.140x,numeric/no-shadow pass |

## Corrected Engineering Read

6/3 evidence-lock 推翻了"只要写 read-4bit / zero-shadow decode kernel 就能到 70 TPS"的旧前提:

- decode 的 active MoE 已经读 packed 4-bit 并在寄存器里反量化;
- 60 GiB BF16 shadow 是 prefill 专用,decode 可以释放并继续跑;
- full-attn / linear-attn 改 FP4 没有带来速度收益;
- reusable decode CUDA graph 在 Spark 上净负;
- Spark sm_121 没有 FP4 MMA,35B NVFP4 decode 结构性卡在 ~45 TPS 级别。

这不是放弃追 llama.cpp。新的路线更硬也更诚实:追赶 llama.cpp 不能靠 Python loop 或单点开关,需要 native runtime + fused kernels。

## Restart Target

**目标:** Lynn engine = Python 控制面 + MTP runtime + compiled/C++ decode hot loop + C++/CUDA/Triton native kernel 核心。这里的 hot-loop 降 dispatch 指 **降低每个 launch 的 host/Python 调度成本**,不是继续押已净负的 reusable CUDA graph。

短期 gate:

1. **P0.1 packed-prefill no-reload smoke:已过。** `LYNN_PACKED_PREFILL_SLOW_MODE=stream_bf16` 证明 BF16 shadow 已释放后无需 reload 仍能从 packed NVFP4 完成 token-exact prefill;旧 `decode_kernel` replay memory-clean 但 token 不等价,仅保留诊断。
2. **P0.2 resident inventory:已过。** release 后 BF16 resident 只剩 4.72 GiB,排序为 projection / embed / lm_head / shared-expert;router 只有 0.039 GiB,不是第一杠杆。
3. **P1 single dense projection:已过。** 真实 `linear_attn.in_proj_qkv` 从 packed E2M1 + FP16 scale 直接跑 Triton matvec,numeric/no-shadow/microbench 全过。
4. **P1-A naive + tiled scalar batched projection:已反证。** 两版都数值正确且不读 BF16 shadow,但 M>1 仍输 BF16 tensor-core GEMM;dense M>1 若继续追,必须换 native FP4-MMA/CUTLASS-style bridge。
5. **P2 grouped MoE census:已过。** 单层实测把 20.75s no-reload proof 的主耗时钉死在 `stream_bf16` 全层临时 dequant;small-M verifier 证明 selected-expert 路径可低内存但仍慢。
6. **P2-A single-expert gate/up:组件证据已过但不 promote。** 说明 scalar-dequant packed gate/up 可低内存运行,但追不上 BF16 tensor-core;下一步看 routed grouping 总 launch/unique-expert 成本,不是继续调单 kernel。
7. **P2-B routed gate/up grouping:lower-bound 已过。** 207 unique experts 的 M64 gate/up 为 20.0ms/layer,虽慢于 BF16,但比 stream-dequant proof 低一个数量级;P2 作为 no-reload 服务路径仍成立。
8. **P2-C active routed MoE lower-bound:已过。** packed active path M64 23.83ms/layer,约 21x 快于 stream_bf16 proof,证明 no-reload 服务路径还有价值;但仍慢于 BF16 active。
9. **P2-D router/shared-inclusive one-layer hybrid:混合通过。** numeric/no-shadow 过,M64 29.21ms/layer,约 17x 快于 stream_bf16,但 0.741x vs BF16 full MoE;不接 serving。
10. **P2-E grouped scheduler / active retune:已过。** sort scheduler + `block_inter=8` 后,M64 hybrid 20.25ms/layer,1.057x vs BF16 full MoE,仍保持 no-active-shadow。
11. **P2-F one-layer opt-in replacement:已过。** P2-E 组合已进入 engine dispatch mode `p2e_hybrid`;M64 20.23ms/layer,1.043x vs BF16 full MoE,24.96x vs `stream_bf16`。
12. **P2-G multi-layer no-reload smoke:已过。** 4 层 residual-scale MoE smoke numeric/no-shadow/speed 全过;M64 80.97ms/layer-stack,1.051x vs BF16,30.04x vs stream。
13. **P2-H selected-layer full prefill smoke:已过。** `p2e_hybrid` 已进入完整 `_prefill_layer` 链路;full-attn L3 T64、linear-attn L0 T16、mixed L0-3 T16 均 numeric/no-shadow/speed 通过。
14. **P2-I selected-MoE expansion:已过。** mixed L0-7 T16 继续保持 numeric/no-shadow/speed 通过;P2E 88.96ms,1.279x vs BF16,46.70x vs stream。
15. **P2-J linear-attn prefill trace:已过。** T16..512 trace 精确,`chunk_gated_delta_with_state` 占 71-76%,下一 native kernel 目标明确。
16. **P2-KA recurrent-loop PoC:已反证。** 现有 single-token Triton decode recurrent kernel 可复现 gated-delta 数学(min cosine 0.999989555,argmax match),但逐 token launch 在 T512 只有 0.266x vs chunk reference;不 promote。
17. **P2-KB block-kernel PoC:已过 core gate。** 一个 Triton launch 内循环 T token,修掉 P2-KA 的逐 token launch 失败;T512 1.16ms vs host loop 16.28ms=14.04x,vs chunk 4.55ms=3.92x,numeric pass。
18. **P2-L linear-attn integration:已过。** `LYNN_LINEAR_ATTN_PREFILL_BLOCK_GQA=1` 接入 `prefill_linear_attn`;T512 2.18ms vs 5.58ms=2.56x,T16..512 output/state/conv numeric pass。
19. **P2-M selected-layer smoke:已过。** layers 0-3 同时打开 block linear-attn + P2-E MoE opt-in;T16/T64 numeric/no-shadow/speed 全过,且比 P2E-only 继续加速。
20. **P2-N wider coverage:已过。** layers 0-7 同时打开 block linear-attn + P2-E MoE opt-in;T16/T64 numeric/no-shadow/speed 全过,且比 P2E-only 继续加速。
21. **下一关:** RC/server-promotion smoke;未过前不做 server 默认。
22. **P3 server promotion:** `LYNN_PACKED_PREFILL=1` 后多请求服务常驻 27-28 GiB,无 reload,decode TPS 不回退。
23. **P4C native zero-shadow go/no-go:** P4B single-CTA 和 active recompute 已由 microbench 反证;P4C basic server smoke 已过,但 rc-mini wider agreement 已拒绝(completion exact 3/6,chat 2/2)。P4C 仅保留 opt-in diagnostic/basic smoke;没有 RC-exact 且 e2e A/B 真快过 ~45 的候选,不做 promotion。
24. **Speed next:** MTP draft/accept 信号好,真正 blocker 是 eager speculative runtime;下一轮优先削 snapshot/restore、per-event dispatch/sync 与 token-exact batched verify。另一条是把 decode hot loop 搬出 Python、用 compiled/C++ 服务循环降低 per-launch host dispatch;reusable decode graph 已实测 0.75x 净负,不作为当前默认路线。

## Relation To 2026-05-20 Notes

`RELEASE_NOTES_20260520.md` 保留为历史决策记录,但不再是当前状态权威。当前 GitHub 入口以本文件、README 顶部 6/3 banner、以及 6 月知乎连载为准。
