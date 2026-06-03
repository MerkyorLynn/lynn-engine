# Lynn Engine — Restart Notes 2026-06-03

> **状态校正:** Lynn engine 已从 2026-05-20 的"产品默认后端降级为 R&D"状态重启为并行主线。客户端短期仍可用 llama.cpp/GGUF 作务实默认后端,但 Lynn engine 的目标重新明确为:同模型、同硬件对标 llama.cpp,用自研 native kernels 追赶并最终形成 Lynn 自有核心。

## 公开连载

- 知乎 2026 年 6 月连载: [从零开始 Qwen 3.6 35B-A3B 写专用推理引擎踩坑心得分享](https://zhuanlan.zhihu.com/p/2045562329396400486)
- GitHub 战役文档: [Decode launch-overhead campaign](reports/qwen36_35b/DECODE_LAUNCH_OVERHEAD_CAMPAIGN_20260603.md)
- Stage-6 服务能力: [Decode-only shadow-free serving recipe](reports/stage6/DECODE_ONLY_SHADOW_FREE_SERVING_RECIPE.md)
- 下一关 contract: [Stage 6 Phase-0 packed-prefill / zero-reload](reports/stage6/PHASE0_TRACE_SPEC.md)

## Banked Results

| 项目 | 结论 |
|---|---|
| Decode TPS | Spark NVFP4 35B-A3B `38.96 -> ~45 TPS`,5 个 launch-cut 已 RC 验证 |
| 质量 | 40/40 greedy 输出与 baseline token-identical,继承 MMLU 84.40 / GPQA-Diamond 49.49 |
| MTP | correctness / accept 成立,但 eager speculative runtime 净亏;不是当前 45->70 捷径 |
| Reusable graph | A/B 为 `44.60 -> 33.46 TPS`,净负 0.75x,不进入默认 |
| 60GiB memory win | decode 阶段释放 BF16 dequant-shadow,常驻 `88 -> 28 GiB`,token-exact,TPS 0.998x |
| 服务集成 | `server/openai_http.py` 已接 `reload -> prefill -> release -> decode` cycle 与 health metrics |

## Corrected Engineering Read

6/3 evidence-lock 推翻了"只要写 read-4bit / zero-shadow decode kernel 就能到 70 TPS"的旧前提:

- decode 的 active MoE 已经读 packed 4-bit 并在寄存器里反量化;
- 60 GiB BF16 shadow 是 prefill 专用,decode 可以释放并继续跑;
- full-attn / linear-attn 改 FP4 没有带来速度收益;
- reusable decode CUDA graph 在 Spark 上净负;
- Spark sm_121 没有 FP4 MMA,35B NVFP4 decode 结构性卡在 ~45 TPS 级别。

这不是放弃追 llama.cpp。新的路线更硬也更诚实:追赶 llama.cpp 不能靠 Python loop 或单点开关,需要 native runtime + fused kernels。

## Restart Target

**目标:** Lynn engine = Python 控制面 + C++/CUDA/Triton native kernel 核心。

短期 gate:

1. **P0.1 packed-prefill no-reload smoke:** `LYNN_PACKED_PREFILL_SLOW=1` 只作为 proof harness,验证 BF16 shadow 已释放后无需 reload 仍能从 packed NVFP4 完成 prefill。
2. **P1 batched packed projections:** 替换 row-loop proof,做 full-attn / linear-attn 的 batched packed-NVFP4 prefill projections。
3. **P2 grouped packed MoE prefill:** 写 M>1 grouped expert kernels,消除每请求 ~23s reload。
4. **P3 server promotion:** `LYNN_PACKED_PREFILL=1` 后多请求服务常驻 27-28 GiB,无 reload,decode TPS 不回退。
5. **P4 native-kernel chase:** 继续向 llama.cpp 的低 dispatch / fused ggml CUDA 路线追赶;有 FP4-MMA 硅时兑现 NVFP4 native moat。

## Relation To 2026-05-20 Notes

`RELEASE_NOTES_20260520.md` 保留为历史决策记录,但不再是当前状态权威。当前 GitHub 入口以本文件、README 顶部 6/3 banner、以及 6 月知乎连载为准。
