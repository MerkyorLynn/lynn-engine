# 重启 NVFP4 专用推理引擎:在 DGX Spark 上把 Qwen3.6-35B decode 推到 ~45 TPS(+26%)

> 一场针对「内核启动开销」的内核战役 —— 以及我们踩过的坑、得到的经验、和接下来要啃的硬骨头。
> 2026-06-03 · Lynn engine 工程复盘 · 全文数据均为 **DGX Spark(GB10 / sm_121)单流 decode** 实测口径。

---

## 0. 一句话

今天把 Lynn 自研 NVFP4 推理引擎的主线重启,目标很窄:**在 Spark 这块没有 FP4 MMA 的卡上,把 Qwen3.6-35B-A3B 的单流 decode 速度啃上去**。一天下来,从战役起点 **~36 TPS 推到 ~45 TPS(+26%)**,而且**质量逐字等价**——5 个独立的内核优化,每一个都过了「token 相干 + 干净 e2e A/B + 多样性回归」三道关。

> 口径诚实交代:**~36** 是本次战役的起跑配置;我们 5/18 文档里记录的 W4A16 baseline 是 **38.96**,所以相对那个锚点是 **+~16%**。本文用 36→45 讲战役全弧。同硬件 llama.cpp Q4_K_M 是 **69.77**——它仍领先,后面会诚实拆解为什么、以及我们的终局路线。

---

## 1. 为什么重启 NVFP4 专用引擎

Lynn engine 是给 NVIDIA Blackwell 写的、锁定 Lynn 自家 NVFP4 格式的**单模型专用**推理引擎。前段时间产品侧默认推理 pivot 到了 llama.cpp 生态(跨平台 + Q4_K_M GGUF 成熟),engine 降级为 R&D 探索线。

但「降级」不等于「放弃」。回主线的门槛我们自己定得很硬:**同硬件同模型,速度接近或超过 llama.cpp,且质量有不可替代优势**。今天就是冲着这个门槛,重新把 decode 速度这条最硬的线捡起来。

---

## 2. 诊断:decode 到底卡在哪

直觉上 NVFP4 比 4-bit GGUF「更先进」,应该更快。实测打脸:同卡 llama.cpp Q4_K_M 69.77,我们 38.96,差 ~1.8×。**先搞清楚卡在哪,再动手。**

**第一刀诊断 —— 不是 bytes,是 launch。** 我们写了个 launch census(用 16-token vs 32-token 的 per-kernel 调用数之差,消掉 prefill/warmup 常数):

> **decode 每个 token ≈ 1527 次 CUDA kernel 启动。** 实测显存带宽 240 GB/s,而我们只跑到其 37% —— 也就是说**token 时间里有相当一部分不是在搬数据,是耗在 CPU 端的 kernel dispatch 上**(M=1 decode 的内核都很小,launch 开销盖不住、全暴露出来)。

三个「纯流量」杠杆全是死的(reusable graph −20%、qkv 融合 +0.3%、packed-4bit-attn −0.6%)——进一步印证瓶颈在 launch/调度,不在字节数。

**后来更完整的认识是「两堵墙」**:① launch 墙(1527 次/token);② 带宽墙(我们 decode 时权重被 dequant 成了 BF16 shadow,读的是 2× 字节,这把上限压在 ~40)。今天的战役主要拆的是第一堵 + 一部分第二堵;真正把第二堵推倒是终局(见 §6)。

---

## 3. 战役:5 个 RC-validated launch-cut

打法很朴素:**把每层/每专家的一堆小 launch,逐簇融合成更少更胖的内核**;每一刀都必须过关才计入战栈。

| 刀 | 做了什么 | 单刀 A/B 增益 |
|---|---|---|
| **fused RMSNorm** | ~92 个 norm 站点各从 6-8 个 eager 算子融成 1 个 Triton launch(归一化占了启动开销约一半) | **+8.7%**(最大头) |
| full-attn 融合 | qk-norm+rope+KV cache-write 融成 1 launch、gate-sigmoid+transpose fold | +0.6%(**token-exact**) |
| **shared-expert 融合** | gate_up→SwiGLU→down→gate→add 融进一个内核 | **+2.8%** |
| **linear-attn g/beta-fold** | 把 `β=sigmoid(b)`、`g=neg_exp·softplus(a+dt)` 折进 recurrent 内核内部(per-head 在寄存器里算),消掉 ~4 个微 launch/层 × ~38 层 | **+2.6%** |
| **NVFP4 bf16-out copy-elision** | `_scaled_mm` 直接输出 bf16(原本 fp16→`.float()`→fp32→`.to(bf16)`,每投影 **2 次拷贝**),消掉 per-projection 拷贝 | **+3.3%** |

(单刀增益是各自基线上的,**非线性叠加**;累计 ~36 → ~45。)

**质量怎么保证不掉?** 这是重点。这些融合多数**不是 bit-identical**(reduction order / fp32-vs-bf16 末位差异)。我们建了个 RC(release-candidate)回归 battery:在 structured / V9 数学 / GPQA / tool-call / long-form 五个 suite 上,跑 baseline(全关)vs 融合栈(全开),逐 prompt 比 greedy 输出。结果:

> **40/40 prompt 输出与 baseline 逐字一致**,各 suite 分数完全相同 —— 融合栈**行为等价**,继承既有质量锚点 **MMLU 84.40 / GPQA-Diamond 49.49**。

这比「accuracy 匹配」更强:输出逐字相同 = 零行为漂移,连「补偿性错误」的空间都没有。

---

## 4. 我们踩过的坑(经验)

这一节才是复盘的价值所在。

1. **profile 的 section 计时会被 cuda-sync 灌水。** 早期某段「2.2ms」其实是同步假象。**教训:只信干净的 e2e A/B(同进程、toggle 一个 env、取 3 次最大),不信 section delta、也不信孤立 microbench。**

2. **别把软件瓶颈当成硬件上限。** 中途我一度把 ~40 TPS 写成「Spark 硬件上限」。实测带宽 240 GB/s、只跑到 37% 后**撤回**了——这是软件(launch + BF16 shadow),不是硬件。**口径错了要当场认、当场改文档。**

3. **`TOKEN_EXACT=False` ≠ 质量回归。** 非 bit-identical 的融合在单个长生成上可能末位分叉,但在 40 个多样 prompt 的 RC battery 上仍然 40/40 逐字一致。**单 prompt「相干」不够,必须多样 battery + 多 suite + agreement,才敢说「不掉能力」。**

4. **引擎跑在 docker 里,挂载 home 目录会污染依赖。** 把 `/home` 挂进容器后,用户 site 里的 `huggingface-hub 1.12` 盖掉了容器自带的、和 `transformers` 配套的版本,直接 import 崩。**`PYTHONNOUSERSITE=1` 修好** —— 容器化环境务必隔离 user-site。

5. **最深的一个坑:把「有意禁用」误判成「事故」。** 查一个伪-tool-call 检测器时,我看它被改成了空壳,判定是 TS 迁移的事故、动手「修」回去——结果**测试套件当场报红**:那些测试是**专门改写**来断言「原样放行」的(v0.79.3 有意把抑制改成 pass-through,避免把「正常解释里出现的工具语法」误判成幻觉而抑制掉正文)。**教训:看到功能被禁用,先查 git 历史 + 测试断言意图,别假设是 bug。** 后来用「observe-only 采集器」(只读、记日志、绝不改输出)拿到了想要的语料,同时尊重那个有意的设计——误报进语料无害,进控制流就毁体验。

6. **跨境 frp 隧道会吞 ssh 输出。** 后台 `docker run` 经常输出丢失。**check-then-relaunch + `docker run -d` 拿 container id** 是稳的姿势。

7. **凭感觉估 launch 会错一个量级。** 战役早期估「~140 launch/token」,census 实测 **1527**。**别拍脑袋,要实测。** 同样,census 还证明 **MoE/router 早就是 grouped 融合的(每层一刀,不是 per-expert×8)**——所以那个「高风险 router 融合」对减 launch 已经没意义,**省得我去啃一个高风险又没收益的大坑**。

---

## 5. 方法论

- **多 CLI 协作的「黑灯工厂」**:codex / claude-internal 在各自进程里写内核初版,lead(我)负责集成 + 在 GPU 上验证(编译/相干/token-exact/e2e A-B/RC)。写核的不落 GPU,落 GPU 的只有 lead。
- **每刀都 gated + 默认关 + 可回滚**:任何融合先 opt-in,过了 RC 才考虑设默认。出错零代价。
- **诚实口径**:Spark-only 标注、不 overclaim、baseline 锚点写清楚。速度数字分「干净 A/B」和「RC 内 in-run」两种来源,不混。

---

## 6. 诚实对标 llama.cpp:为什么它强,我们的终局

同卡 Q4_K_M 69.77 仍领先 ~1.5×。**根因不是它用了 FP4 MMA(它也没用,GB10 没这硬件)**,而是:

- **它手写的融合核「直接读 4-bit 权重 → 寄存器内反量化 → bf16 GEMV」**,内存只走 4-bit(M=1 decode 的瓶颈就是权重带宽),单 launch、零 BF16 shadow。
- **我们还卡在 BF16 dequant-shadow**:很多矩阵乘读的是 BF16(2× 字节),38.96 ≈ 6GB/token ÷ 240GB/s ≈ 40 —— 一堵纯带宽墙。

**终局路线已经清晰**:写 Lynn 自己的「**读 4-bit + 寄存器反量化 + bf16 GEMV + 零 shadow + 单 launch**」NVFP4 内核。它一刀拆两堵墙:带宽从 2× 降到 1×(墙从 ~40 推到 ~140,**70 就活在这区间**)+ 顺手把 dequant/cast 的 launch 也融掉。

为什么这是机会不是难题:**llama.cpp 是 MIT 协议,蓝图开源、可 clean-room 参考**;它能搞,我们不仅能搞,还多两张牌——**同一套 NVFP4 权重 + 内核,挪到有 FP4 MMA 的 R6000 上直接变 native 更快**(跨设备核心资产),且 NVFP4(E2M1+E4M3)是更硬件对齐的格式。**啃下来 = Lynn 成为 NVFP4 的 llama.cpp。**

---

## 7. 后续规划

1. **Stage 5 —— MTP(投机解码,本轮结论:wiring 对、sidecar 对齐待修,不判负)**:用训好的 sidecar(`base` 和 `-official-lynn-fused` 两个变体都测了),K≥2 叠在 ~45 栈上。**实测:`TOKEN_EXACT=True`(verify / reject / rollback 一条链正确,最终输出与非-MTP 逐字一致),但 accept 仅 2/82 ≈ 2.4% → effective ~20 TPS —— 本轮是回归,不是加速。** 关键区分:**这不是引擎 correctness 问题,是 draft head ↔ serving 的对齐问题**。hidden 源已确认是 pre-final-norm(对);两个 sidecar 同样 2.4%(说明是系统性、非变体问题),嫌疑收敛到 **offset 错位**(训练契约 offset=2,serving 可能当 offset-1 用,draft 几乎必 reject → 2.4% 正是"基本随机"特征)。**为什么不判负**:llama.cpp 的 APEX-MTP 跑**同一个 head + 同模型**拿到 +13%(79 vs 69.77,accept 60%+)—— 证明 head 是好的、+13% 可达,我们差的只是 serving 对齐。**修复入口**:`engine/mtp_sidecar.py::mtp_logits` 的 offset 契约 vs serving 循环里 draft 的定位;先加一个 draft-vs-actual 的 ±1 offset 探针确认 shift 方向,再改(多半是配置级,不用重训)。修好预期 ~45→~51。
2. **Stage 6 —— 终局 fused 4-bit / 零-shadow 内核**(§6):多天真内核工程,分阶段啃(单投影 PoC → 全 dense 投影 + 删 shadow → MoE grouped 专家 → 融合减 launch),每阶段 gate + RC。这是把带宽墙推倒、真正逼近(乃至在 R6000 上超过)llama.cpp 的主路。

> 一句话收尾:今天的 +26% 是「拆 launch 墙」拿到的;真正的大头在「拆带宽墙」,那是我们要独立啃下来的护城河。**做引擎,要做就自己把内核啃下来。**
