# 我们亲手关掉了自研的 NVFP4 推理引擎:端侧落地 llama.cpp,前沿不弃 NVFP4

> 接上一篇《重启 NVFP4 引擎:在 DGX Spark 上把 decode 推到 ~45 TPS(+26%)》。这一篇讲那之后发生的事:**私有内核实测证伪 → 生态把 NVFP4 商品化 → 改用上游、并把全网缺的那块数据公开 → 引擎线收口、回归产品。**
> 2026-06 · 速度数字全标硬件与口径,`✅` 为一手实测 / 直查确认。完整数据见文末附录 Gist。

---

## 0. 一句话

**这条自研 NVFP4 引擎线,正式关掉主线。** 不是因为它跑不动——它能跑,上一篇还给它做了 +26%;是因为我们把整条线 profile(性能剖析)到底,得到一张「力学图」:**单流(single-stream,单用户一次一个请求)的杠杆在「格式」、并发(concurrent,多请求批量)的杠杆在「服务系统」,自研引擎在这两条轴上都不是杠杆点**——继续投它,是在推一根推不动、也不该推的杆。(NVFP4 的格式 / 速度 / 质量同期也被 llama.cpp 与 NVIDIA 生态商品化,更坐实了没有差异化空间。)

这不是失败检讨,是一次**让实测改路线**的决策记录:有数据,有三次反转,有自省。

---

## 1. 先接上上一篇:卡在哪、还剩什么希望

- **+26% 是真的**:DGX Spark(GB10 / sm_121,无 FP4-MMA)单流 decode,5 个 RC-validated 内核融合把 Qwen3.6-35B-A3B 从 **~36 推到 ~45 TPS**,**40/40 prompt 逐字一致**(零行为漂移,质量锚点 MMLU 84.40 / GPQA-Diamond 49.49)。
- **但赢面不在我们以为的带宽墙**:探针实测 decode 是 **launch/dispatch-bound** —— 每 token ≈ **1527 次** kernel 启动,只跑到 240 GB/s 带宽的 **37%**,卡的是启动次数不是字节数。「读-4bit / 零-shadow → 70 终局」那个前提被当场证伪;reusable graph 反而净负、把仍是 BF16 的 attn 改读 FP4 更慢。结构性卡 ~45,而**同卡 llama.cpp Q4_K_M 是 69.77**(叠 MTP 的 APEX 配置 79),我们慢 ~1.5×。
- **上一篇留了最后一个希望**:Spark 没有 FP4-MMA,**也许在真有 FP4-MMA 的卡上,我们的跨设备内核就能赢**。

这一篇就从验证这个希望开始——然后看着它,连同整条引擎线,被实测一起收掉。

---

## 2. 把希望押在唯一能证明的硬件上(R6000)—— 实测证伪

我们有一张带 FP4-MMA 的 RTX PRO 6000(sm_120,退租在即),这是「自研内核护城河」唯一能兑现的舞台。最后一搏:写一个 grouped-MoE FP4-MMA 真·快内核,在 P138 fixture 的真实 full-active-MoE **prefill** 边界上,和已有 packed-NVFP4 内核做 A/B。

**干净的 NO-GO:**

| 内核 | 时延 | 相对 |
|---|---:|---:|
| 已有 packed-NVFP4(P3) | **0.0925 ms** | 基准 |
| 新 grouped-MoE FP4-MMA | **0.2668 ms** | **慢 2.88×** |
| W4A16 dequant baseline | 0.602 ms | (只赢它不算赢) |

在 **FP4-MMA 的主场**、用真实边界,新内核比自家旧内核慢近 3 倍,只赢了那个本就慢的 dequant baseline——而「赢慢 baseline 不算赢」是我们自己定的纪律。

**两条物理决定了它赢不了:**
1. **FP4-MMA tensor core 只加速 compute(prefill / 高并发批量 GEMM),不加速单流 decode。** 单 token decode 是带宽-bound,有没有 FP4-MMA 都要把 4-bit 权重读一遍。上游 llama.cpp 也实测:开 native FP4 后 **prefill +45%,decode 持平**。我们想为「单用户 decode」写 FP4-MMA 内核,是在追一个对 decode 物理上不存在的收益。
2. **这个模型的 MoE 本来就是 weight-only(W4A16)。** ✅ 直查 `nvidia/Qwen3.6-35B-A3B-NVFP4` 的 `hf_quant_config.json`:`MIXED_PRECISION`——MoE 是 `W4A16_NVFP4`(4-bit 权重 / 16-bit 激活),attention 和 KV 是 FP8。**根本没有 FP4×FP4 激活路可吃**,B200 上也一样。我们对着一个不存在的 W4A4 主场写了个 W4A4 内核。

至此,「自研内核当护城河」在它唯一能赢的硬件、该赢的场景被实测打死,**没留「再调调说不定行」的尾巴。**

---

## 3. 我们埋头造引擎时,生态把 NVFP4 整个商品化了

这是最该认的一刀。深度调研(扇出搜索 + 一手源对抗式核查)后,结论是 NVFP4 三件套都被生态做完了:

- **格式**:llama.cpp 合并原生 NVFP4 GGUF 类型(`GGML_TYPE_NVFP4=40`,PR #19769,2026-03);Blackwell 上真 `e2m1×e2m1` FP4-MMA prefill 路也合了(PR #22196,2026-04)。
- **质量 + 权重**:NVIDIA modelopt 直接发布**目标模型本尊**官方 NVFP4 checkpoint(`nvidia/Qwen3.6-35B-A3B-NVFP4`)。
- **速度 / serving**:vLLM 有 W4A4 NVFP4 MoE kernel(`CompressedTensorsW4A4Nvfp4MoEMethod` + PR #28892),SGLang 也在追。

**我们想当壁垒的东西,英伟达和开源社区免费送了。** 继续自建 = 跟 llama.cpp + modelopt 正面拼内核,无差异化、无底洞。

---

## 4. 反转:并发的杠杆在「服务系统」,不在内核 —— 改跑 vLLM,实测兑现

把姿态从「造一个更快的引擎」换成「跑生态现成的栈 + 把缺的数据补上」,事情立刻顺了。在**同一张 R6000** 上做双路 serving 验证 ✅:

**赢家 = vLLM forced-Marlin NVFP4**(`VLLM_MOE_FORCE_MARLIN=1`)。并发扫(1024/256):

| 并发 c | 1 | 8 | 16 | 32 | **64(soak)** |
|---|---:|---:|---:|---:|---:|
| 输出 tok/s | 175 | 810 | 1289 | 1872 | **2434** |
| mean TTFT ms | 88 | 304 | 463 | 649 | 1122 |

- **release-grade final soak(2 轮 0 failed)**:c16 均值 1234、c32 均值 1849、**c64 均值 2434**;c8 512/128 交互档 **832 tok/s / TTFT 190ms**。
- **product soak(280 条真实 prompt,0 failed)**:c8 mixed **938 / TTFT 84.7ms / p99 113ms**、c32 mixed **2216 / TTFT 148 / p99 176**、long-ctx c8 814 / TTFT 400ms —— production-grade 延迟实锤(早先 7s TTFT 已调没)。

**同机同 harness 的 NVFP4 vs FP8 A/B**(NVFP4 全档赢):

| c(1024/256) | 1 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| NVFP4 / FP8 | **1.34×** | **1.28×** | **1.14×** | **1.18×** |

而且因为模型 MoE 是 W4A16,**marlin 本就是它的正路,不是「残废 fallback」**——那个「sm_120 native FP4 MoE 出乱码」的雷区主要砸 W4A4 模型,基本不砸我们。**SGLang 当晚 sm_120 NO-GO**(无可用 SM120 kernel wheel;同档 vLLM 1289 vs SGLang ~900,vLLM 快 34%)。

这里有个该说的对比:**我们自研引擎死磕「单流 decode(单 token 生成)峰值」,而生态真正的价值在「并发 serving 吞吐」——靠的是 continuous batching(连续批处理)+ PagedAttention(分页注意力),那是一整个我们还得从零重建的服务系统工程,不是写个快算子能解决的。** 杠杆从来不在内核本身。

---

## 5. 最后一条单流捷径(MTP),也没有免费午餐

不死心,想用投机解码(MTP)在端侧再榨一截。结论又不漂亮:**无损但不提速。**

- 严格 verifier / commit 路径(保证逐 token 无损)比 greedy **更慢**(Spark spec-k1 23.71 vs greedy 34.78,0.68×)——投机环开销吃掉 accept 收益,不能做 product default。
- 病根和生态对得上:**MTP head 按 native-FP4 激活训练,跑在 dequant 路上激活数值不同 → 误预测**。Spark 实测 accept **60.6%**,vLLM 社区 Blackwell marlin 实测 **59%**——两条 dequant 路撞在同一个 ~60%。社区用 bf16-grafted matched head 把 27B 拉到 **87% accept / 1.74×**——那是另一摊 model-artifact 工程,不是调阈值。

所以 **Spark / 端侧单流当前就是 llama.cpp Q4_K_M**(MTP 暂不默认叠);单流提速的下一跳判据不是「谁理论更猛」,是「谁更可能可用且不漂移」——答案是 matched/native MTP,DFlash 降级为 parked fallback。

---

## 6. 踩过的坑 / 自省(给同行的价值)

先把散落各节的实测拢成一张「力学图」——下面每条坑,都是这张图上的一个测量点,它们共同指向同一个结论:

| 测量点 | 实测 | 它证明了什么 |
|---|---|---|
| decode(单 token 生成)kernel launch(内核启动)数 | **1527 次/token**,只吃 **37%** 带宽 | 单流瓶颈是**启动次数 + 内存读取**,不是内核写得够不够好 |
| reusable graph(可复用计算图) | dispatch(调度)压到极致仍**净负** | dispatch 不是瓶颈 → 再雕引擎无收益 |
| 自研 NVFP4 vs llama.cpp Q4_K_M(同卡单流) | Spark **~45 / 69.77**;R6000 **~108-116 / 207** | 自研单流稳输上游 ~1.5×,逼近同一堵带宽墙 |
| FP4-MMA(4-bit 张量核乘加)内核 A/B(prefill 预填充主场) | 0.0925ms vs **0.2668ms(慢 2.88×)** | FP4-MMA 加速 compute 不加速 decode;且模型 MoE 是 W4A16,无 FP4×FP4 激活路可吃 |
| vLLM forced-Marlin NVFP4 并发 | c1 **175** → c64 **2434** tok/s | 并发红利在**服务系统**(continuous batching 连续批处理),不在 kernel |
| NVFP4 vs FP8(同机同 harness) | 全档 **1.14–1.34×** | NVFP4 在并发侧是真红利 |
| MTP(投机解码)accept(接受率) | dequant 路 **60.6% / 59%**(Spark / vLLM 撞同一上限) | 单流提速没免费午餐,瓶颈是 head 训练口径不是阈值 |

**一句话:这不是七个零散数字,是同一张力学图的七个测量点——单流卡在带宽 / 格式,并发卡在服务系统,自研引擎两头都够不着杠杆。** 下面的自省,就是怎么走到这张图的。

1. **「用引擎做护城河」这个前提,我们太晚去证伪。** 内核优化每刀都扎实(gated / RC / token-exact),但**战术上的扎实掩盖了战略上的未验证**:优化了很久「怎么更快」,没早点回答「就算更快,它能当壁垒吗」。**先证伪最贵的前提,再优化细节。**
2. **让实测改路线,别让叙事绑架实测。** 这条线撤回过三次:把 ~40 当硬件上限(撤)、把「带宽墙→70」当终局(证伪)、把「FP4-MMA 卡上能赢」当最后希望(R6000 A/B 打死)。每次都是实测赢、叙事让步。**一个工程组织的健康度,看它撤回自己 roadmap 的速度。**
3. **「赢慢 baseline」不算赢。** R6000 候选只赢 dequant baseline、输给自家 packed——拿「比 baseline 快」自我安慰,就会把死线又续几周。**基准选错,假胜利能骗你很久。**
4. **不要和生态比赛造同一个轮子。** NVFP4 的格式/质量/速度被 modelopt + llama.cpp + vLLM 商品化,是这两年最该读懂的信号:**当一个能力变成生态默认,它就不再是任何人的护城河。**
5. **口径纪律是真资产。** 全程标硬件 / 同卡基线、速度数字分「干净 A/B」和「in-run」两种来源不混、token-exact 闸把 lossy 冒充无损的当场拦下——正是这些纪律,让我们敢相信「NO-GO」这个不舒服的结论,也敢把数据公开给同行复核。

---

## 7. 收口:引擎不是杠杆点,产品才是

把整条线 profile 到底,我们没拿到「快了多少」,拿到的是**一张力学图**——它直接说明了为什么继续投引擎是推错了杆,而**不是「生态做了所以我们不做」**:

**单流轴(single-stream,单用户一次一个请求)—— 杠杆在「格式」,不在引擎。**
decode(单 token 生成)是带宽 + launch(内核启动)的游戏:每 token ≈ 1527 次 kernel 启动、只吃 37% 带宽,reusable graph(可复用计算图)压到极致仍净负——**瓶颈是启动次数与内存读取,不是内核写得够不够好**。4-bit 的红利来自「少读字节」,那是**格式**给的;带宽是物理天花板,llama.cpp + Q4_K_M 已经吃满(同卡 69.77 vs 自研 ~45)。再写 kernel,只是逼近同一堵墙。

**并发轴(concurrent,多请求批量服务)—— 杠杆在「服务系统」,不在内核。**
NVFP4 的真红利在 compute(算力,体现在 FP4-MMA),但兑现它的瓶颈不是算子,是 PagedAttention(分页注意力)+ continuous batching(连续批处理)+ KV 分页 + 调度——一整套服务系统工程,而 vLLM / SGLang 本质就是这套系统(实测 c64 = 2434 tok/s,NVFP4 全档赢 FP8 1.14–1.34×)。自己从零重造它是几年的活,且不是护城河。

**所以收束的真正理由是:profile 证明了自研引擎在两条轴上都不是杠杆点。** 单流的杠杆是格式(用 llama.cpp + Q4_K_M 就拿满),并发的杠杆是服务系统(用 vLLM + NVFP4)。不是打不过生态,是**测明白了力该往哪使**——继续投引擎,是在推一根推不动、也不该推的杆。

落地分工(全部实证收口、零悬念、零待维护):

| 场景 | 选择 |
|---|---|
| 端侧 / 桌面(Mac / Win)单流 | **llama.cpp + Q4_K_M** |
| Spark(sm_121,无 FP4-MMA) | **llama.cpp + Q4_K_M**(暂不叠 MTP) |
| Blackwell server 并发 serving | **vLLM forced-Marlin NVFP4**(c64 实测 2434 tok/s) |
| 自研引擎 / 私有内核 / fork | **关闭** |

前沿 NVFP4 没弃——只是放回它真正起作用的**并发侧**。而这条线真正的收尾,不是关掉一个 repo,是把「整条线实测出来、当时全网还缺的那块数据」,以可复用、可复核的形式**回馈上游**。落到 4 个 PR:

| PR | 内容 | 它把什么从「私有知识」变成「公共证据」 |
|---|---|---|
| **llama.cpp [#24273](https://github.com/ggml-org/llama.cpp/pull/24273)** | NVFP4 转换 / 后端路径 / 基准实测指南文档(open,review 中) | 怎么转、走哪条后端、实测多少——把散落各处的转换口径写成一份官方文档 |
| **vLLM [#44671](https://github.com/vllm-project/vllm/pull/44671)** | Add ModelOpt W4A16 lm_head regression tests | 给 lm_head 量化路径补 regression(回归)门禁,防它被后续改动悄悄改坏 |
| **vLLM [#44672](https://github.com/vllm-project/vllm/pull/44672)** | Document ModelOpt W4A16 NVFP4 Marlin path | 把我们实测跑通的那条 forced-Marlin NVFP4 路径写进官方文档 |
| **vLLM [#44673](https://github.com/vllm-project/vllm/pull/44673)** | Add speculative decoding correctness gate | 把 token-exact(逐 token 一致)无损校验做成 CI 门禁 |

这 4 个 PR 就是整条引擎线的资产去向:**我们没造出护城河,但把「怎么把 NVFP4 用对」这件事,从自己 repo 里的私有内核知识,变成了生态里任何人都能复用、复核的公共证据。** 一条线该有的体面收尾,不是悄悄删库,是留下别人能站上去的台阶。

除了这 4 个 PR,还留下三样真东西:① decode-only 删 shadow(影子权重)→ 常驻显存 **88 → 28 GiB**,给 KV / 长上下文腾 60 GiB,已接进 serving;② 一整套「让实测改路线」的工程纪律 + 跨设备量化知识;③ 想清楚了护城河到底在哪——**不是格式、不是内核、不是几个 TPS,是上面那个会派活的蒸馏编排器(产品本身)**。

做引擎最大的收获,是学会在数据面前关掉自己的引擎——把力气从拧不动的螺丝,搬到杠杆真正在的地方。

> **把一件事 profile 到底,你常会发现真正的杠杆不在你正拧的那颗螺丝上。我们拧了几周引擎,测出杠杆在格式、在服务系统、在产品——于是关掉引擎,把力气搬过去。这不是认输,是终于把力使对了地方。**

---

*附:全口径 benchmark 数据(Spark sm_121 + R6000 sm_120,vLLM/FP8/SGLang/FP4-MMA-A/B/MTP 全表)见 [GIST_nvfp4_benchmark_appendix](./GIST_nvfp4_benchmark_appendix_20260607.md)。*
