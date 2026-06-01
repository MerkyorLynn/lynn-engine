# 过夜结论 — Lynn Engine FP8 9B Spark 复活 (2026-06-01 夜)

## TL;DR(诚实版)
**9B dense FP8 在 Spark 上的天花板探到底了:~15 TPS,内存瓶颈。** 今晚证明的"容易的杠杆"都不够:
- **CUDA graph(M3)对 9B 无效** —— graph 跑通且 token 完全一致,但 14.7 vs 15.3 baseline(略慢)。dispatch 只占 ~10%,不是主成本;9B dense decode 是**内存瓶颈**(FP8 8-bit 权重读取 ~9GB/token)。
- **K=1 MTP 只在高 accept 时 +10%**(0.90→16.9;0.45→10.1 反亏)。
- **K≥2 MTP 受限于 head 没训多步**(1 层 head 只会 1-ahead,链式做不了 2-ahead → accept≈0)。这是训练任务,不是代码 bug。
- **对比基线**:llama.cpp 9B Q4_K_M = 36.8 TPS(AR)/ 60.95(MTP n_max=4)。Lynn 9B FP8 ≈ 15 → **落后 ~2.4×**。差在:① 8-bit(llama.cpp 4-bit,内存 1.7×)② Python/kernel overhead(graph 只去掉 ~6.6ms)。

**结论:9B dense 不是 Lynn engine 该打的战场 —— 那是 llama.cpp 对小模型 M=1 极致优化的主场。** 要在 9B 上超它,只有两条大路:**(A) 完整 native/C++ 解码栈**(无 Python、4-bit fused kernel —— 真正"啃下来"的路,数月级)或 **(B) 训练多步 draft head**(达到 llama.cpp 的 K=4 MTP 量级)。两者都不是过夜能成的。

## 今晚证明 / 交付(真实工程财产,已推 GitHub)
分支 `claude/fp8-9b-revival-graph-mtp-20260601`(2 个 commit):
1. **引擎从"只支持 35B MoE"扩成 dense-capable**:`mtp_sidecar` 嵌入式 MTP 加载(FP8→BF16 dequant)+ dense-aware 层;`full_forward` 的 `_decode_layer_k2/_decode_layer_block` 加 dense-FFN 分支;`incremental_decode` fixed-shape full-attn。
2. **可复用 decode graph(capture-once/replay-many)** `_capture_reusable_decode_graph` + `LYNN_REUSABLE_DECODE_GRAPH=1` —— token-exact 跑通,是引擎的**新能力**(对 35B MoE 有用)。
3. **K=1 batched MTP 在 dense FP8 上跑通**(16.9@0.90 accept 超 baseline)。
4. FP8 路径**数值无误**(输出连贯;repack cos>0.999)。

## 推荐路线(给清醒时决策)
1. **9B 产品服务继续用 llama.cpp** —— 它已是该硬件该模型的最优,Lynn engine 在 9B M=1 上短期赢不了。
2. **Lynn engine 主战场转 35B-A3B MoE** —— 这才是 Lynn 能差异化的地方:
   - 35B MoE 每 token ~480 launch(vs 9B dense 少得多)→ **今晚的 reusable graph 在这里才真正去 dispatch**。
   - llama.cpp 的 35B 也只有 ~70 TPS(不是 9B 那种 200+)→ 差距可追。
   - Lynn 的 NVFP4 质量 + grouped FP8 GEMM(task#3)+ APEX-MTP(已有 35B 多步 sidecar!)是组合拳。
   - **关键:35B 已有训好的 MTP sidecar**(`mtp_sidecars/qwen36-35b-a3b-mtp`),不像 9B 缺多步 head → K≥2 MTP 在 35B 上可直接用。
3. **若坚持要 9B 引擎超 llama.cpp**:需 (A) 训 9B 多步 draft head,或 (B) native fused 4-bit kernel 栈。都是独立的多周项目。
4. **flash-attn 动态 seqlen kernel**:让 reusable graph 去掉全窗口惩罚 → graph 净正(对 35B 也有用)。

## 未做 / 数据缺口
- **质量回归 500/198 没跑**:Spark 上缺 GPQA 源 csv(只有 nemotron 的结果 jsonl,无题干+选项)。MMLU 数据齐(`/home/merkyor/lynn-nemotron-eval/mmlu_csv/data`)。需下载 GPQA Diamond 源 csv 才能跑 canonical。FP8 质量低风险(连贯 + cos>0.999),但 canonical 验证待补。
- APEX `:18098` 全程没动(过夜无需停;单模型 job ~83G 安全)。
