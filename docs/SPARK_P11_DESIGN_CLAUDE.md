# Spark P11 设计 — Claude 赛马版

> **Context**: 2026-05-15 用户给我赛马权 — Codex 主线 P11 从 decode-only projection 的 resident ownership 入手拆 BF16 shadow;我可以做我自己的 P11,看谁的 approach 更好。
>
> **目标**: 让 Lynn-native packed NVFP4 文件层 ~20G 红利**真正进入运行显存**,不再被 BF16 shadow 抵消。

---

## Codex P11 推测路径(主线)

从 P10 doc 推断:
- **Decode-only projection 的 resident ownership transfer**: decode hot path 上的 projection layers(linear_attn in_proj / out_proj / MoE gate_up / down)把 ownership 从 BF16 shadow 切到 packed NVFP4
- BF16 shadow 在 prefill 路径上仍保留(prefill 数值精度敏感)
- 类似 R6000 LRU eviction policy:hot decode tensors → packed,cold tensors → BF16 保留或 evict

**优点**:跟 R6000 production 同 architecture,赛马终点同 hardware 对照
**风险**:resident_runner.py 改动大(BF16 shadow 是当前 source-of-truth,改 ownership 等于 invariant 翻新)

---

## Claude P11 — Spark-friendly 替代

**核心洞察**: Spark GB10 = **Grace ARM + Blackwell unified memory architecture**,CPU 和 GPU **物理共享 119GiB**,UMA bandwidth(实测 ~150-300 GB/s 双向,远高于 PCIe Gen5 64 GB/s)。这是 R6000(分离 96G GPU + 独立 CPU RAM)没有的硬件特性。

### 4 个 Spark-specific 切口

#### Path A — UMA Pinned CPU Shadow(我推荐主攻)
- 不 evict BF16 shadow,**把 BF16 shadow 从 GPU 端搬到 CPU 端 pinned memory**(numa 0,Grace CPU 直连)
- Decode hot path: 走 packed NVFP4 in GPU(20G,全部 packed)
- Prefill 或 fallback path: 从 pinned BF16 zero-copy 映射回 GPU(UMA bandwidth)
- **GPU mem 占用**: ~20G packed + workspace ≈ **25-30G**(目标)
- **CPU mem 占用**: ~50G BF16 shadow(pinned)
- 总 RAM: ~75-85G(跟 P10 当前 79G used 接近,但 GPU 端从 ~60G 降到 ~25G)
- 优势:对 OOM 不再敏感,启动 mem budget 从 ≥80G 降到 ≥50G

#### Path B — Just-in-time Streaming Dequant
- **完全删除 resident BF16 shadow**
- 任何需要 BF16 weight 的 path,**from packed NVFP4 on-demand dequant**(每 forward pass 重新算)
- 缺点:重复 dequant cost(~1.5GB/3s on R6000),decode hot path 不能用
- 适用:仅 prefill 偶发场景(每 conversation 一次),分摊后 cost 可接受

#### Path C — Prefill/Decode Hard Split
- Prefill 阶段 BF16 shadow 在 GPU,decode 进入前 **explicit evict shadow**(`torch.cuda.empty_cache` + 重新 alloc 只 packed)
- Decode 整个 loop 不再有 BF16 shadow
- 缺点:multi-turn 对话进入新 turn 又要 prefill,反复 alloc/evict,**fragmentation 风险**
- 适用:单 turn long-form generation

#### Path D — Selective Layer Shadow(混合 A+C)
- 前 N 层(prefill-heavy / token-level features)保留 BF16 shadow GPU side
- 后 40-N 层(task-specialized,decode hot)只 packed
- 利用 Lynn-V4 variable expert skeleton 已经做的 layer-priority decision
- 启发自 27B 剪枝 plan:`layer 2 = 0 cut`(前层 essential)、后层可剪重

### 选择: Path A 主攻

理由:
1. **Spark hardware advantage**: UMA pinned 是 Grace+Blackwell 独有,跟 Codex 主线分化路径有 architectural justification(不是同一份代码 fork)
2. **GPU mem 显著降**: 20G packed 红利**真正落地 GPU**,startup mem budget 从 80G → 50G,启动门槛降 30G
3. **数值精度无损**: BF16 shadow 仍存在(在 CPU pinned),只是位置变了,任何走 BF16 path 的 fallback 都仍能 work,**不是 Path B 那种破坏性删 shadow**
4. **Codex Path 兼容**: 即使 Codex 把 decode projection 切到 packed,我的 Path A 也能 stack 在上面(orthogonal):decode 走 packed,prefill/fallback 走 CPU pinned BF16,GPU mem **double-win**

---

## P11 Path A 实现 plan

### Step 1 — measure baseline
- Spark 当前 79G used / 60G+ 估计是 GPU 端 BF16 shadow + outside weights
- `torch.cuda.memory_allocated()` 跑出来精确 GPU mem 分配数字
- 同 layer 看 BF16 shadow 是哪些 tensor (engine/resident_runner.py 里 ~Layer.{q,k,v,o,gate,up,down}_proj.weight 都是 BF16)

### Step 2 — pinned CPU buffer infrastructure
- 加 `engine/uma_pinned_shadow.py`:
  - `pin_bf16_shadow_to_cpu(layer_weights)`: 从 GPU `torch.Tensor` clone 到 CPU pinned `torch.empty(..., pin_memory=True)`
  - `gpu_view(pinned_tensor)`: zero-copy view back to GPU via `.to('cuda', non_blocking=True)`(UMA 应该 0 拷贝)
- Lynn-V4 27B 模型 layer dict 接入(resident_runner.py 加 `--shadow-on=cpu` opt-in flag)

### Step 3 — verify zero-copy
- Smoke test: pinned tensor `.data_ptr()` vs `gpu_view.data_ptr()` 在 Spark 上是否同 address(UMA)
- 如果是 — 零搬运成本
- 如果不是(实际 copy)— 测 copy bandwidth + decide 是否值得

### Step 4 — production smoke
- 起 server with `--shadow-on=cpu`
- 验证 `torch.cuda.memory_allocated()` 从 ~60G → ~25G
- 跑 TPS benchmark,期望 **TPS 不退化 ≥ 25-30**(目标用户给的)
- 数值 parity:对照同 prompt BF16 shadow 在 GPU 时的输出,top-1 一致

### Step 5 — Spark-only 数字 + 与 Codex 主线对比
- 同 prompt 三测:
  - **Codex P11 (主线)**: decode-only projection packed ownership(R6000 上 / 或我 merge 到 Spark)
  - **Claude P11 Path A**: UMA pinned CPU shadow(Spark only)
  - **Stacked**: Codex P11 + Claude Path A 共存
- 看 GPU mem 占用 vs TPS,哪个 Pareto frontier 优

---

## 时间线

- **Tonight**: Step 1 baseline 测量 + Step 2 pinned buffer 原型
- **Tomorrow**: Step 3-4 production smoke + Spark TPS 数字
- **Days 3-4**: Step 5 与主线 Codex P11 对比

---

## 风险

- Spark sm_121 PyTorch UMA pinned 实测 zero-copy 不一定**纯 zero-copy**(may still copy via PCIe ring buffer even on UMA)— Step 3 必须 verify
- `non_blocking=True` 异步 copy 跟 CUDA graph capture 可能冲突(memory `feedback_thinking_loop` 那类 trap)
- 如果 baseline GPU mem 大头不在 BF16 shadow 而在 KV cache / activation,Path A 收益小,需要 readjust 到 Path D

---

## 不做的事

- **不动 Codex 主线**(R6000 production):任何修改都在 `engine/uma_pinned_shadow.py`(新文件)+ resident_runner.py 加 opt-in flag,不破坏 BF16 shadow 默认路径
- **不混 Spark 数字进 Lynn engine 主线宣传**:遵守 `feedback_spark_branch_isolation_from_engine_main_20260515.md` 边界
