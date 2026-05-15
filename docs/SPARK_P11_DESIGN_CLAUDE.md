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

**核心洞察(2026-05-15 23:43 更正)**: Spark GB10 = **Grace ARM + Blackwell unified memory architecture**,CPU 和 GPU **物理共享 同一 119GiB pool**,**不是分离 GPU mem + 独立 CPU RAM**。

**关键证据**:`nvidia-smi --query-gpu=memory.used` 在 Spark 上返回 `[N/A]`(因为没有"独立 GPU 显存"概念可单独查)。

### 这颠覆了我原先 Path A 设计:
- ❌ **错的方向**:"把 BF16 shadow 从 GPU 端搬到 CPU 端 pinned memory" — UMA 下 GPU mem = host mem 同一 pool,搬位置不省任何 physical memory
- ✅ **修正方向**:**删 BF16 shadow,resident only packed NVFP4 ~20G** — 直接节省 ~30-50G 统一 mem 占用(因为不再需要 ~50G BF16 shadow 这个中间产物)

这跟 Codex P11 思路同向(都是删 shadow,只剩 packed),**赛马在实现细节**:
- prefill 仍需要 BF16 数值精度 → 谁的兜底策略更稳?
- decode hot path 切 packed → 谁的 ownership transfer 不破坏 invariant?
- BF16 shadow 删之后多次 prefill(multi-turn 对话)怎么避免重复 dequant?

### 修正后的 4 个 Path

#### Path A — 完全删 BF16 shadow + Just-in-time dequant for prefill(主攻)
- Resident memory **只 packed NVFP4 ~20G**(从 ~50G BF16 shadow → 0)
- Decode hot path: 走 packed NVFP4 + 已有的 `.packed` alias path(P10 实现)
- Prefill 路径需 BF16: **从 packed on-demand streaming dequant 一个 chunk 用完即释放**(不 cache,不 resident)
- 总统一 mem: ~20G packed + workspace ~15G = **~35G total**(vs 当前 ~96G)
- **节省 ~60G 统一 mem**,startup mem budget 从 ≥80G → ~50G

#### Path B — Selective Layer Shadow(分层混合)
- 前 N 层(prefill-heavy / token-level features)保留 BF16 shadow
- 后 40-N 层(task-specialized,decode hot)只 packed,**delete BF16**
- 启发自 27B 剪枝 plan(memory `project_lynn_27b_pruning_plan_0509.md`):`layer 2 = 0 cut`(前层 essential),后层可剪重 → 前层保 shadow,后层 kill shadow
- 节省:~50% × 50G = ~25G

#### Path C — Multi-tier Cache(prefill BF16 cache LRU)
- 不 resident BF16 shadow,但保留 **LRU BF16 cache pool**(e.g. 10G)
- Decode 不 touch shadow,prefill 第一次 dequant + 进 cache,后续 prefill 命中 cache
- 适用:多 turn 对话或同 prompt 重测场景
- 实现复杂度:中

#### Path D — vs Codex 主线赛马差异化
- Codex 推测做 "**decode-only projection ownership transfer**" → decode 路径 packed,prefill 路径 BF16 shadow 仍保留(部分或全部)
- 我做 Path A "**全删 BF16 shadow + streaming dequant for prefill**" → 更激进,GPU mem 节省更多,prefill 慢一点
- 赛马维度:谁的 GPU mem 数字更低 + TPS 保住 + 数值 parity 一致

### 选择: Path A 主攻

理由:
1. **节省最多 mem**: ~60G vs Codex 推测 ~30G(decode projection only)
2. **代码改动单点**: resident_runner.py 在 load 阶段不 dequant 即可,decode path 已经走 packed `.packed` alias
3. **数值 parity 容易保**: prefill 路径仍走 BF16(streaming dequant 出来的),token-level 输出跟 BF16 shadow 完全一致
4. **prefill cost 可接受**: 每 prompt 第一次 prefill ~5-10s extra(40 layer × per-projection dequant),后续 decode hot path 不影响,multi-turn 优化在 Path C 上叠加

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
