# Lynn Engine 工程避坑指南(2026-05-10 实战累积)

> 本文记录 2026-05-10 Lynn engine session 实际踩到的坑 + 修法。
> 优先级:**真踩过的坑**(标 ⭐ 表示踩 ≥2 次/同类多次出现)。
> 不是 best-practice 文章,是事故复盘。

---

## 1. AutoDL 容器环境陷阱

### 1.1 ⭐ `python3` 不在 PATH

```bash
❌ python3 -c "..."
   bash: python3: command not found
   exit 127

✅ /root/miniconda3/bin/python -c "..."
   或:export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
```

**真因**:AutoDL 镜像只装 `/root/miniconda3/bin/python`,**没装 `python3` symlink**。inline 走 PATH 必失败。

**踩坑场景**:
- background watcher 末尾 `python3` parse JSON → exit 127 silent failure
- bash heredoc 内 `python3 << EOF` 同失败
- 任何 `subprocess.run(["python3", ...])` 同失败

**检查模板**(每个新 SSH 命令开头):
```bash
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
```

### 1.2 ⭐ `/etc/network_turbo` + pip 冲突

```bash
❌ source /etc/network_turbo; pip install transformers
   ERROR: Could not find a version that satisfies the requirement transformers
   (网络代理 → PyPI mirror 路由错乱)

✅ unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
   pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
              --trusted-host pypi.tuna.tsinghua.edu.cn
```

**真因**:turbo 代理是给 GitHub/HuggingFace 用的,会污染 PyPI 路由 → pip 找不到包。

**何时用 turbo**:GitHub clone / HuggingFace 下载
**何时禁 turbo**:pip / apt / 任何走 PyPI 的工具

### 1.3 GitHub 直连 HTTP/2 framing 错误

```bash
❌ git clone https://github.com/MerkyorLynn/lynn-engine.git
   error: RPC failed; curl 16 Error in the HTTP2 framing layer

✅ source /etc/network_turbo
   git clone https://github.com/MerkyorLynn/lynn-engine.git   # 2.6 秒搞定
```

**真因**:中国大陆 → github.com HTTP/2 经常被中断,turbo 代理 GitHub 流量稳定。

**ghproxy.com / gh-proxy.com / mirror.ghproxy.com 都不通**(2026-05-10 实测 130s timeout)。AutoDL 学术加速是首选。

### 1.4 `apt update` + 装包

```bash
❌ apt-get install aria2 tmux
   E: Unable to locate package aria2

✅ apt-get update -qq && apt-get install -y -qq aria2 tmux wget
```

容器 apt 索引默认空,需要先 update。

### 1.5 数据盘路径

- ✅ `/root/autodl-tmp` 是 AutoDL 标准的数据盘 mountpoint
- ❌ `/` 是系统盘(只 30GB)
- 模型 ckpt / 大文件 / 实验结果**必须**写 `/root/autodl-tmp`,系统盘 99% 时间应该闲

---

## 2. HuggingFace / 大文件下载陷阱

### 2.1 ⭐ `huggingface-cli` / `hf download` 死锁

```bash
❌ hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ...
   静默死锁 在 xethub CDN read timeout
   留 .lock 文件
   进程死了也不报错

✅ aria2c -x 16 -s 16 -c -k 1M --max-tries=30 --retry-wait=5 \
   https://hf-mirror.com/{repo}/resolve/main/<file>.safetensors

✅ 或用 hfd.sh(hf-mirror 提供,内部用 aria2c):
   ./hfd.sh REPO --tool aria2c -x 10 \
            --include "*.safetensors" "*.json" "*.txt" \
            --local-dir ./LOCAL
```

**踩坑历史**:2026-04-27 / 2026-04-28 同坑两次,2026-05-10 第三次。

### 2.2 hfd.sh `-x` 上限

```bash
❌ hfd.sh REPO --tool aria2c -x 16
   [Error] threads (-x) must be 1-10

✅ hfd.sh REPO --tool aria2c -x 10
```

aria2c 自身支持 `-x 16`,但 hfd.sh wrapper 内部限制 1-10。

### 2.3 hf-mirror 间歇 reset

```
DL 跑中常出 "Connection reset by peer / EOF from server" errors
```

**修法**:`aria2c -c`(continue 续传)+ `--max-tries=30`。aria2c 自动 retry,不影响最终完整性,但 log 会刷红色 ERROR 行,不要被吓到。

### 2.4 Qwen3.6 ckpt 命名约定

```
✗ 老约定 model-00001-of-00009.safetensors + model.safetensors.index.json
✓ Qwen3.6 用:layers-0.safetensors ... layers-39.safetensors + mtp.safetensors
              + model.safetensors.index.json (仍然有,但映射到 layers-N)
```

**别 hardcode `model-XXXXX-of-YYYYY` pattern**,index.json 是 source of truth。

### 2.5 ⭐ NVFP4 量化 ckpt 通常 single-file

```
nerkyor/Qwen3.6-27B-NVFP4-v8-RTN:
  model.safetensors  (18.8 GB, single file)
  ❌ 没 model.safetensors.index.json (HTTP 404)
```

**踩坑**:lynn engine `loader.py` 强制要求 `index.json`,single-file ckpt 撞 `FileNotFoundError`。

**修法**:loader 加 `if index_path.exists()` 二分支,fallback 把 `weight_map = {key: "model.safetensors" for key in st.keys()}` 虚拟出来。

```python
index_path = model_dir / "model.safetensors.index.json"
if index_path.exists():
    # Sharded layout
    weight_map = json.load(open(index_path))["weight_map"]
else:
    # Single-file (NVFP4 v8-RTN typical)
    single_file = model_dir / "model.safetensors"
    with safe_open(single_file, framework="pt", device="cpu") as st:
        weight_map = {key: "model.safetensors" for key in st.keys()}
```

### 2.6 ⭐ `nerkyor/Qwen3.6-27B-NVFP4-v8-RTN` ≠ 27B-A3B stand-in

```
27B-NVFP4-v8-RTN config:
  architectures: ["Qwen3_5ForCausalLM"]    ← Dense
  hidden_size: 5120                         ← vs A3B 的 2048
  intermediate_size: 17408                  ← Dense FFN
  没有 num_experts / mlp.experts.* 字段

→ 是 Qwen3.6-27B Dense 模型的 NVFP4,不是 27B-A3B MoE
→ 不能当 5/15 Lynn-V4-Distill-Qwen-27B-A3B 的测试基准
```

**修法**:5/15 之前要联调 harness,用已有的 35B-A3B-FP8 当 baseline candidate(架构 same MoE,只是 35B 不 27B)。

---

## 3. PyTorch Profiler 测量陷阱

### 3.1 ⭐ API attribute 改名

```python
❌ e.cuda_time_total           # PyTorch 2.x AttributeError
❌ e.self_cuda_time_total      # 同

✅ e.device_time_total         # PyTorch 2.x renamed
✅ e.self_device_time_total

# Sort key 同步:
✅ prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
```

`cuda` 重命名为通用 `device` 是 2.x 趋势。

### 3.2 `with_stack=True` 自身慢 2-4×

```
normal profile mode:    1767 ms / 10 steps = 176 ms/step → 5.66 t/s (slow due to profile overhead)
with_stack=True mode:   4167 ms / 10 steps = 417 ms/step → 2.40 t/s (slower)
no profile baseline:     430 ms / 10 steps =  43 ms/step → 23 t/s ✓ truth
```

**别用 profile mode 的 wall time 当 t/s baseline**。profile 模式的 self_cuda_time 相对值可信,绝对 wall 不可信。

### 3.3 Profile categories 估算只可信相对比例

```
normal mode:     aten::cat = 19.8 ms (15.6% of CUDA total)
with_stack mode: aten::cat = 1.29 ms     (15× 不同!)
```

**真因**:profiler 自身 CPU 开销影响 GPU events 测量;两种 mode 不可直接比较绝对数字,只比 % share。

### 3.4 ⭐ `with_stack=True` 不给 GPU events stack

```python
events = prof.events()
for e in events:
    if e.name == "aten::cat":
        print(e.stack)   # → empty list / None / "<no_stack>"
```

**真因**:`with_stack=True` 只 capture **CPU dispatch events** 的 Python frame。GPU kernel events(`aten::cat` 实际 GPU op)没 stack。

**修法**:
- A. 用 `prof.key_averages(group_by_stack_n=15)` 看 grouped CPU events(但 GPU kernel 仍无 stack)
- B. 解析 chrome trace JSON 找 CPU "ProfilerStep" 父 events
- C. **手动 grep 源码**(我们最终用了这条:`grep -nE "torch.cat|torch.stack|repeat_interleave" engine/*.py` 直接定位)

`prof.export_stacks(path, "self_cuda_time_total")` 在 PyTorch 2.8 写 0 字节,silent fail。

### 3.5 Categories 不能 sum 到 100%

```
我的 profile script 输出:
  Total CUDA time: 253.66 ms (10 decode steps)
  实际 Self CUDA total = 126.86 ms

→ categorize logic double-count parent/child events
→ 单个 category % 可信,sum 到 100% 不能
```

**修法**:用 `prof.events()` 各 events 唯一计 + 仔细 categorize,或直接看 top-N table。

### 3.6 GPU compute time vs wall time

```
profile 给的:Self CUDA time total
wall time:    实际 t/s 测出
diff:          Python orchestration / kernel launch / sync overhead

例:Self CUDA = 12.7 ms/step  (GPU 真活)
   Wall      = 43 ms/step    (实际 t/s)
   Python overhead = 30 ms/step (~70% wall)
```

**关键**:GPU side optimization ceiling = wall - Python overhead。**不解决 Python overhead,GPU 端再优化天花板封死**。

---

## 4. BF16 数值精度陷阱

### 4.1 ⭐ FP32 accumulator promote 是 redundant

```python
❌ 假设 BF16 input 用 BF16 accumulator,要 promote FP32:
   gate_out = torch.bmm(h.float(), w.float()).to(h.dtype)

✅ cuBLAS 处理 BF16 input 时**本来就内部用 FP32 accumulator**(标准 Tensor Core 行为)
```

**踩坑**:bmm 跟 optimized 数值不同,假设是 BF16 accumulator 不够,promote FP32 再试 → smoke test 行为完全不变。

**真因**:不是 accumulator 精度问题,是 cuBLAS 不同 kernel 的 reduce schedule 不同。

### 4.2 ⭐ bmm vs F.linear 不可能 byte-exact

```
F.linear  → cuBLAS gemm                tile size A, schedule X
torch.bmm → cuBLAS gemmStridedBatched  tile size B, schedule Y
两者 FP32 accumulator 都用了
但 reduce 顺序不同(浮点加法不结合律)
→ ε 量级 differences (BF16 epsilon ~ 8e-5)
→ 单 step 看不出
→ 40 层 cascade 累积
→ router top-K borderline 时翻转
```

**这不是 bug**。cuBLAS 算法选择的产物。BF16 物理上让 bmm = F.linear byte-exact 是不可能的。

**修法**:不追求 bmm = F.linear。改对标 HF transformers reference 看哪个 closer。或留 bmm 作 opt-in 不 default。

### 4.3 ⭐ Multi-meaning prompt 是数值 ε 的 canary

```
通过的 prompts(11/14):
  "The capital of France is" → "Paris" (deterministic 续接)
  "def fibonacci(n):" → code 模式 strong gradient
  "import torch" → standard

失败的 prompts(3/14):
  "2+2="           → 模型可能选 "4" / "Let me" / 多个候选 logits 接近
  "Python is"      → 描述性极广
  "I love eating"  → 情感/食物多义
```

**Pattern**:数值 ε differences 在 router top-K logits 拉开时不影响,**borderline 时翻转**。

**Canary prompts 必须包含 multi-meaning prompts**,deterministic-only test 100% pass 是 self-consistent bug 假阳。

### 4.4 ⭐ Single-prompt PASS 永不背书

**Codex review #2 血泪教训**:
- bmm 单 prompt `The capital of France is` PASS 10/10 token
- 据此 default 切 bmm
- multi-prompt 14 prompt 揭示 bmm 78.6% match,3 prompt fail
- → revert default,撤 production-ready 标识

**Universal review rule**:任何 default-impl 切换 / production-ready 标识 / merge-to-main 决策必须以 multi-prompt N≥14 exact-match gate 为唯一依据。**single-prompt PASS 永不背书**。

### 4.5 SDPA `enable_gqa=True` byte-exact 等价 `repeat_interleave`

```python
# Before:
K_attn = K.repeat_interleave(H_Q // H_KV, dim=1)  # 8x mem expand
attn_out = F.scaled_dot_product_attention(q, K_attn, V_attn, ...)

# After (byte-exact, idiomatic):
attn_out = F.scaled_dot_product_attention(q, K, V, enable_gqa=True, ...)
```

**性能差 < 0.5%**(broadcast 跟 explicit repeat 在 BF16 wall time 几乎无差)。**所以 GQA repeat_interleave 不是 GPU 瓶颈**,误诊它会浪费时间。

---

## 5. torch.compile 陷阱

### 5.1 ⭐ `mode="reduce-overhead"` 撞 mutated_inputs

```python
torch.compile(_decode_layer, mode="reduce-overhead")

→ RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run
```

**真因**:`reduce-overhead` 启用 CUDA Graph,**CUDA Graph 不兼容 in-place state mutation**(KV cache write / linear_attn conv state replace)。

**修法**:
- A. 把 state 改 functional(每 step 返回新 state,不 mutate)— **大重构**
- B. 显式 `torch.compiler.cudagraph_mark_step_begin()` between calls
- C. KV cache 用 stable buffer,内部 index slice 写入(避 mutation)

### 5.2 ⭐ `mode="default"` 撞 specialization recompile

```python
_decode_layer(h, pos_int, layer_type, w, cfg, state, layer_idx)
              ^^^^^^^^^                 ^                 ^^^^^^^^^
              Python int               Python dict       Python int

torch.compile 把这些当 specialization keys
→ 40 layers × 2 layer_types = 80+ unique cache entries
→ 撞 cache_size_limit (默认 64)
→ fall back to eager (slow)
→ 9× slower than baseline
```

**修法**(都需要 refactor):
- A. 不传 `layer_idx` / 不传 dict — 改传 individual tensors
- B. 拆 linear_attn / full_attn 两个独立 compiled functions
- C. cache_size_limit 提高到 256(**反而更慢 200×**,因为每 step 重 inductor compile)

### 5.3 cache_size_limit 提高反向加速

```
cache_size_limit = 64:   1.5 t/s (撞 limit fall back)
cache_size_limit = 256:  0.06 t/s (200× slower!)
```

**真因**:不撞 limit 但每 step 仍 trigger compile work,inductor 重新生成 graph cost 高于 eager。

**结论**:不要靠 cache_size_limit "fix" specialization 问题,**根本上需要重构 function 签名**避免 specialize。

### 5.4 torch.compile 数值正确性保留

即使速度反向,torch.compile **数值正确性 byte-exact** 保留:

```
baseline tokens:  [34756, 364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300]
compiled tokens:  [34756, 364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300]
✓ EXACT MATCH (10/10)
```

→ compile 不破数值,只破速度(在错误使用模式下)。

---

## 6. Lynn engine 接口 / wire-in 陷阱

### 6.1 ⭐ Triton kernel wired 但路径不全

```
triton_kernels/rope.py::make_triton_rope          ← 已写
但只 wired in: engine/qwen36_block.py (Phase 2 brute path)
没 wired in:   engine/incremental_decode.py (Phase 3.1 decode path)
```

**踩坑后果**:profile 显示 RoPE 内部 5 个 cat 都来自 incremental_decode 纯 PyTorch 实现,不是 Triton。

**修法**:wire 之前先 grep 实际调用路径,验证 kernel 真在 hot path 上。

### 6.2 ⭐ `generate_incremental` 每次 reload weights

```python
def generate_incremental(model_dir, prompt, max_new=5, ...):
    layer_weights = []
    for i in range(n_layers):
        w, _ = load_qwen36_layer(model_dir, i, ...)  # 每次 12s × 40 layers
        layer_weights.append(w)
    # ... prefill + decode
```

**Multi-prompt benchmark 浪费**:14 prompts × 3 impls × 12s load = 504s = 8.4 min 在 loading。

**修法**:写一个 `generate_incremental_reuse_weights(layer_weights, outside, ...)` 接受预加载权重。但**会 break LYNN_MOE_IMPL pre-stack hook 的 idempotency**(stack 后不能再 stack),需要 careful state machine。

### 6.3 ⭐ pre-stack hook 时序

```
✗ pre-stack 在 prefill 之前 → prefill 用 baseline _moe_forward 仍要原始 expert
                              → KeyError on mlp.experts.{e}.gate_proj.weight

✓ pre-stack 在 prefill 之后 / decode 之前 → prefill 用原 expert,decode 用 stacked
                                          → del 原 expert 释放 60GB
```

Codex 修这个时序 bug 的 patch 是 reference design。

### 6.4 LynnInferenceState 是 mutable dataclass

```python
state.update_full_attn_kv(layer_idx, K_new, V_new, position_start)
  → in-place K[..., t:t+1, :] = K_new

state.update_linear_attn_state(layer_idx, new_S, new_C)
  → dict assignment self.recurrent_state[layer_idx] = new_S
```

**特点**:
- ✅ 内存高效(pre-allocated buffer)
- ❌ 跟 CUDA Graph / functional refactor 不兼容
- ❌ 跟 torch.compile reduce-overhead 不兼容(mutated_inputs)

**修法**:留独立分支慢慢做 functional 改写,不在时间压力下硬冲。

---

## 7. Multi-prompt gate 工程纪律

### 7.1 N=14 是底线,N≥20 是 stress test

```
14 prompt 已经足够暴露 self-consistent bug(prompts 5/6/12 fail)
N=20 加 6 个更 borderline prompts(2+2= 类多义续接)stress 更严

修复后必须 N=20+ 全 PASS 才考虑 default
```

### 7.2 防假绿 watchdog 规则

```
DO    检查 multi_prompt_gate.exact_match_rate < 1.0 → FAIL
DO    检查 multi_prompt_gate.mismatches_count[*] > 0 → FAIL
DO    dump 具体 mismatch prompt 的 GT vs got token 序列
DON'T 信任任何 "verdict": "PASS" 字段(可能错标或缺失)
DON'T 用 single-prompt phase32_bench 数据做 default 决策
```

### 7.3 Multi-prompt 必须包含多义

```
✅ 必含:
  数学 (2+2= / 5+7=)
  描述 (Python is / AI 是 / The meaning of life is)
  情感 (I love eating / I don't know what to)
  极短 (你好)

❌ 全 deterministic prompt(The capital of France is / def fibonacci(n):)
   100% pass 也不算过 — 数值 ε 差异在这些上看不出
```

---

## 9. 工具链快速参考(SSH 命令模板)

### 9.1 标准 SSH 命令前缀

```bash
SSHPASS='<pwd>' sshpass -e ssh -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -p 27302 \
  root@connect.bjb1.seetacloud.com '
  export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
  cd /root/autodl-tmp/lynn-engine
  # ... actual work
'
```

### 9.2 长任务 background

```bash
SSHPASS='...' sshpass -e ssh ... '
  echo "STARTED $(date)"
  while ! grep -q "DONE_MARKER" /tmp/some.log 2>/dev/null; do
    sleep 60
    PROG=$(tail -1 /tmp/some.log 2>/dev/null | head -c 150)
    echo "$(date +%T) $PROG"
  done
  echo "COMPLETE at $(date)"
  tail -100 /tmp/some.log
'
# Bash tool with run_in_background: true
```

### 9.3 跑 multi-prompt gate 模板

```bash
# Sequential per impl per prompt(每次 fresh Python process,12s load)
LYNN_MOE_IMPL=$IMPL PYTHONPATH=. timeout 240 \
  /root/miniconda3/bin/python engine/full_forward.py \
    --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-FP8 \
    --prompt "$P" --max-new 8 --mode incremental
```

### 9.4 写远端脚本(避 heredoc / quote 陷阱)

```bash
# A. Use Write tool locally + scp:
SSHPASS='...' sshpass -e scp -P 27302 -o ... \
  /tmp/local_script.py root@host:/tmp/

# B. printf line-by-line(快速但易错):
echo "#!/bin/bash" > /tmp/foo.sh
echo "cd /tmp" >> /tmp/foo.sh
printf "%s\n" "command with (parens)" >> /tmp/foo.sh   # 用 printf 避 paren expansion

# DON'T:
# bash -c 'cat > /tmp/foo.sh <<EOF ... EOF'   ← ssh stdin / quoting 极脆弱
```

### 9.5 验证 patch 应用

```bash
# 1. Backup
cp engine/foo.py engine/foo.py.bak.<patch_name>

# 2. Apply (Python script with assert)
python << "PYEOF"
src = open("engine/foo.py").read()
assert OLD_BLOCK in src, "anchor not found"
src = src.replace(OLD_BLOCK, NEW_BLOCK)
open("engine/foo.py", "w").write(src)
PYEOF

# 3. Syntax check
python -c "import ast; ast.parse(open('engine/foo.py').read()); print('OK')"

# 4. Smoke test (single prompt) 数值
LYNN_MOE_IMPL=optimized PYTHONPATH=. python engine/full_forward.py ...

# 5. Multi-prompt N=14 gate

# 6. 失败 → revert: cp engine/foo.py.bak.<patch_name> engine/foo.py
```

---

## 11. 速查 — 我已知道的"幻影提速"

不会显著提升 t/s 的优化(踩过 / 推过 / 算过):

| 优化 | 期望 | 实际 | 教训 |
|---|---|---|---|
| SDPA `enable_gqa=True` 替 explicit `repeat_interleave` | -28% memcpy | <0.5% | broadcast vs repeat byte-exact 等价 |
| FP32 accumulator promote(BF16 input) | 修 bmm 数值 | 完全没改变 | cuBLAS 已用 FP32 accumulator |
| `torch.compile(mode="default")` 简单 wrap | 减 Python overhead | 9× slower | layer_idx specialization 撞 cache limit |
| `torch.compile(mode="reduce-overhead")` 简单 wrap | CUDA Graph 加速 | 撞 mutated_inputs runtime error | KV cache mutation 不兼容 |
| Phase 3.3 Triton MoE FFN kernel | unlock 高 t/s | matmul 只占 9% GPU time | GEMM 已不是瓶颈 |
| F2.x cat/copy 全消减 | 25 t/s | 最多 14-16 t/s | Python overhead 30 ms 是绝对 wall time 下限 |

真正能上 25+ t/s 的路径(已知):
- ✅ Python overhead refactor(`_decode_layer` 签名 / functional state / CUDA Graph 兼容)— 1-2 周
- ✅ B1 NVFP4 grouped GEMM(weight 4bit → mem-BW 减半)— 4-6 周
- ✅ 两者一起做(在 DGX)

---

## 修订历史

- **2026-05-10 v1**:首版,基于 Lynn engine 实战 session 累积经验。
  涵盖:AutoDL 环境 / HF 下载 / PyTorch profiler / BF16 数值 / torch.compile / Lynn engine 接口 / multi-prompt gate / 通用工程纪律 / 速查表。
