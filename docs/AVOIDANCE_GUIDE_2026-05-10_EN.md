# Lynn Engine Engineering Pitfalls Guide (2026-05-10)

> [中文版](AVOIDANCE_GUIDE_2026-05-10.md)

> Engineering pitfalls actually encountered during a 2026-05-10 Lynn engine
> session. Not a best-practices article — a postmortem reference list.
> Items marked ⭐ were hit ≥ 2 times or warrant extra caution.

---

## 1. AutoDL container environment pitfalls

### 1.1 ⭐ `python3` not in PATH

```bash
❌ python3 -c "..."
   bash: python3: command not found
   exit 127

✅ /root/miniconda3/bin/python -c "..."
   or:  export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
```

**Cause**: AutoDL image only ships `/root/miniconda3/bin/python` and
**does NOT install a `python3` symlink**. Inline calls hitting PATH fail.

**Pitfall scenarios**:
- Background watcher final `python3` JSON parser → exit 127, silent failure
- bash heredoc with `python3 << EOF` fails the same way
- Any `subprocess.run(["python3", ...])` fails

**Standard prefix** for every new SSH command:
```bash
export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
```

### 1.2 ⭐ `/etc/network_turbo` conflicts with pip

```bash
❌ source /etc/network_turbo; pip install transformers
   ERROR: Could not find a version that satisfies the requirement transformers
   (proxy + PyPI mirror routing collision)

✅ unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
   pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
              --trusted-host pypi.tuna.tsinghua.edu.cn
```

**Use turbo for**: GitHub clone / HuggingFace download
**Disable turbo for**: pip / apt / any tool routing through PyPI

### 1.3 GitHub direct connection HTTP/2 framing errors

```bash
❌ git clone https://github.com/MerkyorLynn/lynn-engine.git
   error: RPC failed; curl 16 Error in the HTTP2 framing layer

✅ source /etc/network_turbo
   git clone https://github.com/MerkyorLynn/lynn-engine.git   # 2.6s
```

`ghproxy.com` / `gh-proxy.com` / `mirror.ghproxy.com` were all unreachable
(2026-05-10 measurement: 130s timeout). AutoDL academic acceleration is
the reliable path.

### 1.4 `apt` requires `update` first

```bash
❌ apt-get install aria2 tmux
   E: Unable to locate package aria2

✅ apt-get update -qq && apt-get install -y -qq aria2 tmux wget
```

### 1.5 Data disk path

- ✅ `/root/autodl-tmp` is the standard AutoDL data disk mountpoint
- ❌ `/` is system disk (only 30GB)
- Model checkpoints / large files / experiment results **must** go to
  `/root/autodl-tmp`; system disk should sit ~99% empty

---

## 2. HuggingFace / large file download pitfalls

### 2.1 ⭐ `huggingface-cli` / `hf download` deadlock

```bash
❌ hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ...
   silent deadlock at xethub CDN read timeout
   leaves .lock files
   process death is not reported

✅ aria2c -x 16 -s 16 -c -k 1M --max-tries=30 --retry-wait=5 \
   https://hf-mirror.com/{repo}/resolve/main/<file>.safetensors

✅ Or use hfd.sh (provided by hf-mirror, internally aria2c):
   ./hfd.sh REPO --tool aria2c -x 10 \
            --include "*.safetensors" "*.json" "*.txt" \
            --local-dir ./LOCAL
```

### 2.2 hfd.sh `-x` upper limit

```bash
❌ hfd.sh REPO --tool aria2c -x 16
   [Error] threads (-x) must be 1-10

✅ hfd.sh REPO --tool aria2c -x 10
```

aria2c itself supports `-x 16`, but the hfd.sh wrapper limits to 1-10.

### 2.3 hf-mirror intermittent reset

```
Frequent "Connection reset by peer / EOF from server" errors during large
downloads.
```

**Mitigation**: `aria2c -c` (continue/resume) + `--max-tries=30`. aria2c
auto-retries; final integrity is unaffected, but logs show many red ERROR
lines — don't be alarmed.

### 2.4 Qwen3.6 checkpoint naming convention

```
✗ Old convention: model-00001-of-00009.safetensors + model.safetensors.index.json
✓ Qwen3.6 uses:   layers-0.safetensors ... layers-39.safetensors + mtp.safetensors
                  + model.safetensors.index.json (still present, mapping to layers-N)
```

**Don't hardcode** the `model-XXXXX-of-YYYYY` pattern; `index.json` is the
source of truth.

### 2.5 ⭐ NVFP4-quantized checkpoints are typically single-file

```
nerkyor/Qwen3.6-27B-NVFP4-v8-RTN:
  model.safetensors  (~18.8 GB, single file)
  ❌ no model.safetensors.index.json (HTTP 404)
```

**Pitfall**: Lynn engine `loader.py` previously required `index.json`,
hitting `FileNotFoundError` on single-file ckpts.

**Fix**: loader adds an `if index_path.exists()` two-branch fallback,
synthesizing `weight_map = {key: "model.safetensors" for key in st.keys()}`
when the index file is absent.

```python
index_path = model_dir / "model.safetensors.index.json"
if index_path.exists():
    # Sharded layout
    weight_map = json.load(open(index_path))["weight_map"]
else:
    # Single-file (NVFP4 v8-RTN typical output)
    single_file = model_dir / "model.safetensors"
    with safe_open(single_file, framework="pt", device="cpu") as st:
        weight_map = {key: "model.safetensors" for key in st.keys()}
```

### 2.6 ⭐ `nerkyor/Qwen3.6-27B-NVFP4-v8-RTN` ≠ 27B-A3B stand-in

```
27B-NVFP4-v8-RTN config:
  architectures: ["Qwen3_5ForCausalLM"]    ← Dense, not MoE
  hidden_size: 5120                         ← vs A3B's 2048
  intermediate_size: 17408                  ← Dense FFN
  no num_experts / mlp.experts.* fields

→ This is the NVFP4-quantized Qwen3.6-27B Dense model, NOT the 27B-A3B MoE.
→ Cannot be used as a test stand-in for Lynn-V4-Distill-Qwen-27B-A3B.
```

For pre-arrival harness testing, use the existing 35B-A3B-FP8 checkpoint
(same MoE architecture, just larger) as the baseline candidate.

---

## 3. PyTorch Profiler measurement pitfalls

### 3.1 ⭐ API attribute renames

```python
❌ e.cuda_time_total           # PyTorch 2.x AttributeError
❌ e.self_cuda_time_total      # same

✅ e.device_time_total         # PyTorch 2.x renamed
✅ e.self_device_time_total

# Sort key follows the same renaming:
✅ prof.key_averages().table(sort_by="self_device_time_total", row_limit=40)
```

The `cuda` → `device` rename is the 2.x trend toward generic accelerator
naming.

### 3.2 `with_stack=True` is 2-4× slower itself

```
Normal profile mode:    1767 ms / 10 steps = 176 ms/step → 5.66 t/s (profile overhead)
with_stack=True mode:   4167 ms / 10 steps = 417 ms/step → 2.40 t/s (slower)
No-profile baseline:     430 ms / 10 steps =  43 ms/step → 23 t/s ✓ truth
```

**Don't use profile-mode wall time as t/s baseline**. Profile-mode
self_cuda_time relative ratios are trustworthy, absolute walls are not.

### 3.3 Profile categories are only trustworthy as relative ratios

```
normal mode:     aten::cat = 19.8 ms (15.6% of CUDA total)
with_stack mode: aten::cat = 1.29 ms     (15× different!)
```

**Cause**: profiler CPU overhead affects GPU event measurement; the two
modes cannot be compared in absolute numbers, only in % share.

### 3.4 ⭐ `with_stack=True` does NOT attach stack to GPU events

```python
events = prof.events()
for e in events:
    if e.name == "aten::cat":
        print(e.stack)   # → empty list / None / "<no_stack>"
```

**Cause**: `with_stack=True` only captures Python frames on **CPU dispatch
events**. GPU kernel events (the actual `aten::cat` op on device) carry no
stack.

**Workarounds**:
- A. Use `prof.key_averages(group_by_stack_n=15)` to view grouped CPU
  events (GPU kernels still lack stack)
- B. Parse chrome trace JSON to find CPU "ProfilerStep" parent events
- C. **Manual source grep** (most pragmatic in practice):
  `grep -nE "torch.cat|torch.stack|repeat_interleave" engine/*.py` directly
  locates the call sites

`prof.export_stacks(path, "self_cuda_time_total")` writes 0 bytes silently
in PyTorch 2.8.

### 3.5 Categories cannot sum to 100%

```
my profile script outputs:
  Total CUDA time: 253.66 ms (10 decode steps)
  Actual Self CUDA total = 126.86 ms

→ categorize logic double-counts parent/child events
→ individual category % is trustworthy, sum-to-100% is not
```

**Fix**: use `prof.events()` to count each event uniquely + categorize
carefully, or just read the top-N table directly.

### 3.6 GPU compute time vs wall time

```
profiler returns:    Self CUDA time total
wall time:           measured t/s baseline
diff:                Python orchestration / kernel launch / sync overhead

Example: Self CUDA = 12.7 ms/step  (GPU active time)
         Wall      = 43 ms/step    (real t/s ground truth)
         Python overhead = 30 ms/step (~70% of wall)
```

**Critical**: GPU-side optimization ceiling = wall - Python overhead.
**Without solving Python overhead, GPU-side optimization hits a hard
ceiling**.

---

## 4. BF16 numerical precision pitfalls

### 4.1 ⭐ FP32 accumulator promote is redundant

```python
❌ Assumption: BF16 input means BF16 accumulator, must promote to FP32:
   gate_out = torch.bmm(h.float(), w.float()).to(h.dtype)

✅ cuBLAS handling BF16 input **already uses FP32 accumulators internally**
   (standard Tensor Core behavior).
```

**Pitfall**: bmm and optimized differ numerically; the natural assumption
is "BF16 accumulator insufficient, promote to FP32". Smoke test shows zero
behavioral change after the patch.

**Real cause**: not accumulator precision, but cuBLAS reduce-schedule
differences across kernels.

### 4.2 ⭐ bmm vs F.linear cannot be byte-exact under BF16

```
F.linear  → cuBLAS gemm                tile size A, schedule X
torch.bmm → cuBLAS gemmStridedBatched  tile size B, schedule Y
Both use FP32 accumulators
But reduction order differs (FP addition is non-associative)
→ ε-magnitude differences (BF16 ε ~ 8e-5)
→ invisible in single layer
→ accumulates through 40 residual layers
→ flips router top-K on borderline prompts
```

**Not a bug**. A consequence of cuBLAS algorithm selection. Achieving
bmm = F.linear byte-exactly under BF16 is physically impossible.

**Fix direction**: Don't pursue bmm = F.linear equality. Compare both to
the HF transformers reference instead, or keep bmm as opt-in (not default).

### 4.3 ⭐ Multi-meaning prompts are the canary for ε drift

```
Passing prompts (11/14):
  "The capital of France is" → "Paris" (deterministic continuation)
  "def fibonacci(n):" → code mode, top-K logits well-separated
  "import torch" → standard

Failing prompts (3/14):
  "2+2="           → multiple top-K candidates within ε
  "Python is"      → broadly descriptive
  "I love eating"  → ambiguous emotion/food
```

**Pattern**: ε differences don't affect router selection when top-K logits
are well-spread; they **flip selection at borderline**.

**Canary prompts must include multi-meaning prompts**. Deterministic-only
test sets at 100% pass are self-consistent-bug false positives.

### 4.4 ⭐ Single-prompt PASS never endorses any implementation

**The blood lesson**:
- bmm single-prompt `The capital of France is` PASS 10/10 token
- Default switched to bmm based on this
- Multi-prompt 14 prompt revealed bmm 78.6% match, 3 prompt fail
- → revert default, retract production-ready label

**Universal review rule**: any default-impl change / production-ready
label / merge-to-main decision MUST gate on multi-prompt N≥14 exact-match.
**Single-prompt PASS is never an endorsement**.

### 4.5 SDPA `enable_gqa=True` is byte-exact with `repeat_interleave`

```python
# Before:
K_attn = K.repeat_interleave(H_Q // H_KV, dim=1)  # 8x mem expand
attn_out = F.scaled_dot_product_attention(q, K_attn, V_attn, ...)

# After (byte-exact, idiomatic):
attn_out = F.scaled_dot_product_attention(q, K, V, enable_gqa=True, ...)
```

**Performance delta < 0.5%** (broadcast vs explicit repeat numerically
equivalent in BF16 wall time). **GQA repeat_interleave is NOT the GPU
bottleneck**; misdiagnosing it wastes time.

---

## 5. torch.compile pitfalls

### 5.1 ⭐ `mode="reduce-overhead"` hits mutated_inputs

```python
torch.compile(_decode_layer, mode="reduce-overhead")

→ RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run
```

**Cause**: `reduce-overhead` enables CUDA Graph; **CUDA Graph is
incompatible with in-place state mutation** (KV cache write,
linear_attn conv state replace).

**Fixes** (all require refactoring):
- A. Make state functional (return new state per step, no mutation)
- B. Explicit `torch.compiler.cudagraph_mark_step_begin()` between calls
- C. Use stable buffer for KV cache with internal slice writes

### 5.2 ⭐ `mode="default"` hits specialization recompile

```python
_decode_layer(h, pos_int, layer_type, w, cfg, state, layer_idx)
              ^^^^^^^^^                 ^                 ^^^^^^^^^
              Python int               Python dict        Python int

torch.compile treats these as specialization keys
→ 40 layers × 2 layer_types = 80+ unique cache entries
→ hits cache_size_limit (default 64)
→ falls back to eager (slow)
→ 9× slower than baseline
```

**Fixes** (all require refactoring):
- A. Don't pass `layer_idx` / dict — pass individual tensors instead
- B. Split linear_attn / full_attn into independent compiled functions
- C. Raising `cache_size_limit` to 256 is **200× SLOWER**, not faster

### 5.3 Higher `cache_size_limit` makes things slower

```
cache_size_limit = 64:   1.5 t/s (hits limit, falls back)
cache_size_limit = 256:  0.06 t/s (200× slower!)
```

**Cause**: not hitting the limit, but every step still triggers
inductor compile work; graph regeneration cost exceeds eager.

**Conclusion**: don't try to "fix" specialization issues with
`cache_size_limit`. **Refactor function signatures** to avoid
specialization at the source.

### 5.4 torch.compile preserves numerical correctness

Even when speed regresses, torch.compile **preserves byte-exact
numerical output**:

```
baseline tokens:  [34756, 364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300]
compiled tokens:  [34756, 364, 1141, 25438, 57902, 1680, 430, 279, 242476, 300]
✓ EXACT MATCH (10/10)
```

→ compile doesn't break correctness, only speed (when used incorrectly).

---

## 6. Lynn engine interface / wire-in pitfalls

### 6.1 ⭐ Triton kernels written but not wired into all paths

```
triton_kernels/rope.py::make_triton_rope          ← exists
But only wired in: engine/qwen36_block.py (Phase 2 brute path)
Not wired in:      engine/incremental_decode.py (Phase 3.1 decode path)
```

**Pitfall consequence**: profile shows 5 internal `cat` ops in RoPE
all coming from incremental_decode's pure PyTorch implementation,
not Triton.

**Fix**: before wiring in, grep actual call paths to verify which
kernels are on the hot path.

### 6.2 ⭐ `generate_incremental` reloads weights every call

```python
def generate_incremental(model_dir, prompt, max_new=5, ...):
    layer_weights = []
    for i in range(n_layers):
        w, _ = load_qwen36_layer(model_dir, i, ...)  # 12s each × 40 layers
        layer_weights.append(w)
    # ... prefill + decode
```

**Multi-prompt benchmark waste**: 14 prompts × 3 impls × 12s load =
504s = 8.4 min just on loading.

**Fix**: write `generate_incremental_reuse_weights(layer_weights, outside, ...)`
accepting pre-loaded weights. **But will break LYNN_MOE_IMPL pre-stack
hook idempotency** (cannot stack twice); requires careful state machine.

### 6.3 ⭐ pre-stack hook timing

```
✗ pre-stack BEFORE prefill → prefill uses baseline _moe_forward which
                              still needs original per-expert tensors
                              → KeyError on mlp.experts.{e}.gate_proj.weight

✓ pre-stack AFTER prefill / BEFORE decode → prefill uses original experts,
                                            decode uses stacked layout
                                            → del original to free 60GB
```

The Codex fix for this timing bug is the reference design.

### 6.4 LynnInferenceState is a mutable dataclass

```python
state.update_full_attn_kv(layer_idx, K_new, V_new, position_start)
  → in-place K[..., t:t+1, :] = K_new

state.update_linear_attn_state(layer_idx, new_S, new_C)
  → dict assignment self.recurrent_state[layer_idx] = new_S
```

**Properties**:
- ✅ Memory-efficient (pre-allocated buffer)
- ❌ Incompatible with CUDA Graph / functional refactor
- ❌ Incompatible with `torch.compile(reduce-overhead)` (mutated_inputs)

**Fix**: leave functional rewrite for an isolated branch / unhurried
session; don't force it under time pressure.

---

## 7. Multi-prompt gate engineering discipline

### 7.1 N=14 is the floor, N≥20 is stress test

```
14 prompts is sufficient to expose the self-consistent bug class
(prompts 5/6/12 fail in our case)
N=20 adds 6 more borderline prompts (2+2= type) for stricter stress

After fix, must pass N=20+ in full before considering default switch
```

### 7.2 Watchdog rules against false greens

```
DO    Check multi_prompt_gate.exact_match_rate < 1.0 → FAIL
DO    Check multi_prompt_gate.mismatches_count[*] > 0 → FAIL
DO    Dump GT vs got token sequences for mismatch prompts
DON'T Trust any "verdict": "PASS" field (may be mislabeled or absent)
DON'T Use single-prompt phase32_bench data for default decisions
```

### 7.3 Multi-prompt set must include multi-meaning continuations

```
✅ Required:
  Math       (2+2= / 5+7=)
  Description (Python is / AI 是 / The meaning of life is)
  Emotion    (I love eating / I don't know what to)
  Short      (你好 / Hi)

❌ Insufficient (deterministic-only):
  The capital of France is / def fibonacci(n):
  100% pass on these means nothing — ε differences don't show
```

---

## 8. General engineering hygiene pitfalls

### 8.1 ⭐ Single-GPU container does not run experiments in parallel

```
Wrong:  tmux p1p2 running multi-prompt + tmux profile running stack profile
        → GPU OOM / race condition / data contamination

Right:  Sequential — one task completes, triggers the next (marker grep / wait loop)
        Pipeline:profile → multi-prompt → write daily JSON → ALL_DONE marker
```

### 8.2 ⭐ background bash silent failure

```
❌ ssh '... | python3 -c "..."'   ← exit 127 silent
   tasks/<id>.output ends with "bash: line N: python3: command not found"
   But background bash exit_code = 127 doesn't trigger notification properly

✅ Use /root/miniconda3/bin/python explicit path
   Or export PATH at the start of the SSH command
```

---

## 9. Toolchain quick reference (SSH command templates)

### 9.1 Standard SSH command prefix

```bash
SSHPASS='<pwd>' sshpass -e ssh -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -p <port> \
  root@<host> '
  export PATH=/root/miniconda3/bin:/usr/local/cuda-12.8/bin:$PATH
  cd /root/autodl-tmp/lynn-engine
  # ... actual work
'
```

### 9.2 Long-running task in background

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

### 9.3 Multi-prompt gate template

```bash
# Sequential per impl per prompt (each launches a fresh Python process, ~12s load)
LYNN_MOE_IMPL=$IMPL PYTHONPATH=. timeout 240 \
  /root/miniconda3/bin/python engine/full_forward.py \
    --model /root/autodl-tmp/models/Qwen3.6-35B-A3B-FP8 \
    --prompt "$P" --max-new 8 --mode incremental
```

### 9.4 Writing remote scripts (avoid heredoc / quote traps)

```bash
# A. Use Write tool locally + scp:
SSHPASS='...' sshpass -e scp -P <port> -o ... \
  /tmp/local_script.py root@host:/tmp/

# B. printf line-by-line (fast but error-prone):
echo "#!/bin/bash" > /tmp/foo.sh
echo "cd /tmp" >> /tmp/foo.sh
printf "%s\n" "command with (parens)" >> /tmp/foo.sh   # printf to avoid paren expansion

# DON'T:
# bash -c 'cat > /tmp/foo.sh <<EOF ... EOF'   ← ssh stdin / quoting extremely fragile
```

### 9.5 Verifying patch application

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

# 4. Smoke test (single prompt) numerical
LYNN_MOE_IMPL=optimized PYTHONPATH=. python engine/full_forward.py ...

# 5. Multi-prompt N=14 gate

# 6. On failure → revert: cp engine/foo.py.bak.<patch_name> engine/foo.py
```

---

## 11. Quick reference — known "phantom speedups"

Optimizations that do NOT actually improve t/s (tried / proposed / calculated):

| Optimization | Expected | Actual | Lesson |
|---|---|---|---|
| SDPA `enable_gqa=True` replacing explicit `repeat_interleave` | -28% memcpy | <0.5% | broadcast vs repeat is byte-exact equivalent |
| FP32 accumulator promote (BF16 input) | fix bmm numerics | zero change | cuBLAS already uses FP32 accumulator |
| `torch.compile(mode="default")` simple wrap | reduce Python overhead | 9× slower | layer_idx specialization hits cache limit |
| `torch.compile(mode="reduce-overhead")` simple wrap | CUDA Graph speedup | mutated_inputs runtime error | KV cache mutation incompatible |
| Phase 3.3 Triton MoE FFN kernel | unlock high t/s | matmul is only 9% of GPU time | GEMM is no longer the bottleneck |
| F2.x cat/copy elimination across the board | 25 t/s | max ~14-16 t/s | Python overhead 30 ms is the absolute wall floor |

Real paths to ≥25 t/s (known):
- ✅ Python overhead refactor (`_decode_layer` signature / functional state /
  CUDA Graph compatibility) — 1-2 weeks
- ✅ B1 NVFP4 grouped GEMM (weight 4-bit → memory bandwidth halved) — 4-6 weeks
- ✅ Both together (in a calmer environment than rented machine time)

---

## Revision history

- **2026-05-10 v1**: First version, based on a Lynn engine session's
  accumulated experience. Covers AutoDL env / HF download / PyTorch
  profiler / BF16 numerics / torch.compile / Lynn engine interfaces /
  multi-prompt gate / general engineering hygiene / quick reference.
