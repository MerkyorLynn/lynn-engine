# Lynn 27B W4A8 Device Feasibility

Date: 2026-05-16

## Ground Truth Inputs

Source checkpoint:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-bf16-final
```

Measured artifact sizes:

```text
BF16 final:         59.49 GiB
Lynn-native NVFP4: 19.93 GiB
```

Text model shape:

```text
layers:                 40
full-attention layers:  10   (one every 4 layers)
linear-attention layers:30
hidden_size:            2048
attention heads:        16
KV heads:               2
head_dim:               256
top_k experts/token:    8
moe_intermediate_size:  512
remaining experts/layer avg: 230.75 / 256
```

The config contains:

```text
mtp_num_hidden_layers: 1
mtp_use_dedicated_embeddings: false
```

but the current BF16 artifact has no MTP/NEXTN/draft-head tensors. This means
the architecture has a slot for MTP, but the shipped artifact still needs a new
head/training pass.

## KV And Recurrent-State Budget

Only 10 layers use full attention. BF16 KV cache per token is:

```text
10 full-attn layers * K/V * 2 KV heads * 256 dim * 2 bytes
= 20 KiB/token
```

Approximate single-session KV cache:

| Context | KV Cache |
|---:|---:|
| 4K | 80 MiB |
| 8K | 160 MiB |
| 16K | 320 MiB |
| 32K | 640 MiB |
| 64K | 1.25 GiB |
| 128K | 2.50 GiB |
| 262K | 5.00 GiB |

The 30 linear-attention layers use recurrent state instead of full KV growth.
Even if kept in FP32, the recurrent state is tens of MiB per session, not a
multi-GiB blocker. The main memory problem is therefore **resident weights and
runtime shadow copies**, not KV cache.

## W4A8 Artifact Families

### Lynn-native W4A8

Expected memory shape:

```text
packed E2M1 weights + per-16 scales: ~20 GiB
FP8 activation contract metadata:    small
MTP one-layer head:                  ~0.3-0.8 GiB packed estimate
```

This is the canonical Lynn engine path. It keeps:

- physical variable experts;
- per-16 scale contract;
- custom R6000/5090 kernels;
- Spark FP8 mirror compatibility.

### Vendor-friendly W4A8/NVFP4-v2

Expected memory shape:

```text
base packed weight: ~20 GiB
padding/mask overhead if forced to fixed 256 experts: likely +5-12%
framework metadata/workspace: framework-dependent
```

This path is valuable for ModelOpt/SGLang/vLLM ecosystem work, but it should be
a second artifact. It may give up part of the physical variable-expert memory
advantage unless the framework supports variable expert counts directly.

## Device Matrix

| Device | Memory | W4A8 Feasibility | Expected Role | Main Pitfall |
|---|---:|---|---|---|
| RTX 5090 laptop | 24 GB | **Possible but razor-thin** | portable local brain, short/medium context | needs true packed-only runtime; no BF16 shadow, no padded vendor artifact |
| RTX 5090 desktop | 32 GB | **Good** | consumer production target | still needs packed-only runtime and careful allocator control |
| RTX PRO 6000 / R6000 | 96 GB | **Excellent** | primary kernel dev + production validation | none on memory; quality/runtime gates are the blocker |
| DGX Spark / GB10 | 128 GB unified | **Excellent memory, different ISA** | long-term host, multi-service, Spark FP8 mirror | no FP4/FP6 MMA on sm_121; cannot use R6000 FP4 kernels directly |

## Per-Device Notes

### 5090 Laptop 24GB

This is feasible only in the strict Lynn-native packed path:

```text
weights:       ~20.0 GiB
KV @ 16K:      ~0.3 GiB
linear state:  <0.1 GiB
MTP head:      ~0.3-0.8 GiB
workspace/JIT: ~1.0-2.0 GiB target
```

Budget lands around:

```text
21.7-23.2 GiB before fragmentation
```

Practical constraints:

- must release BF16 decode shadows;
- should not load vendor-padded fixed-256 artifact;
- should avoid large CUDA graph captures at first;
- should cap context to 16K initially, then test 32K;
- no co-resident ASR/TTS/vision services;
- allocator fragmentation can be the real failure, not nominal model size.

Conclusion: **possible, but it needs a laptop-specific low-memory profile**.
If the engine still shows >24GB resident on R6000 after shadow release, laptop
5090 is not ready.

### 5090 Desktop 32GB

This is the best consumer target.

Estimated production budget:

```text
weights + MTP:     ~20.5-21.0 GiB
KV @ 32K:          ~0.6 GiB
runtime workspace: ~2-4 GiB
headroom:          ~6-8 GiB
```

Conclusion: **very plausible**. It should run Lynn-native W4A8 comfortably if
R6000 kernels port cleanly to the consumer Blackwell ISA/driver stack.

Required probe before claiming support:

```text
P103-style FP8 x E2M1 MMA compile/run probe
P105-style generation gate
resident memory report after shadow release
```

### R6000 96GB

R6000 is the reference target:

- enough memory for BF16 reference, W4A8 artifact, and debug copies;
- SM120a supports FP8 activation x E2M1 weight MMA;
- enough headroom for MTP, long context, and instrumentation.

Conclusion: **primary development and quality-gate machine**.

The blocker is no longer memory. The blockers are:

- W4A8 Recovery quality gate;
- runtime promotion parity;
- MTP/NEXTN training quality;
- active-MoE kernel integration.

### Spark 128GB Unified Memory

Spark has plenty of memory and is the best long-lived host, but its GPU ISA is
different:

```text
sm_121: FP8 MMA yes
sm_121: FP4/FP6 MMA no
```

W4A8 helps Spark because E2M1 weights can be losslessly expanded to FP8 E4M3
and consumed by Spark's FP8 MMA path. It does **not** make Spark run the same
FP4 MMA kernels as R6000.

Important correction from Spark SP-13:

```text
existing SP-12 spark_fp8 path:
  activation: E2M1
  weight:     E2M1 expanded through FP8 machinery
  meaning:    W4A4 mirror, not true W4A8

true Spark W4A8 mirror needed:
  activation: FP8 E4M3
  weight:     E2M1 losslessly LUT-expanded to FP8 E4M3
  MMA:        FP8 x FP8 on sm_121
```

So Spark is artifact-compatible with W4A8, but the current SP-12 kernel should
not be described as the W4A8 production mirror. It is a useful W4A4/W4A4-like
probe and a kernel scaffold. A true W4A8 Spark kernel remains a separate
research task.

Expected role:

- long-context and multi-service host;
- Spark FP8 mirror validation;
- cross-framework oracle if a vendor-friendly artifact exists.

Expected performance if a true W4A8 FP8 x FP8 mirror is implemented:

```text
current Spark SP-08:        ~49 TPS
W4A8 Spark FP8 mirror est:  ~60-90 TPS
R6000 W4A8 native est:      155-200 TPS target
```

Conclusion: **excellent product host, not the fastest single-stream kernel
target**.

## Compatibility Conclusions

1. **R6000 and 5090 are the same strategic runtime family** if 5090 exposes the
   same FP8 x E2M1 MMA support. R6000 work should transfer well to desktop 5090,
   and probably to laptop 5090 after a low-memory profile.
2. **Spark is compatible at the artifact level but not kernel-identical**.
   It needs a true FP8-activation x FP8-expanded-weight mirror path, not the
   current SP-12 E2M1-activation probe and not the R6000 native FP4 path.
3. **Vendor-friendly artifacts are additive**. They are useful for ecosystem
   compatibility, but the first W4A8 win should target Lynn-native packed
   runtime because it preserves variable-expert memory savings.
4. **KV cache is not the reason 24GB fails or succeeds**. The model's hybrid
   attention makes KV cheap. The real 24GB risk is BF16 shadows, padding, CUDA
   workspace, and allocator fragmentation.

## Promotion Gates

Before saying "W4A8 supports 5090/R6000/Spark":

```text
quality:
  P105 generation gate after Recovery
  6-prompt smoke
  strict tool-call
  no-think loop guard
  V8/V9 retention

memory:
  resident packed-only report
  no BF16 shadow after warmup
  16K and 32K KV growth measurement

hardware:
  R6000 P103 FP8 x E2M1 probe
  5090 P103-equivalent probe
  Spark SP-11/SP-12 FP8 mirror probe

runtime:
  OpenAI server decode TPS
  long-context TPS
  1-hour stability soak
```

## Bottom Line

W4A8 is the strongest near-term path because it is both quality-friendlier than
W4A4 and hardware-meaningful on Blackwell.

The most likely deployment tiers are:

```text
R6000:         first-class, fastest, easiest validation
5090 desktop: first consumer target, likely strong
5090 laptop:  possible, but needs low-memory packed-only mode
Spark:        long-term host with FP8 mirror, slower than R6000 but very useful
```
