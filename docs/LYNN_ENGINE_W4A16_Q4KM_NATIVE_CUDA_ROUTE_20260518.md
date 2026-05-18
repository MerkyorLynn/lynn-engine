# Lynn Engine W4A16/Q4_K_M Native CUDA Route

Date: 2026-05-18

## Decision

Q4_K_M's quality behavior is the right mental model for the official
Qwen3.6-35B-A3B serving route. It is a weight-only 4-bit path:

```text
W4 weights + BF16/FP16 activations = W4A16
```

This explains why Q4_K_M can stay close to BF16 quality while W4A8 can drift on
structured/code/tool-call prompts. W4A8 changes the activation contract; W4A16
does not.

## CUDA Implication

W4A16 can still be CUDA-native. It just does not rely on an FP8-activation
contract.

The clean kernel shape is:

```text
packed 4-bit weight load
  -> per-group scale dequant into registers/shared memory
  -> BF16/FP16 GEMV or GEMM accumulation
  -> fused activation / routing combine where possible
```

That is compatible with the quality-safe W4A16 artifact. It is also the
license-clean route for Lynn: Atlas may be used only as a high-level reference
for kernel island shape, not copied into this repository.

## Hardware Boundary

R6000 testing already separates the two hardware routes:

| Route | Result | Meaning |
|---|---|---|
| BF16/FP16 activation x E2M1 FP4 weight MMA | Not exposed by current CuTe/CUDA stack | No direct W4A16 tensor-core shortcut |
| FP8 activation x E2M1 FP4 weight MMA | Exposed | W4A8 can use tensor cores, but quality must be earned |
| W4A16 dequant + BF16/FP16 GEMV/GEMM | Software/native CUDA route | Quality-stable default speed work |

So the next speed work should not wait for W4A8 quality. The default path is to
write W4A16 kernel islands:

1. active routed MoE gate/up + SiLU + down;
2. shared expert fused BF16/W4A16 path;
3. linear-attention recurrent/GDN block fusion;
4. full-attention decode layer fusion.

## 2026-05-18 Fast-Decode Result

The first safe W4A16 speed win is `LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode`.
It keeps BF16 activations and changes only the E2M1 decode expression inside the
existing Triton gate/up kernel.

| Probe | Result |
|---|---:|
| P53 gate/up micro, official 35B W4A16 | 1.069x mean speedup |
| P37 generate parity | 3/3 exact |
| P37 median TPS | 100.43 -> 102.57 |
| P25 512-token service wall / decode TPS | 76.58 / 86.60 |
| Structured OpenAI gate | GREEN, 14/14 format-clean, mean 87.71 decode TPS |

This is not the 155 TPS breakthrough by itself, but it is the first validated
official-35B W4A16 serving gain after the route pivot.

The next safe W4A16 gain is `LYNN_QK_NORM_ROPE_BACKEND=triton_pair`, which
fuses the Q/K norm plus RoPE pair in full-attention layers while keeping the same
BF16 activation contract.

| Probe | Result |
|---|---:|
| P27 layer 3 `attn.qk_norm_rope` | 0.361 ms -> 0.129 ms |
| P27 layer 31 `attn.qk_norm_rope` | 0.353 ms -> 0.136 ms |
| P26 full-attention layers | 4.08 -> 3.11 ms/token |
| P26 decode TPS | 85.48 -> 93.07 |
| P25 512-token service wall / decode TPS | 83.38 / 95.88 |
| Structured OpenAI gate | GREEN, 14/14 format-clean, mean 96.43 decode TPS |

This is now part of the default W4A16 fast profile. It moves the official 35B
service path close to 96 decode TPS without touching W4A8 activation quantization
or MTP credit.

`LYNN_RMSNORM_GATED_BACKEND=triton` is the next default W4A16 win. It fuses the
gated RMSNorm used inside linear-attention blocks while preserving BF16
activations.

| Probe | Result |
|---|---:|
| P26 linear graph blocks | 7.10 -> 6.45 ms/token |
| P26 decode TPS | 93.07 -> 99.18 |
| P25 512-token service wall / decode TPS | 85.23 / 102.22 |
| Structured OpenAI gate | GREEN, 14/14 format-clean, mean 102.61 decode TPS |

This is the first official 35B W4A16 service profile above 100 decode TPS.

## Closed Micro-Paths

- Triton router top-k softmax is not useful as a standalone full router path:
  it is faster only when logits are already cached, while full router regresses.
- A `tl.dot` gate/up rewrite is slower on official 35B layer 28.
- More W4A16 tile sweeps are low ROI unless they change the actual grouped
  MoE memory/layout contract.

## Next Kernel Island

The clean-room Atlas lesson is the shape, not the source code: single-call
grouped MoE kernels reduce launch overhead and keep packed weights resident in
the right layout. Lynn should implement that in its existing `csrc/lynn_native`
extension against the current manifest layout and promotion gates.

The expected first target is a Lynn-owned W4A16 grouped active/shared MoE kernel
that preserves BF16 activation semantics. W4A8 remains a separate
activation-aware artifact line for later tensor-core acceleration.

R6000 native-kernel readiness checks now pass for this route:

| Probe | Result |
|---|---:|
| P76 CUTLASS/CuTe SM120 smoke | GREEN, compile ok |
| P70 grouped-per16 fused ABI | GREEN, fail-loud replacement point reserved |
| P39 gate/up fastdecode | 0.031 ms/layer sampled mean |
| P39 down | 0.026 ms/layer sampled mean |
| P39 shared BF16 | 0.060 ms/layer sampled mean |
| P38 current full MoE | 0.213 ms/layer sampled mean |

This means the native CUDA work should target boundary fusion first: remove
separate routed-expert launches and intermediate tensors before spending time on
another scalar tile sweep.

Do not promote the old scalar/native tile references directly. They have real
speed signal but fail generation:

| Candidate | Median Decode TPS | Speedup | Gate |
|---|---:|---:|---|
| `LYNN_NATIVE_DOWN_BACKEND=cuda_tile` | 108.58 | 1.07x | RED, 0/3 greedy IDs match |
| `LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic` | 124.30 | 1.23x | RED, 0/3 greedy IDs match |

Both emitted repeated exclamation marks in P37. The useful lesson is the target
shape and speed budget, not the numeric path itself.

## R6000 llama.cpp Q4_K_M Baseline

The clean R6000 CUDA reference is now:

| Probe | Result |
|---|---:|
| Single 128 tokens | 154.39 wall TPS |
| Single 256 tokens | 202.47 wall TPS |
| Single 512 tokens | 207.29 wall TPS |
| Concurrent 2 total | 306.34 wall TPS |
| Concurrent 4 total | 400.39 wall TPS |
| Concurrent 8 total | 500.66 wall TPS |
| 5.8k prompt tokens + 128 decode | 91.13 wall TPS including prefill |
| 11.6k prompt tokens + 128 decode | 84.19 wall TPS including prefill |

This is the strongest evidence so far that W4A16-class quality does not cap the
serving route near 110 TPS. Lynn's current gap is implementation structure:
layout, launch count, and fused decode boundaries.

## How Lynn Catches llama.cpp

The plan is not to replace Lynn with llama.cpp. It is to copy the engineering
shape in a license-clean way while keeping Lynn-native NVFP4 artifacts and
promotion gates.

### 1. Offline Repack First

llama.cpp wins partly because GGUF is already a serving layout, not just a
checkpoint layout. Lynn's safetensors + manifest format is flexible, but the
decode kernel still pays for generic strides and intermediate layouts.

Add a second, optional serving layout:

```text
manifest checkpoint layout
  -> offline repack
  -> decode_tile layout for active experts / shared expert / linear projections
```

Acceptance:

- no quality change;
- loader can fall back to the current manifest layout;
- P25 gains are visible before any math rewrite;
- artifacts remain compatible with `dl.merkyorlynn.com` distribution.

### 2. Native Active-MoE Boundary

Use llama.cpp's `topk-moe.cu` as a high-level MIT reference for the graph shape:
fuse softmax/top-k/get_rows-style routing decisions and avoid scattering work
back through many small operations. Do not reuse the source.

For Lynn, the first safe kernel should still preserve:

```text
router logits
  -> top-k ids and routing weights
  -> gate/up
  -> BF16 inter store
  -> down
  -> route weighted sum
  -> shared expert add
```

Only after exact parity passes should the BF16 inter boundary be challenged.
The P125 update shows current native candidates still drift too early, so the
next change must tighten numeric order before chasing larger tiles.

Target:

- Stream A first useful step: P37 `3/3` exact and P25 512 decode >=115 TPS.
- Stretch: >=120 TPS with hard structured gate clean.

### 3. GDN/SSM Decode Boundary

llama.cpp now has dedicated `gated_delta_net.cu` and `ssm-scan.cu` kernels.
Those are directly relevant to Qwen3.6 hybrid SSM. Lynn already has Triton
pieces for conv, recurrent update, and gated RMSNorm, but they are still
multiple decode boundaries per linear-attention layer.

For Lynn, Stream B should converge the current sequence:

```text
in_proj -> conv/silu -> split -> beta/g -> recurrent -> gated RMSNorm -> out_proj
```

into fewer explicit CUDA/Triton boundaries while preserving the current
GQA-recurrent and BF16 rounding contract.

Target:

- first useful step: P37 `3/3` exact and P25 512 decode >=113 TPS;
- stretch: >=118 TPS before revisiting service-loop C++.

### 4. CUDA Graph and Scheduler Discipline

llama.cpp reports `USE_GRAPHS=1` in the clean CUDA run. Lynn already benefits
from graph reuse in linear blocks, but full-attention graph slots are not yet
cross-prompt safe. Keep graph work at explicit subgraph boundaries instead of
capturing whole decode per token.

Target:

- only promote reusable graph segments that pass fresh-prompt P37 and hard
  structured gates;
- never repeat the strict-slot whole-token graph path, which was already
  measured as slower due to per-token capture.

### 5. Defer W4A8 and MTP Credit

llama.cpp's Q4_K_M win is W4A16-class. It does not justify sacrificing
structured/tool-call quality for W4A8, and it does not prove MTP works on
Qwen3.6 hybrid SSM. Keep W4A8 and MTP as gated acceleration branches after the
W4A16 kernel route is closer to llama.cpp.
