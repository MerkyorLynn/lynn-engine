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
