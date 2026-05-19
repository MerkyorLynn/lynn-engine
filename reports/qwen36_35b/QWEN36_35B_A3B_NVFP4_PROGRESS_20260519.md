# Qwen3.6-35B-A3B NVFP4 Progress Snapshot - 2026-05-19

## Current Position

Default promote candidate remains:

- official `Qwen/Qwen3.6-35B-A3B`
- Lynn-native W4A16 NVFP4
- graph reuse + in-place state serving profile
- W4A8 only as a speed experiment
- MTP only as warm-start / diagnostic until accept-rate and end-to-end TPS pass

The 35B line is no longer blocked on basic quality. It is blocked on strict, resident-safe speedup.

## Quality Results

| Artifact | MMLU 500 5-shot | GPQA Diamond | Status |
|---|---:|---:|---|
| official BF16 | 86.40% | 45.45% | quality reference |
| Lynn-native W4A16 NVFP4 | 84.40% | 49.49% | promote-quality safe |
| Q4_K_M-imatrix GGUF | 83.00% | 50.00% | llama.cpp reference |

Interpretation:

- W4A16 NVFP4 keeps almost all BF16 MMLU and does not lose GPQA.
- Compared with Q4_K_M, NVFP4 is slightly better on MMLU and effectively tied on GPQA.
- This is why W4A16 NVFP4 is the quality-stable native candidate.

## Serving Speed Status On R6000

| Profile | P37 exact | Structured | P25 512 decode TPS | Verdict |
|---|---:|---:|---:|---|
| safe default W4A16 | true | 40/40 | about 107-108 | default candidate |
| shared_gate=triton + conv=triton_inplace | false | 40/40, 70-set stress 69/70 | about 113-114 | AMBER / research only |
| strict_fused_boundary variants | false or slow | closed | about 96-97 | closed |
| full-attn qkv row fusion | false | closed | no real gain | closed |

The default line is stable but not enough for 120+. The faster AMBER line shows there is speed available, but it drifts exact greedy parity and fails the stricter 70-prompt stress gate.

## What Was Built

### 1. Quality pivot to official 35B

We moved away from the custom 27B/V4 distill route for default promotion and made official Qwen3.6-35B-A3B the primary 35B route. This removed the endless Recovery/structured-quality loop: W4A16 NVFP4 quality is good enough to focus on runtime.

### 2. Promotion gates

The gate stack now separates:

- P37 exact greedy parity
- P25 service TPS
- 40 hard structured default gate
- 70 hard structured AMBER stress gate
- P26/P28 phase timing

This prevented unsafe 113-134 TPS candidates from being accidentally promoted.

### 3. Native MoE fixture infrastructure

Built and merged fixture gates around real 35B W4A16 active MoE:

- p133/p134 full/routed fixtures
- p135/p136 slot-order repack fixtures
- p138/p139 packed-slot NVFP4 offline exact gates
- candidate metrics/admission wrappers

This turns Native MoE work from 5-minute full-model experiments into fixture-level iteration.

### 4. Native MoE candidates

Important findings:

- naive warp-per-row native candidate was exact locally but 4x slower than Triton.
- output-owned BF16 candidate reached about 0.052 ms vs Triton active about 0.059 ms, but strict RED because BF16/intermediate rounding differs from PyTorch/Triton reference.
- packed NVFP4 grouped/non-atomic candidates can be fast offline but remain strict RED or graph unsafe.
- one graph-on native path reached about 134 TPS but collapsed generation after the first token, likely from graph-capture allocation/state ABI issues.
- latest graph-safe fixture candidate is about 0.044 ms, but still AMBER with max_abs around 0.00195 and cosine around 0.999988.

No Native MoE candidate is resident-promotable yet.

### 5. Full-attn / linear-GDN probes

Closed several tempting micro-routes:

- manual GQA slower
- qkv row fusion not strict
- conv_inplace alone causes P37 drift
- outconv/gate math can be faster but not exact
- recurrent-from-outconv is exact but service TPS stays about 108

P181 is the best positive evidence:

- all 10 full-attn tail graph captures passed exact
- eager tail about 0.324 ms
- graph replay about 0.109 ms
- local speedup about 2.98x

But this is not wired into resident serving yet. The prepared eager candidate did not improve service TPS.

## Current Bottleneck

The hard blocker is not host overhead. Host gap is already small.

The blocker is strict boundary coarsening:

1. active MoE boundary still fragmented and/or graph unsafe
2. full-attn tail graph replay is exact locally but not integrated into the live decode loop
3. approximate or reordered math quickly gives speed but fails P37 / structured gates
4. MTP official sidecar does not align, so it cannot be counted as a speed lever

## Is There Still A Path To Improve?

Yes, but the route is narrower now:

1. Resident full-attn tail graph integration from P181.
   - This is the cleanest exact graph evidence.
   - Goal: turn local 2.98x tail replay into real P25 gain.

2. Native MoE graph-safe ABI.
   - Caller-owned scratch, no allocation inside CUDA graph capture.
   - Candidate must pass fixture strict or a clearly defined Triton-equivalent contract before P37.

3. Offline MoE repack.
   - Slot/order layout is now available.
   - Need a packed layout that native kernels consume directly without dynamic gather/indexing.

4. W4A8 later.
   - Only after W4A16 boundary work is stable.
   - W4A8 is not a quality/default lever today.

## Practical Decision

35B should remain a high-quality MoE / native-kernel research branch. It is worth continuing, but it should not burn all mainline time trying to brute-force 120 TPS this week.

9B dense should become the immediate speed mainline because:

- no active MoE routing complexity
- Q4_K_M already proves R6000 can run the model much faster
- NVFP4 quality is close enough to justify a native-speed push
- the 155 TPS target is more likely to be reached first on dense 9B

