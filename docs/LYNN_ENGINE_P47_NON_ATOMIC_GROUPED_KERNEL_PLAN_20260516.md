# Lynn Engine P47 · Non-Atomic Grouped Active-MoE Kernel Plan

Date: 2026-05-16

## Goal

Move beyond P46's fused atomic negative result and build the first useful
non-atomic grouped/block-diagonal active expert kernel for Lynn 27B NVFP4.

Target path:

```text
hidden[2048]
  + top_k=8 expert ids
  + routing weights
  + packed E2M1 gate/up/down weights
  + per-16 scales
  -> moe_out[2048]
```

## Why P47 Exists

P43-P46 closed the obvious shortcuts:

- shared expert optimization is too small;
- merged-top-k scheduling loses parallelism;
- cross-expert `torch._scaled_mm` over-computes and drifts;
- fused scalar atomics are 3x slower than Triton and drift slightly.

The next kernel must avoid atomics and avoid cross-expert over-compute.

## Candidate Kernel Shape

### Option A: output-hidden tile owns reduction

One program owns a tile of output hidden rows, for example `[16 or 32 hidden]`.
It reduces over `top_k * intermediate` locally:

```text
for hidden_tile:
  acc[hidden_tile] = 0
  for slot in top_k:
    compute/reuse gate_up_inter[slot, i]
    for i in intermediate:
      acc += route[slot] * down(hidden_tile, i) * gate_up_inter
  store hidden_tile
```

Pros:

- no atomics;
- output ownership is clean;
- down projection becomes the primary locality axis.

Cons:

- gate/up intermediate may be recomputed per hidden tile unless cached;
- recompute factor can erase the benefit.

### Option B: two-stage non-atomic with fused scratch layout

Keep two kernels but make the scratch layout and launch contract native-owned:

```text
kernel 1: top_k gate/up -> compact inter scratch [8,512]
kernel 2: down weighted-sum -> output [2048]
```

This resembles Triton today, but removes Python/Triton wrapper overhead and
lets the CUDA implementation evolve toward tensor-core fragments.

Pros:

- low numerical risk;
- natural stepping stone from P45 ABI;
- can benchmark against P45 scalar and Triton directly.

Cons:

- still two kernels;
- may not be enough for 155 without tensor-core math.

### Option C: CuTe/CUTLASS grouped block-diagonal GEMM

Express the active experts as 8 small block-diagonal GEMMs:

```text
[1,2048] x [2048,1024] per active expert -> [1,1024]
[1,512]  x [512,2048]  per active expert -> [1,2048]
```

Pros:

- closest to real Blackwell FP4 tensor-core utilization;
- long-term route to 155/200+ TPS.

Cons:

- requires deeper CUDA/CuTe work;
- per-16 scale contract must be handled carefully.

## Recommended Order

1. **P47-A**: implement a native two-stage contract that owns allocation and
   launch, with exact parity against Triton. This is the safe bridge.
2. **P47-B**: replace gate/up scalar inner loop with tiled FP4 fragments or a
   better non-atomic CUDA schedule; beat P45 scalar contract first.
3. **P47-C**: replace down scalar inner loop; beat Triton active path on at
   least one representative layer.
4. **P47-D**: multi-layer gate, then full-generate gate, then server TPS gate.

## Promotion Gates

No P47 variant can become default unless:

- layer-level cosine >= 0.999999 against current Triton active path;
- max_abs stays at or below existing P45/P46 diagnostic range;
- 6-prompt greedy full-generate ids match default path;
- tool-call and no-think loop guards still pass;
- server TPS improves, not just isolated layer timing.

## Current Baseline To Beat

From P45/P46:

```text
Triton active mean:             0.0583-0.0592 ms/layer
CUDA scalar contract mean:      0.0658 ms/layer
Fused atomic scalar mean:       0.1768 ms/layer
```

First meaningful win:

```text
P47 native active <= 0.058 ms/layer with parity
```

Breakthrough target:

```text
P47/P48 active path <= 0.035 ms/layer
```

That is the scale of win needed to make 155 TPS plausible from the current
stable/default 100 TPS class runtime.
