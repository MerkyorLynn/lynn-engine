# Lynn Engine P72: non-atomic grouped per-16 kernel design

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P72

P71 closes the atomic branch:

```text
fused atomic candidate: 0.17704 ms
current Triton active:  0.05704 ms
speedup:                0.322x
P69 acceptance:         fail
```

The next implementation must be non-atomic. It also must avoid P68's two-stage
tax:

```text
P67 down-only tile:       1.266x vs Triton down
P68 active two-stage:     1.108x vs Triton active
```

P72 defines the first non-atomic grouped kernel shape before filling the CUDA
body.

## Ownership Rule

The fused kernel must use **output ownership**, not atomic accumulation:

```text
one CTA / warp-group owns hidden_out tile H_TILE
for each owned hidden row:
  acc = 0
  for slot in top_k:
    for inter in 0..511:
      acc += route[slot] * down_weight(expert, hidden, inter) * inter_value(slot, inter)
  store out[hidden]
```

No two CTAs write the same `out[hidden]`, so no `atomicAdd`.

## The Gate/Up Recompute Problem

Naively computing `inter_value(slot, inter)` inside every output tile would
recompute gate/up for each hidden tile:

```text
2048 hidden / H_TILE(16) = 128 output tiles
top_k=8
inter=512
```

That would be far worse than P68. Therefore the first practical non-atomic
shape is a **native-owned two-kernel fused schedule**, not one monolithic kernel:

1. `kernel_gateup`: computes `inter[top_k,512]` into native-owned scratch.
2. `kernel_down_nonatomic`: each output tile owns its hidden rows and consumes
   scratch without atomics.

This still has two kernels, but it differs from P68 in the important ways:

- the scratch allocation and layout are owned by the native extension;
- the down side can become a persistent/non-atomic grouped kernel;
- the same ABI can later replace `kernel_gateup` with CUTLASS/CuTe FP4 MMA;
- Python/Triton wrapper overhead is removed from the active-MoE boundary.

## First CUDA Skeleton Target

P72/P73 should add a new native extension entry point:

```text
active_moe_grouped_per16_nonatomic_reference(
  hidden[2048],
  expert_ids[top_k],
  routing_weights[top_k],
  gate_up_packed / gate_up_scale / gate_up_global_scale,
  down_packed / down_scale / down_global_scale
) -> out[2048]
```

Version 0 may internally allocate scratch and call the existing tiled scalar
gate/up + tiled scalar down kernels. That is not a final speed path; it is the
bridge that lets future work replace:

- gate/up inner math;
- down ownership tile;
- scratch layout;
- launch strategy;

without changing Python/runtime ABI.

## Why This Differs From P68

P68 exposes the two-stage reference directly as a measurement path. P72's
nonatomic backend is a future implementation point with a stricter contract:

| Property | P68 tile reference | P72 non-atomic target |
|---|---|---|
| purpose | measurement | implementation scaffold |
| runtime backend | not default | reserved candidate |
| scratch | ordinary tensor return path | native-owned scratch |
| writer ownership | down tile owns hidden | down tile owns hidden |
| atomics | no | no |
| P69 minimum | failed 1.108x | must exceed 1.25x before full gates |

## Acceptance

The first non-atomic implementation is useful only if:

1. it passes P69 numerical thresholds;
2. it improves over P68's 1.108x active boundary;
3. ideally approaches or exceeds the 1.25x P69 speed threshold.

If it does not, the next step should be CUTLASS/CuTe FP4 MMA rather than more
scalar CUDA rearrangement.

## Relationship To GGUF / imatrix

The GGUF/Q4_K_M imatrix route remains a separate public production path. It is
valuable for user-facing llama.cpp speed and quality, but it does not replace
this NVFP4 engine work. P72 is specifically for Lynn-native per-16 NVFP4
serving.
