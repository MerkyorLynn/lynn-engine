# Lynn Engine P32-P34 · Native active-MoE generate gate negative

Date: 2026-05-16  
Hardware: RTX PRO 6000 Blackwell Server Edition (`sm_120`)  
Model: `lynn-27b-variable-recovery-step5000-nvfp4-final`

## Why this gate exists

P31 proved that `LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar` is exact and faster at the **MoE function boundary**. That is not enough to promote it into decode. P32/P33/P34 test the same backend inside the full autoregressive `runner.generate()` path.

## Results

| Gate | Config | Result | Verdict |
|---|---:|---:|---|
| P32-A | `cuda_scalar` + reusable linear-block graph | 121.74 tok/s mean, but token-0 / `!` loop from decode token 2 | ❌ unsafe |
| P32-B | `cuda_scalar`, graph disabled | coherent text, 28.20 tok/s mean, but greedy ids mismatch | ❌ not parity |
| P33 | first-divergence probe, graph disabled | first top-1 divergence at step 0; first hidden drift below threshold at layer 9 | ❌ drift source confirmed |
| P34 | `cuda_scalar` only on full-attention eager layers | no `!` loop, ~99 tok/s median, but greedy ids mismatch | ❌ not parity |

## Key evidence

### P32-A: graph replay is toxic for full `cuda_scalar`

Default graph-enabled run:

```text
triton mean decode TPS:      99.99
cuda_scalar mean decode TPS: 121.74
pass:                        false
first divergence:            token index 1 for all 3 prompts
output symptom:              "!!!!!!!!!!!!!!!!..."
```

This is a hard failure: it is fast but silently wrong.

### P32-B: disabling graph removes the hard failure but not drift

Graph-disabled run:

```text
triton mean decode TPS:      27.42
cuda_scalar mean decode TPS: 28.20
pass:                        false
output symptom:              coherent text, but greedy ids mismatch
```

So there are two separate issues:

1. graph replay + cuda scalar active MoE can produce token-0 loops;
2. eager cuda scalar active MoE still changes logits enough to flip greedy top-1 on low-margin steps.

### P33: first divergence

```text
initial top1:              3709
first top1 divergence:     step 0
triton top1:               97273
cuda_scalar top1:          96181
triton margin:             0.2109375
cuda_scalar margin:        0.0546875
first hidden divergence:   layer 9
layer 9 cosine:            0.9999941587
layer 9 rel_l2:            0.0034206412
```

The hidden drift looks small layer-by-layer, but logits are close enough that greedy decoding flips immediately.

### P34: full-attention-only allowlist is still not enough

`LYNN_NATIVE_ACTIVE_MOE_LAYERS=full_attention` avoids capturing cuda scalar kernels inside the reusable linear-attention block graphs. It avoids the `!` loop, but still fails greedy-id parity.

```text
triton mean decode TPS:      100.18
cuda_scalar median TPS:      99.48
pass:                        false
native layers:               full_attention only
```

## Code guard

The runner now fail-louds when this unsafe combination is requested:

```text
LYNN_NATIVE_ACTIVE_MOE_BACKEND=cuda_scalar
LYNN_LINEAR_BLOCK_GRAPH=1
LYNN_NATIVE_ACTIVE_MOE_LAYERS unset or includes linear_attention
```

To run explicit diagnostics anyway, set:

```bash
export LYNN_ALLOW_UNSAFE_CUDA_SCALAR_GRAPH=1
```

Do not use that in serving.

## Decision

`cuda_scalar` remains an opt-in diagnostic backend. It is **not promoted** to default and should not be used for production generate.

The next safe optimization track is:

1. Python/orchestration reduction and graph-state refresh slots, which should not alter arithmetic;
2. a real grouped native-FP4 active expert kernel with a stricter full-generate parity gate;
3. only after parity, re-enter serving TPS gates.

