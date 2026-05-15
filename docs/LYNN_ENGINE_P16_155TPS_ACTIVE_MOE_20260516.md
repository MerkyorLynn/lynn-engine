# Lynn Engine P16 — 155 TPS Active-MoE Boundary (2026-05-16)

P16 investigated why the R6000 path stalls around the reproducible
`103-107 tok/s` ceiling after P15 fixed the runtime config regression.

The short version:

> 155 TPS is physically possible on the non-MoE path, but not with the current
> active routed expert kernels. The next real speedup requires a new grouped
> native-FP4 active expert kernel, not more environment toggles.

## Baseline

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Correct P15 env:

```bash
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
export LYNN_MOE_IMPL=packed_nvfp4
export LYNN_PACKED_SHARED_EXPERT=0
export LYNN_QK_NORM_ROPE_BACKEND=triton_pair
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
export LYNN_NATIVE_FP4_LM_HEAD=1
export LYNN_LINEAR_STATE_UPDATE=inplace
export LYNN_PACKED_DECODE=0
export LYNN_PACKED_DECODE_PREPARE_NATIVE=0
```

R6000 group-size 20 graph gate:

| Path | Latency | TPS |
|---|---:|---:|
| groups only | 9.353 ms | 106.91 |
| strict full graph path | 9.686 ms | 103.24 |
| replay-only ceiling | 9.334 ms | 107.13 |

This reproduces the P15 103/107 TPS ceiling.

## What Actually Blocks 155 TPS

P16 added profiling-only env flags:

```bash
LYNN_MOE_PROFILE_SKIP_ACTIVE=1
LYNN_MOE_PROFILE_SKIP_SHARED=1
LYNN_MOE_PROFILE_TOPK_LIMIT=N
```

These are deliberately named `PROFILE`; they are not production switches.

| Experiment | Replay-only TPS | Strict full TPS | Meaning |
|---|---:|---:|---|
| baseline | 107.13 | 103.40 | current correct P15 path |
| skip shared expert | 119.60 | 114.98 | shared expert is worth ~1 ms/token |
| skip active routed experts | 173.84 | 164.24 | active experts are the main bottleneck |
| skip active + shared | 208.78 | 194.97 | non-MoE path has enough headroom |
| top-k limit 4 | 116.08 | 111.78 | reducing routed experts does not scale linearly |
| top-k limit 1 | 124.67 | 119.60 | still far from 155, quality would be unsafe |
| top-k 1 + skip shared | 141.91 | 135.46 | even extreme approximation misses 155 |

The key result is `skip active routed experts -> 164 TPS`. The hardware and
the rest of the runtime can clear 155. The active expert implementation cannot.

## Active Expert Split

Layer 28 P10-H split:

| Segment | Latency |
|---|---:|
| packed gate/up | 0.0429 ms |
| packed down only | 0.0351 ms |
| packed active total | 0.0778 ms |
| BF16 active experts | 0.1502 ms |

Current active expert kernels are already ~1.9x faster than BF16. The problem
is that 40 layers multiply `~0.078 ms` into roughly `3.1 ms/token`.

To hit 155 TPS from a 9.33 ms replay baseline, token latency must drop to about
6.45 ms. That requires removing roughly 2.9 ms. In other words, most of the
active expert cost must go away.

## Dead Ends Checked

### Block-size sweep

The existing default active kernel shape remains near the local optimum.
Aggressive shapes such as `gate_inter=64` or `down_hidden=64` regress badly.

Representative failures:

| Config | Active total |
|---|---:|
| default `gate_i=8, gate_h=64, down_h=8, down_i=256` | 0.0778 ms |
| `gate_i=64, gate_h=64, down_h=16, down_i=512` | 0.306 ms |
| `gate_i=64, gate_h=128, down_h=32, down_i=512` | 0.422 ms |
| `gate_i=32, gate_h=256, down_h=64, down_i=512` | 1.468 ms |

### Top-k approximation

Top-k limiting does not provide enough speed and would be a quality-risking
approximation anyway. Even `topk=1 + skip_shared` only reaches 141.9 replay TPS.

### Naive native `_scaled_mm` gate/up

P10-G selected-expert gate/up on native FP4 tensor cores:

| Path | Latency | Quality |
|---|---:|---|
| native hot gate/up | 0.0779 ms | cosine 0.977 vs BF16 |
| native cold gate/up | 0.1485 ms | cosine 0.977 vs BF16 |
| activation quant only | 0.0214 ms | — |
| gather + scale_b only | 0.0877 ms | — |

The hot path is only a small speedup and fails the strict quality gate. The cold
path is slower because dynamic expert gather + scale construction dominate.

### Native down overcompute

P16 also tested a tempting shortcut for the down projection: stack the selected
expert down matrices and run one `_scaled_mm`, then keep only the diagonal
expert blocks.

| Path | Latency | Quality |
|---|---:|---|
| scalar grouped down | 0.0545 ms | reference |
| native overcompute down | 0.3328 ms | cosine 0.985 vs scalar |

This computes an 8x overcomplete `[top_k, top_k * hidden]` result. It is both
slower and less accurate. The conclusion is narrow but important: a useful
native-FP4 down path must be a **true grouped/diagonal kernel**, not a dense
overcompute wrapper around `_scaled_mm`.

## P16 Decision

The next production-relevant path is **not** another runtime flag. It is a new
active expert implementation:

1. Precompute or cache selected expert native layouts where possible.
2. Avoid per-token dynamic `scale_b` construction on the critical path.
3. Use a grouped native-FP4 active expert kernel that handles the top-k selected
   experts without computing cross-expert products.
4. Preserve the P10-H scalar packed kernel as the correctness reference.
5. Gate promotion with:
   - layer-level cosine / rel_l2 vs current packed active path,
   - full-layer router top-k parity,
   - 6-prompt greedy parity,
   - P9/P10 group graph TPS.

## Practical Target

The reachable milestones are:

| Milestone | Meaning |
|---|---|
| 107 TPS | current reproducible graph ceiling |
| 120 TPS | shared/active minor improvements or limited native pieces |
| 135-145 TPS | aggressive approximate path, not production-safe |
| **155 TPS** | requires new grouped native-FP4 active expert path |
| 200+ TPS | requires active expert plus broader fused graph/kernel redesign |

P16 therefore converts the 155 target from "try harder" into a precise kernel
task: **make active routed expert compute close to the skip-active budget without
dropping the experts.**
