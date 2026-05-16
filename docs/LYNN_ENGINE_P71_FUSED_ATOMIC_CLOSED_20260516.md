# Lynn Engine P71: fused atomic branch closed under P69

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P71

P46 already showed that the first one-kernel fused atomic active-MoE probe was
too slow. P71 reruns that branch under the current P69 grouped per-16 acceptance
schema so the negative result is expressed in the same terms as future fused
kernel candidates.

This matters because P70 created the clean `grouped_per16_fused` replacement
point. We should not fill it with an atomic kernel just because it is "fused".

## Probe

```bash
python benchmarks/p71_grouped_per16_fused_atomic_acceptance_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p71_grouped_per16_fused_atomic_acceptance_probe.json \
  --layers 2 8 14 20 28 36 \
  --warmup 4 \
  --iters 20
```

P69 acceptance gate:

```bash
python benchmarks/p69_grouped_kernel_acceptance_gate.py \
  --report reports/p16_155/p71_grouped_per16_fused_atomic_acceptance_probe.json \
  --out reports/p16_155/p71_p69_acceptance_on_fused_atomic.json
```

## Result

| Metric | Value |
|---|---:|
| mean Triton active MoE | 0.05704 ms |
| mean fused atomic candidate | 0.17704 ms |
| candidate vs Triton speedup | 0.322x |
| min cosine vs Triton | 0.99999720 |
| max relative L2 vs Triton | 2.69e-3 |
| P69 speed threshold | 1.25x |
| P69 acceptance | fail |

The failure is decisive:

- speed is roughly **3.1x slower** than the current Triton active path;
- cosine misses the strict P69 threshold;
- relative L2 is under the loose sub-kernel threshold, but that is irrelevant
  because both speed and cosine fail.

## Decision

Close the fused atomic branch.

The next `grouped_per16_fused` implementation must be non-atomic. The likely
shape is a grouped/block-diagonal active expert kernel that either:

- keeps each selected expert's down output non-atomic and reduces deterministically;
- or uses a CUTLASS/CuTe-style grouped FP4 MMA schedule that preserves Lynn's
  per-16 scale contract.

P71 prevents a tempting but wrong shortcut: "one kernel" is not enough. The
kernel must also avoid atomics, avoid the P68 intermediate tensor, and pass P69
before any full-generate/server gate.
