# Lynn Engine P73: grouped per-16 non-atomic reference backend

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P73

P72 defined the non-atomic direction: output ownership, no atomics, and a
native-owned scratch boundary. P73 implements the first runtime-visible version:

```bash
export LYNN_NATIVE_ACTIVE_MOE_BACKEND=grouped_per16_nonatomic
```

The implementation still uses tiled scalar gate/up and tiled scalar down
internally. Its purpose is to create the replaceable native-owned active-MoE
boundary, not to claim the final speed path.

## Probe

```bash
python benchmarks/p73_grouped_per16_nonatomic_reference_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p73_grouped_per16_nonatomic_reference_probe.json \
  --layers 2 8 14 20 28 36 \
  --tile-inter 2 \
  --tile-hidden 2 \
  --warmup 4 \
  --iters 30
```

P69 acceptance:

```bash
python benchmarks/p69_grouped_kernel_acceptance_gate.py \
  --report reports/p16_155/p73_grouped_per16_nonatomic_reference_probe.json \
  --out reports/p16_155/p73_p69_acceptance_on_nonatomic_reference.json
```

## Result

| Metric | Value |
|---|---:|
| mean Triton active MoE | 0.05658 ms |
| mean grouped-per16 non-atomic reference | 0.05098 ms |
| candidate vs Triton speedup | 1.110x |
| min cosine vs Triton | 0.99999988 |
| max relative L2 vs Triton | 4.98e-4 |
| P69 speed threshold | 1.25x |
| P69 acceptance | fail |

P73 is essentially tied with P68:

| Candidate | Speedup |
|---|---:|
| P68 active tile reference | 1.108x |
| P73 native-owned non-atomic reference | 1.110x |

## Decision

P73 passes as a scaffold and fails as a promotion candidate.

This is the result we needed to see. Merely moving the P68 two-stage reference
behind a native-owned active-MoE boundary does not unlock 155TPS. The next gain
must come from changing the inner math or launch strategy:

- replace tiled scalar gate/up with grouped FP4/CUTLASS/CuTe math;
- replace tiled scalar down with a persistent/non-atomic grouped schedule;
- or re-layout the artifact for a vendor-compatible FP4 kernel, gated by V8/V9
  retention.

The runtime hook is now ready for those replacements:

```text
grouped_per16_nonatomic -> active_moe_grouped_per16_nonatomic_reference(...)
```

Future candidates should keep this ABI and must pass P69 before any
full-generate/server gate.
