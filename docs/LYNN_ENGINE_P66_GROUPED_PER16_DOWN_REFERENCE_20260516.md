# Lynn Engine P66: grouped per-16 down reference probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P66

P65 froze the full grouped per-16 active-MoE ABI. P66 isolates the down
projection half as the first replaceable sub-kernel:

```text
inter[top_k, 512] + expert_ids + routing_weights + Lynn per-16 down weight
  -> hidden[2048]
```

The native implementation is still scalar/reference today. Its job is to prove
that the ABI, real routed inputs, and Lynn-native per-16 down layout all line up
before the inner loop is replaced with CUTLASS/custom CUDA math.

## Probe

```bash
python benchmarks/p66_grouped_per16_down_reference_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p66_grouped_per16_down_reference_probe.json \
  --layers 2 8 14 20 28 36 \
  --warmup 4 \
  --iters 30
```

Report:

```text
reports/p16_155/p66_grouped_per16_down_reference_probe.json
```

## Result

| Metric | Value |
|---|---:|
| sampled layers | 2, 8, 14, 20, 28, 36 |
| contract pass | true |
| mean Triton down | 0.02643 ms |
| mean native reference down | 0.03089 ms |
| native/reference speed vs Triton | 0.856x |
| min cosine vs Triton | 0.99999988 |
| max relative L2 vs Triton | 2.68e-5 |

Layer-level note: four sampled layers were bit-identical against Triton down;
layers 14 and 36 had tiny BF16-level differences but stayed well inside the
contract threshold.

## Decision

P66 passes as a **reference ABI**, not a speed path:

- It proves the down half of grouped per-16 active-MoE can consume real routed
  decode inputs and Lynn-native per-16 packed weights.
- It is slower than Triton because the inner loop is still scalar/reduction.
- It should not be promoted.

The next engineering step is to replace `down_grouped_per16_reference`'s scalar
inner loop with a true grouped per-16 CUDA/CUTLASS down kernel while keeping the
same Python/runtime ABI and the same P66 parity report green.
