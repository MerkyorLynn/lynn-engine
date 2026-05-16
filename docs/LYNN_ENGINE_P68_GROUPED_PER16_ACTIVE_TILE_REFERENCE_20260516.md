# Lynn Engine P68: grouped per-16 active tile-reference probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P68

P66 and P67 proved the grouped per-16 down projection ABI and a fast tile-hidden
down sub-kernel. P68 composes the two tiled scalar CUDA halves into a complete
active expert FFN candidate:

```text
hidden[2048]
  -> gate/up tile-inter reference
  -> SiLU(gate) * up inter[top_k, 512]
  -> down tile-hidden reference
  -> hidden[2048]
```

This is still scalar CUDA reference math. It is not the final tensor-core
grouped FP4 kernel. The purpose is to measure the combined active-MoE signal on
real routed inputs under one grouped-per16 API.

## Probe

```bash
python benchmarks/p68_grouped_per16_active_tile_reference_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p68_grouped_per16_active_tile_reference_probe.json \
  --layers 2 8 14 20 28 36 \
  --tile-inter 2 \
  --tile-hidden 2 \
  --warmup 4 \
  --iters 30
```

Report:

```text
reports/p16_155/p68_grouped_per16_active_tile_reference_probe.json
```

## Result

| Metric | Value |
|---|---:|
| sampled layers | 2, 8, 14, 20, 28, 36 |
| tile inter / tile hidden | 2 / 2 |
| sub-kernel contract pass | true |
| mean Triton active MoE | 0.05654 ms |
| mean native tiled active MoE | 0.05105 ms |
| native tiled vs Triton speedup | 1.108x |
| min cosine vs Triton | 0.99999988 |
| max relative L2 vs Triton | 4.98e-4 |
| runtime promote | false |

Layer-level speedups were consistent across the sampled stack: 1.097x to
1.124x. The numerical drift stayed inside the sub-kernel contract, but it is
larger than the down-only P67 drift because P68 also changes the gate/up
accumulation order.

## Decision

P68 passes as a **complete active-MoE reference candidate**, not as a production
runtime default.

The combined 1.108x speedup is useful because it proves the grouped per-16 ABI
can host both halves of active-MoE under one native extension route. The result
is also sobering: P67 down-only was 1.266x, but composing gate/up tile + down
tile drops the full active-MoE win to 1.108x. The two-stage reference path still
pays launch overhead and writes an intermediate tensor, so it cannot be the
155TPS answer.

P68 narrows the next step:

- Stop treating gate/up and down as two independently promoted scalar kernels.
- Build a fused grouped per-16 active expert kernel that avoids the intermediate
  round-trip and keeps routing/top-k fixed inside one active-MoE boundary.
- Validate with full decode gates, not sub-kernel parity alone.

The measured signal is real, but the production path must be a larger fused
kernel, not another runtime toggle around this tiled reference.
