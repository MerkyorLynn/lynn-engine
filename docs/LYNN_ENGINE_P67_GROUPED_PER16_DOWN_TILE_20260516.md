# Lynn Engine P67: grouped per-16 down tile probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P67

P66 froze the grouped per-16 down projection ABI with a scalar reference
implementation:

```text
inter[top_k, 512] + expert_ids + routing_weights + Lynn per-16 down weight
  -> hidden[2048]
```

P67 moves the existing tile-hidden non-atomic down kernel behind a
grouped-per16-specific native API:

```text
down_grouped_per16_tile_reference(..., tile_hidden)
```

This keeps the experimental fast sub-kernel on the grouped per-16 route instead
of reusing the older P48 diagnostic API. The kernel is still scalar CUDA math;
it is not the final tensor-core grouped active expert kernel.

## Probe

```bash
python benchmarks/p67_grouped_per16_down_tile_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p67_grouped_per16_down_tile_probe.json \
  --layers 2 8 14 20 28 36 \
  --tile-hidden 2 \
  --warmup 4 \
  --iters 40
```

Report:

```text
reports/p16_155/p67_grouped_per16_down_tile_probe.json
```

## Result

| Metric | Value |
|---|---:|
| sampled layers | 2, 8, 14, 20, 28, 36 |
| tile hidden | 2 |
| sub-kernel contract pass | true |
| mean Triton down | 0.02620 ms |
| mean native scalar reference down | 0.03089 ms |
| mean native tile reference down | 0.02068 ms |
| tile vs Triton speedup | 1.266x |
| tile vs scalar reference speedup | 1.493x |
| min cosine vs Triton | 0.99999988 |
| max relative L2 vs Triton | 2.68e-5 |
| runtime promote | false |

P67 confirms that the tile-hidden down kernel is a real grouped per-16
sub-kernel building block. It is faster than both the Triton down segment and
the P66 scalar native reference while staying inside BF16-level numerical noise
on sampled real routed decode inputs.

## Decision

Do not promote P67 as a runtime default.

The positive result is intentionally scoped to the down sub-kernel. P48/P50
already showed that this family of down kernels can pass local parity while
complete decode drifts after several tokens because tiny accumulation-order
differences cascade through later layers.

P67 should be treated as a **kernel design signal**:

- The grouped per-16 down ABI is now stable enough to host faster inner loops.
- A non-atomic tile shape is a valid candidate for the down half.
- Production promotion still requires a full active-MoE kernel and full decode
  quality gates, not down-only parity.

The next engineering step is to use this ABI as the down half of a larger
grouped per-16 active expert FFN path, then validate the complete decode loop
with greedy parity plus V8/V9/tool-call retention gates.
