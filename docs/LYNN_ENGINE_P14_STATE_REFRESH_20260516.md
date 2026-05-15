# Lynn Engine P14 State-Refresh Slot Direction (2026-05-16)

P13 showed the graph replay primitive is correct, but future graph windows are
not generally safe across prompts. The safe contract is still:

```text
capture/replay from the current real state
```

P14 investigates whether we can keep that contract while removing graph
capture from the hot path by refreshing graph-owned state buffers.

## Hypothesis

Reusable current-position graph slot:

```text
real state -> graph-owned state buffers
graph replay
graph-owned state buffers -> real state
```

If the state refresh roundtrip is much cheaper than graph capture, this route
can replace capture-per-token.

## R6000 Copy-Cost Probe

Command:

```bash
python benchmarks/p14_state_refresh_copy_cost_probe.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out /tmp/lynn_p14_state_refresh_copy_cost_probe.json \
  --prefix-new 16 \
  --iters 20
```

Result at sequence length 23:

| Component | Size |
|---|---:|
| KV cache | 0.039 GiB |
| Recurrent state | 0.059 GiB |
| Conv state | 0.001 GiB |
| Total mutable state | 0.099 GiB |

| Copy direction | Avg time |
|---|---:|
| real -> graph | 0.395 ms |
| graph -> real | 0.393 ms |
| roundtrip | 0.788 ms |
| roundtrip equivalent | 1269 tok/s |

## Interpretation

This is a strong green light for the state-refresh route:

```text
capture-per-token: 60-105 ms
state refresh roundtrip: ~0.8 ms
graph replay: ~12.6 ms
```

Even full-state refresh is far cheaper than capture. The first production-shaped
target is therefore:

```text
refresh state -> replay current-position graph -> commit graph state
```

If implemented correctly, the expected token cost is roughly:

```text
~0.8 ms state refresh + ~12.6 ms graph replay + small bookkeeping
```

That puts the route near the 70-80 TPS class before further graph/kernel
optimization. It is not yet the 100 TPS target, but it removes the largest P13
blocker and gives a clean engineering path.

## Next Gate

P14-B should implement a reusable graph-owned-state slot for one fixed position
and validate:

1. greedy ID parity,
2. graph/eager top-1 parity,
3. no future-window drift,
4. measured refresh + replay cost.

Only after P14-B passes should this be considered for an opt-in server path.

