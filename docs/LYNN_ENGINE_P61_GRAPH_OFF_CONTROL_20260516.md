# Lynn Engine P61: graph-off Triton control

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P61

P58 disabled the linear-block graph while testing `cuda_tile_inter`, then saw
greedy mismatch.  To avoid a false causal claim, P61 runs the missing control:

```text
same P37 generate gate
candidate only disables LYNN_LINEAR_BLOCK_GRAPH*
keeps the active MoE backend on Triton
```

## Result

Report:

```text
reports/p16_155/p61_graph_off_triton_control_generate_gate.json
```

Summary:

```text
new_ids_all_match: true
baseline median:   ~100.13 tok/s
candidate median:  28.22 tok/s
median speedup:    0.282x
promote_default:   false
```

Turning graph off is slow but exact. Therefore P58's greedy mismatch is not
caused by graph-off mode itself.

## Decision

This tightens the P58 conclusion:

```text
cuda_tile_inter + graph-off mismatch
Triton + graph-off exact
=> cuda_tile_inter scalar accumulation/order is the unsafe variable
```

Keep `tile_inter=2` only as a kernel-shape hint. The next backend remains the
reserved `grouped_per16` implementation from P60.
