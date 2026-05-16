# Lynn Engine P58: gate/up tile-inter graph-off retest rejected

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P58

P56 found a tempting runtime signal:

```text
LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter
LYNN_NATIVE_GATEUP_TILE_INTER=2

median decode: 114.43 tok/s
failure: `!` loop / greedy mismatch
```

The open question was whether the failure came from the CUDA extension being
captured/reused by the linear-block graph path, or whether the tile-inter
scalar accumulation order itself was unsafe for full greedy decode.

P58 reruns the same candidate with the reusable linear-block graph fully
disabled:

```bash
LYNN_LINEAR_BLOCK_GRAPH=0
LYNN_LINEAR_BLOCK_GRAPH_REUSE=0
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=0
```

## Result

Report:

```text
reports/p16_155/p58_gateup_tile_graph_off_generate_gate.json
reports/p16_155/p61_graph_off_triton_control_generate_gate.json
```

Summary:

```text
baseline median:   99.55 tok/s
candidate median:  28.43 tok/s
median speedup:    0.286x
new_ids_all_match: false
promote_default:   false
```

The candidate no longer shows the P56 high-TPS signal because disabling the
graph path returns decode to a much slower eager cadence. More importantly,
greedy IDs still do not match baseline.

Control:

```text
Triton + graph disabled:
  new_ids_all_match: true
  candidate median:  28.22 tok/s
```

So "graph disabled" alone is exact. The P58 mismatch is caused by the
`cuda_tile_inter` gate/up scalar implementation, not by the absence of the
linear-block graph path.

## Decision

P56 was **not** merely a graph/extension interaction bug, and P58 proves the
graph-off control itself is exact.

The scalar `tile_inter=2` implementation changes enough decode numerics/order
to perturb greedy generation even when graph capture is removed. Therefore:

- keep `tile_inter=2` as a shape hint;
- do not promote `LYNN_NATIVE_GATEUP_BACKEND=cuda_tile_inter`;
- do not spend more time wrapping the scalar tile kernel as a production path;
- move the effort to a true grouped per-16 native-FP4 active expert kernel.

## Next path

P59 locks the dual-artifact NVFP4 dispatch contract so Lynn engine can support
both:

1. the current Lynn-native per-16 variable-expert artifact; and
2. a future vendor-friendly ModelOpt / compressed-tensors NVFP4 v2 artifact.

P60 then resumes the real 155TPS path: grouped per-16 active expert FFN.
