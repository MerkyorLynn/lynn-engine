# Lynn Engine P19 — Active MoE Block Retune (2026-05-16)

P18 ruled out the simple `tl.dot_scaled` scale bridges. P19 returns to the
current quality-safe per-16 active MoE kernels and retunes the launch/block
shape.

## Result

R6000, 27B NVFP4 step5000, `p9p_hybrid_group_latency_breakdown.py`, group size
20, native FP4 lm_head enabled:

| Config | Strict full | Replay-only |
|---|---:|---:|
| P15 baseline (`gate_hidden=64`, `down_inter=256`) | 103.40 TPS | 107.13 TPS |
| P19 best (`gate_hidden=256`, `down_inter=512`) | **115.41 TPS** | **120.25 TPS** |

The active block sweep first found the best local shape on layer 28:

```text
LYNN_MOE_GATE_BLOCK_INTER=8
LYNN_MOE_GATE_BLOCK_HIDDEN=256
LYNN_MOE_DOWN_BLOCK_HIDDEN=8
LYNN_MOE_DOWN_BLOCK_INTER=512
```

Then the full 40-layer graph benchmark confirmed it:

```text
groups_ms:                         8.3458
full_decode_final_graph_ms:        8.6649
full_decode_final_graph_tps:       115.41
full_decode_final_graph_replay_ms: 8.3159
full_decode_final_graph_replay_tps:120.25
```

## Quality / Safety

The block retune does not change routing, quantization, or approximation. It
only changes Triton block scheduling for the same scalar per-16 NVFP4 kernels.

Layer-28 active block sweep showed exact output match against the previous
default for the chosen config:

```text
diff_vs_default.max_abs = 0
diff_vs_default.cosine  = 1.0
```

## Meaning

This is a real production-safe gain:

- strict full path improves by about **+11.6%**,
- replay-only graph ceiling improves by about **+12.3%**,
- current R6000 ceiling moves from the 107 TPS class to the **120 TPS class**.

It does **not** close the 155 TPS target by itself. P16/P18 still show that the
remaining gap requires a new grouped native-FP4 active expert kernel or a
quality-safe quantization format that can use Blackwell tensor cores directly.
