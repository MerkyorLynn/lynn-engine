# P179 Qwen3.6-35B Full-Attention MoE Tail Graph Probe

Date: 2026-05-19

## Purpose

Probe a larger exact boundary that keeps the existing Triton active-MoE math as
the numerical authority:

```text
post_attention_layernorm -> resident FFN/MoE -> residual add
```

This was meant to avoid approximate native MoE backends while still shrinking a
full-attention layer launch boundary.

## R6000 Result

| Metric | Result |
|---|---:|
| Full-attention layers probed | 10/10 |
| Captured successfully | 0/10 |
| Final verdict | `blocked` |

Each probed layer failed during CUDA graph capture with
`cudaErrorStreamCaptureInvalidated`.

## Decision

Do not wire a full-attention MoE-tail graph wrapper directly into resident
serving. The current exact Triton MoE path is not capture-safe as a black-box
tail. This does not mean the boundary is mathematically wrong; it means the
next useful MoE work needs a graph-safe scratch/output ABI inside the exact MoE
path before layer-tail graph capture can be retried.

## Follow-Up Direction

1. Keep Triton active-MoE as the exact numerical authority.
2. Move allocation/scratch ownership out of the active-MoE body.
3. Re-run this probe only after the exact MoE path is graph-capture safe.

## Artifacts

- `benchmarks/p179_qwen36_35b_full_attn_tail_graph_probe.py`
- `scripts/r6000_qwen36_35b_full_attn_tail_graph_probe.sh`
- `reports/qwen36_35b/p179_qwen36_35b_full_attn_tail_graph_probe_20260519_092222.json`
