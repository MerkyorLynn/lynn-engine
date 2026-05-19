# Qwen3.6 MTP M15 State Boundary Trace 2026-05-20

## Purpose

M13 proved the official Qwen3.6-35B-A3B MTP head is real:

- shadow/spec-k1 accept is about 75-81%;
- sequential `spec_k1` is exact on the smoke set;
- batched `spec_k1_batched` still drifts and is slower than baseline.

M15 narrows the batched failure. Instead of running another full smoke, it
compares speculative batched state against canonical T=1 state at every
committed-token boundary.

## Artifacts

- `scripts/spark_mtp_state_boundary_trace.py`
- `scripts/spark_mtp_k2_vs_t1_diff_probe.py`
- `reports/mtp/remote_spark_20260520/mtp_state_boundary_trace_20260520_022720.json`
- `reports/mtp/remote_spark_20260520/mtp_state_boundary_trace_kv_20260520_024228.json`
- `reports/mtp/remote_spark_20260520/mtp_state_boundary_trace_inplace_20260520_025646.json`
- `reports/mtp/remote_spark_20260520/mtp_k2_vs_t1_event5_diff_20260520_023538.json`
- `reports/mtp/remote_spark_20260520/mtp_k2_vs_t1_event5_after_mtp_call_20260520_025018.json`
- `reports/mtp/remote_spark_20260520/mtp_k2_vs_t1_event5_k2first_20260520_030910.json`

## Findings

### Boundary Trace

The first bad boundary appears at event 5:

| Field | Value |
|---|---:|
| event | 5 |
| accepted | true |
| committed tokens | `[25, 271]` |
| next pending match | true |
| baseline seq_len / spec seq_len | 29 / 29 |
| worst state field | `conv`, layer 38 |
| state max_abs | 0.25 |
| next hidden max_abs | 0.046875 |

Including KV in the comparison does not move the first bad event. The first
visible state drift is still `conv_state` at linear-attention layer 38.

Setting `LYNN_LINEAR_STATE_UPDATE=inplace` also does not fix it. The first bad
event remains event 5 with the same layer-38 conv drift.

### Event-5 Clean Diff

When the exact token prefix before event 5 is replayed with canonical T=1
decode, then K=2 is compared against two T=1 steps for pending/draft `[25,271]`,
the result is exact:

| Probe | Result |
|---|---|
| K2 after sequential verifier | all layers within threshold |
| logits pos0 / pos1 | max_abs 0 |
| state worst | max_abs 0 |
| calling MTP before compare | still exact |

This rules out the official MTP head call and the event-5 token pair itself.

### K2-First Diff

When the order is reversed to match the real speculative event more closely
(`K2` first, then restore and run the sequential verifier), drift appears:

| Field | Value |
|---|---:|
| first bad layer | 32 |
| layer type | linear_attention |
| bad position | pos1 only |
| pos1 max_abs | 0.00390625 |
| pos1 cosine | 0.99999666 |
| logits pos1 max_abs | 0.8046875 |
| state worst | `conv`, layer 38, max_abs 0.25 |
| pos0 argmax | exact |
| pos1 argmax | exact in this local probe |

## Interpretation

The remaining batched MTP bug is not the head weights, concat order, lm-head
shape dispatch, MoE backend, or a single clean K2 event. It is an
order-dependent K2 verifier/state-boundary issue in the linear-attention path.

The narrowest repro is:

1. advance to the event-5 prefix;
2. run K2 before the sequential verifier;
3. the first layer-level drift appears at linear-attention layer 32, position 1;
4. the accumulated state drift is visible as layer-38 conv-state drift.

This explains why simple single-step probes passed while multi-event batched
generation drifted.

## Promotion Status

Do not promote batched MTP yet.

Current safe facts:

- official MTP head is usable;
- sequential speculative serving is exact but slower than baseline;
- batched speculative serving has high accept but is not exact and remains
  below baseline TPS.

## Next Work

1. Make `decode_linear_attn_k2` order-stable for position 1, starting with
   layer 32 in the event-5 repro.
2. Compare K2-first vs sequential with the same state, not only sequential-first
   vs K2.
3. Only after the M15 K2-first probe is exact, rerun M13 smoke.
4. If exactness returns but TPS remains below baseline, the speed unlock moves
   to true batched linear/GDN kernels; MTP head quality itself is no longer the
   blocker.
