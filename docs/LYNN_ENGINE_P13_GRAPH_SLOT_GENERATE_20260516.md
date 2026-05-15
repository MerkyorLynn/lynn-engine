# Lynn Engine P13 Graph-Slot Generate Wiring (2026-05-16)

P10 proved current-position full-token graph slots are strict as a benchmark
primitive. P12 proved packed-resident memory release is compatible with graph
slots after BF16 shadows are dropped. P13 starts wiring that primitive into the
real `LynnIncrementalRunner.generate()` loop.

## Gate

New opt-in:

```bash
export LYNN_FULL_TOKEN_GRAPH_SLOT=1
```

When enabled, each decode step does:

```text
capture current-position full-token graph slot
replay graph slot
advance greedy token/state
```

This is intentionally a correctness-first implementation. Capturing every
token is expected to be slow. The next production step is moving capture out of
the hot path through a safe slot/window lifecycle.

## R6000 Result

Command:

```bash
python benchmarks/p13_full_token_graph_slot_generate_smoke.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out /tmp/lynn_p13_full_token_graph_slot_generate_smoke.json \
  --max-new 16
```

Result:

| Metric | Value |
|---|---:|
| Greedy ID parity | PASS |
| New tokens | 16 |
| Capture steps | 15 |
| Replay steps | 15 |
| Avg replay | 12.82 ms/token |
| Replay-only TPS | 78.01 tok/s |
| Avg capture | 105.25 ms/token |
| End-to-end graph-slot decode TPS | 8.46 tok/s |

The generated token IDs are exactly identical to the eager path:

```text
[271, 248068, 271, 248069, 271, 24797, 36, 9616,
 44, 12370, 314, 49051, 7313, 96580, 2005, 104916]
```

## Interpretation

This gate proves that the runner's real generation loop can use the
full-token graph-slot primitive without greedy drift. It does **not** claim a
new production speedup yet: capture-per-token dominates.

The useful signal is the split:

```text
replay-only:      ~78 TPS
capture overhead: ~105 ms/token
```

So the next bottleneck is not graph replay correctness; it is graph-slot
lifecycle management.

## Next Step

P13-B should avoid capture-per-token on the hot path. Candidate routes:

1. Lazy slot cache for repeated positions/state contracts.
2. Short sequential capture window after a real prefix, then replay the window.
3. Hybrid path: stable eager / linear-block serving until enough slots are
   warmed, then switch to replay.

Do not promote `LYNN_FULL_TOKEN_GRAPH_SLOT=1` to default serving until a
multi-token replay window passes greedy parity over multiple prompts.

## P13-B Window Probe

We reused the sequential-capture graph-family probe to test whether graph slots
can be captured as a short window and replayed continuously.

### Window = 8

```bash
python benchmarks/p9k_sequential_capture_graph_family_greedy.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out /tmp/lynn_p13b_seq_capture_window8.json \
  --max-new 8
```

Result:

| Metric | Value |
|---|---:|
| Greedy pass | true |
| Replay TPS | 79.27 tok/s |
| Amortized TPS including capture | 13.82 tok/s |
| First diagnostic drift | none |

This proves a short sequential graph window can preserve greedy parity. It is
not fast end-to-end yet because capture is still paid up front.

### Window = 16

```bash
python benchmarks/p9k_sequential_capture_graph_family_greedy.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out /tmp/lynn_p13b_seq_capture_window16.json \
  --max-new 16
```

Result:

| Metric | Value |
|---|---:|
| Gate pass | false |
| Replay TPS | 79.22 tok/s |
| Amortized TPS including capture | 13.27 tok/s |
| First diagnostic top-1 drift | step 9 |
| Step-9 cosine | 0.9935 |
| Step-9 top-10 overlap | 9/10 |

The token-id list happened to remain identical in this prompt, but the
diagnostic top-1 check already drifted at step 9. We therefore treat 16-token
windows as unsafe for production promotion today.

Current safe boundary:

```text
single current-position slot: strict
8-token sequential window: greedy-safe on this prompt
16-token sequential window: not safe (step-9 diagnostic drift)
```

The next useful optimization is not "make a bigger fixed window"; it is a
guarded 8-token window policy with replay/eager parity checks across multiple
prompts, then capture amortization.
