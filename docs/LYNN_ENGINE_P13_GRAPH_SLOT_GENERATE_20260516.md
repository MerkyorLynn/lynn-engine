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

