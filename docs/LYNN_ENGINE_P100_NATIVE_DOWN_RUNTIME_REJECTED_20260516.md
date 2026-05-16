# Lynn Engine P100: native-down runtime gate rejected

## Result

P100 retested the native-down-only runtime shortcut after P99, with CUDA graph
capture disabled, to isolate whether the earlier native-down drift was a graph
artifact.

It was not.

| Gate | Candidate | Result |
|---|---|---|
| Graph-on native down | `LYNN_NATIVE_DOWN_BACKEND=cuda_tile` | greedy mismatch, `!` loop pattern |
| Graph-off native down | `LYNN_NATIVE_DOWN_BACKEND=cuda_tile`, all block graphs disabled | `new_ids_all_match=false`, median `1.049x` |

The graph-off run is the decisive one. It removes graph capture from the
equation and still fails the strict greedy-id contract.

## Evidence

Graph-off report:

```text
reports/p16_155/p99_native_down_only_generate_gate_graph_off.json
```

Summary:

```text
baseline median:   26.66 tok/s
candidate median:  27.97 tok/s
median speedup:    1.049x
new_ids_all_match: false
promote_default:   false
```

The outputs remain coherent in short samples, but the token IDs diverge on all
three prompts. That is enough to reject promotion for production.

## Decision

Do **not** promote native-down-only runtime replacement.

The isolated native-down kernel is still useful as a kernel-design signal:

- P95 showed `native_down_tile1` can beat the Triton down projection locally.
- P96/P97 showed native down can be composed in a quantized-reference active
  MoE experiment.
- P100 shows that the current runtime replacement does not preserve the
  production BF16-activation greedy contract.

So the path forward is not "turn on native down." The path forward is either:

1. preserve BF16 activation semantics and reduce scheduling/Python overhead
   around the current production kernels; or
2. move activation quantization into the A100 MTP/retrain/re-quant line, where
   W4A4 becomes an explicit model contract instead of a hidden runtime shortcut.

## Follow-up

P100 closes the last easy runtime toggle in the P95-P99 family. Future R6000
work should focus on production-safe graph/server consolidation or a new
BF16-preserving grouped active-MoE kernel. A100 owns the activation-quant-aware
artifact line.
