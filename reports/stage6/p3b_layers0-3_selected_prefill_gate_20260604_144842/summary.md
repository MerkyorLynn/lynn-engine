# Stage 6 P3-B Selected-Prefill Gate Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (selected-prefill composition gates passed) |
| Schema | `lynn-stage6-p3b-selected-prefill-gate-v1` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layers | `[0, 1, 2, 3]` |
| Layer types | `['linear_attention', 'linear_attention', 'linear_attention', 'full_attention']` |
| Seq lens | `[16, 64]` |
| Banked fused kernel | `False` |
| Banked server path | `False` |
| Predecessors pass | `True` |
| Numeric | `True` |
| Final stack cosine min | `1.000` |
| Final stack argmax | `True` |
| Active BF16 shadow absent | `True` |
| Reload trap installed | `True` |
| Reload not called | `True` |
| Speed vs P2-N reference | `True` |
| BF16 active expert bytes | `6442450944` |
| Packed active expert bytes | `2415919136` |
| Memory after load | `8.517 GiB` |
| Memory after active-shadow delete | `2.525 GiB` |
| Active-shadow memory drop | `5.991 GiB` |
| Avg BF16 prefill | `75136.610 us` |
| Avg P2-N reference | `62453.167 us` |
| Avg P3-B candidate | `61278.111 us` |
| Avg P3-B vs BF16 | `1.297x` |
| Avg P3-B vs P2-N | `1.020x` |

## Per Sequence

| Seq | BF16 us | P2-N us | P3-B us | P3-B/BF16 | P3-B/P2-N | Cosine | Argmax |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 57279.362 | 39298.401 | 38503.967 | 1.488x | 1.021x | 1.000 | `True` |
| 64 | 92993.858 | 85607.933 | 84052.254 | 1.106x | 1.019x | 1.000 | `True` |

## Caveats

- P3-B is selected-layer composition only; it does not bank a fused P3 kernel.
- Router and shared expert remain on the existing BF16 paths.
- Active routed expert BF16 shadows are deleted before P2-N/P3-B candidates run.
- P3-C server readiness and RC quality promotion remain separate gates.
