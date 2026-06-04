# Stage 6 P3-A Grouped-MoE Contract Probe Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P3-A contract probe pass; fused kernel not banked) |
| Banked fused kernel | `False` |
| Model | `/home/merkyor/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526` |
| Layer | `0` |
| Batches | `[1, 16, 64]` |
| Shape | `H=2048 I=512 E=256 top_k=8` |
| Tiles | `gate T=32 I=8 H=128; down H=8 I=512` |
| Numeric gate | `True` |
| Shadow absent at candidate start | `True` |
| Aggregate pass | `True` |
| BF16 active expert bytes | `1.500 GiB` |
| Packed active expert bytes | `0.563 GiB` |
| Inter scratch estimate | `0.000 GiB` |
| Memory after deleting BF16 active | `0.641 GiB` |
| Max candidate peak | `0.642 GiB` |
| Average P3-A vs BF16 speed | `0.760x` |
| Min cosine | `1.000` |
| Argmax matches | `3/3` |

## Per Batch

| Batch | Unique experts | BF16 active us | P3-A us | Speed | Cosine | Argmax |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 607.232 | 739.600 | 0.821x | 1.000 | True |
| 16 | 99 | 6442.240 | 8484.512 | 0.759x | 1.000 | True |
| 64 | 210 | 13835.768 | 19757.048 | 0.700x | 1.000 | True |

## Caveats

- Active MoE only: shared expert and router are excluded from the P3-A contract.
- The candidate consumes packed active weights after BF16 active shadows are deleted.
- Speed is reported, but this probe does not bank a fused P3 kernel.
