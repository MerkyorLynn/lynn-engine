# Qwen3.5-9B W4A8 Route Gate · 2026-05-19

## Spark Quality Signal

Spark reproduced the 9B Dense W4A16 result and added a W4A8 fake-quant quality
row on the same server/dataset/seed:

| Variant | MMLU 500 | GPQA Diamond |
|---|---:|---:|
| BF16 thinking-off | 77.20% | 44.95% |
| Q4_K_M default | 76.00% | 37.37% |
| Lynn W4A16 NVFP4 | 76.00% | 42.93% |
| Lynn W4A8 fake-quant | 75.80% | 43.94% |

Interpretation: W4A8 is quality-safe for the 9B Dense line within this gate.
It is effectively flat versus W4A16 on MMLU (-0.20pp) and slightly higher on
GPQA (+1.01pp).  Against BF16, GPQA loss is about -1.01pp, much smaller than
the current Q4_K_M loss.

## R6000 Admission Plan

The R6000 route now has two gates:

1. `p185_qwen35_9b_dense_w4a8_fixture_gate.py`
   - Uses P159 dense FFN fixtures.
   - Compares W4A16 reference against W4A8 `gateup` and `full` activation
     fake-quant.
   - Reports drift and emulation timing; this is the kernel-development
     admission gate.

2. `p186_qwen35_9b_dense_w4a8_resident_gate.py`
   - Runs the 9B safe `convstrict` resident profile.
   - Compares W4A16, W4A8 `gateup`, and W4A8 `full` on hard structured
     prompts.
   - Reports exact count, prefix drift, and decode TPS.  Fake-quant TPS includes
     FP8 round-trip emulation overhead and is not the final native FP8 speed.

The queue on R6000 is ordered as:

```text
current Q4_K_M 32K GPQA50
→ official 9B BF16/NVFP4 round
→ W4A8 route gate
→ Q4_K_M-imatrix gate
```

## Engineering Implication

If the R6000 W4A8 generation gate is GREEN or AMBER-prefix, the next useful
work is not more fake-quant testing.  It is a true FP8-active dense FFN kernel:
offline W4A8 activation contract first, then replace fake-quant emulation with
R6000/Spark-specific native kernels.
