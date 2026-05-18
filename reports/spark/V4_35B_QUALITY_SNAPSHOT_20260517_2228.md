# V4 35B Quality Snapshot

Date: 2026-05-17 22:28 CST

## Current Read

V4-Flash BF16 confirms the 35B distillation quality band, while V8-RTN W4A4
still erases most of that gain. This supports the route change: test 35B
W4A16/weight-only first, and only then decide whether W4A8 is worth the speed
bridge.

| Candidate | Quant/Runtime | MMLU 500 | GPQA Diamond |
|---|---|---:|---:|
| V4-Pro Q4_K_M | W4A16-like GGUF | 83.60% | 42.42% |
| V4-Flash Q4_K_M | W4A16-like GGUF | 80.60% | 43.94% |
| V4-Flash BF16 | BF16 | 80.40% | 44.44% |
| V4-Pro V8-RTN | W4A4 | 63.40% | 35.86% |
| V4-Flash V8-RTN | W4A4 | 63.00% | 37.88% |

## Implication

The V4-Flash BF16 result is close to V4-Flash Q4_K_M and far above V8-RTN.
That makes activation quantization the likely quality killer, not the 35B
distilled model family itself.

Immediate gates:

- R6000 direct V4-Pro BF16 download is still in progress.
- R6000 watcher will pack V4-Pro Lynn-native W4A16 after download.
- R6000 matrix watcher will compare W4A16 against W4A8 `gateup/full`.
- Spark V4-Pro BF16 is still pending, and should be treated as the strongest
  single quality signal when it lands.

Raw summaries are stored under:

```text
reports/spark/v4_quality_20260517/
```
