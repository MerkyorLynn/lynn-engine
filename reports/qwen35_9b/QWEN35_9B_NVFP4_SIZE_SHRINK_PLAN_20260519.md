# Qwen3.5-9B NVFP4 Size Shrink Plan · 2026-05-19

## Current Size Split

R6000 audit of `Qwen3.5-9B-lynn-native-w4a16-nvfp4-v0`:

| Bucket | Stored Size | Notes |
|---|---:|---|
| Quantized dense MLP | 2.813 GiB | already Lynn-native NVFP4 |
| Quantized linear attention | 0.941 GiB | already Lynn-native NVFP4 |
| Quantized full attention | 0.273 GiB | already Lynn-native NVFP4 |
| Quantized visual tensors | 0.265 GiB | removable in text-only package |
| Quantized MTP tensors | 0.142 GiB | removable if not shipping MTP |
| `lm_head.weight` BF16 | 1.895 GiB | main shrink/speed lever, quality risk |
| `embed_tokens.weight` BF16 | 1.895 GiB | main shrink lever, little decode speed impact |
| norms / small params | ~0.003 GiB | not worth touching |

Total kept BF16 is about 3.792 GiB, almost entirely `lm_head` plus
`embed_tokens`.

## Important Negative Result

`lm_head.weight` and `model.language_model.embed_tokens.weight` are not tied:

| Check | Result |
|---|---:|
| `tie_word_embeddings` | `false` |
| shape equality | true |
| exact equality | false |
| cosine | 0.0198 |
| mean abs diff | 0.0161 |

So there is no free alias/dedup trick.  Dropping `lm_head` and reusing
`embed_tokens` would be a serious model change.

## Shrink Ladders

| Ladder | Expected Size | Speed Impact | Quality Risk | Recommendation |
|---|---:|---:|---|---|
| Text-only prune visual/MTP | ~7.9 GiB | none/small load win | low | good release packaging option |
| Quantize `embed_tokens` only | ~6.9-7.4 GiB | little decode gain | medium | test after W4A8 route |
| Quantize `lm_head` only | ~6.4-6.9 GiB | possible decode gain | high | exact gate currently failed in prior `native_lm_head_only` sweeps |
| Quantize both embed + lm_head | ~5.5-6.0 GiB | possible load + lm_head gain | high | only after MMLU/GPQA + structured gates |
| GGUF-style Q_K mix | ~5.3 GiB | llama.cpp path | known | keep as Mac/llama.cpp artifact |

## Why `lm_head` Is Sensitive

Prior 9B R6000 sweeps show native FP4 `lm_head` is not yet safe:

- `native_lm_head_only`: exact `1/3`, decode about 41 TPS.
- `graph_plus_triton_core_native_lm`: exact `1/3`, decode about 75.6 TPS.

This means `lm_head` quantization cannot be slipped into default packaging just
for size.  It needs its own gate:

1. top-k / top-1 logit parity on prompt-derived hidden states;
2. P184 structured exact or prefix gate;
3. MMLU/GPQA spot-check;
4. only then service TPS.

## Practical Decision

Short term:

- Keep shipping full W4A16/W4A8 NVFP4 package for correctness.
- Add a text-only slim package only if distribution size matters immediately.
- Put real speed effort into W4A8 dense FFN / FP8-active kernels first.

Medium term:

- Build a `qwen35-9b-nvfp4-slim-lmhead` candidate that stores `lm_head` packed
  offline and loads a native projection path.
- Promote only if it passes the same generation gates as the current
  convstrict W4A16/W4A8 route.
