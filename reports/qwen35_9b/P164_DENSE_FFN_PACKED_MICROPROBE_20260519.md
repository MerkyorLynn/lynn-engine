# P164 Qwen3.5-9B Dense FFN Packed/NVFP4 Microprobe

Date: 2026-05-19

## Purpose

Test whether the existing Lynn-native packed NVFP4 wrappers can replace the
current Qwen3.5-9B dense FFN reference path.  This probe consumes the P159/P160
dense FFN fixtures and writes P160-compatible candidate outputs.

## Result

All tested packed paths are research-only.  The exact scalar bridge is close in
cosine but slower than the PyTorch fixture baseline; the fast native path is
faster but numerically too far from the fixture contract.

| Backend | Exact | Max Abs | Cosine Min | Total ms | Gate/Up ms | Down ms | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `auto` | 0/8 | 2.0 | 0.902371 | 0.4103 | 0.3501 | 0.0669 | closed |
| `scalar_bridge` | 0/8 | 0.0625 | 0.999991 | 0.5116 | 0.2976 | 0.2115 | closed, too slow |
| `native_fast_2d` | 0/8 | 5.125 | 0.895353 | 0.1782 | 0.1062 | 0.0570 | closed, numeric drift |

Reference context:

- P160 dense FFN fixture reference is exact at about 0.216 ms mean.
- `native_fast_2d` is about 18% faster than that reference, but its output is
  far outside the fixture contract.
- `scalar_bridge` confirms the manifest keys and packed weights are wired to the
  right dense FFN projections, but this path is not a speed candidate.

## Decision

Do not wire P164 into resident serving.  The existing packed wrappers are not
the 9B dense speed path as-is.

Next useful work is a dedicated dense FFN kernel or a corrected native-fast
activation/scale contract that can match P160 before P37/P25 escalation.

## Artifacts

- `reports/qwen35_9b/p164_dense_ffn_packed_microprobe_20260519_0630_packed_auto.json`
- `reports/qwen35_9b/p164_dense_ffn_packed_microprobe_20260519_0634_packed_scalar.json`
- `reports/qwen35_9b/p164_dense_ffn_packed_microprobe_20260519_0636_packed_nativefast.json`
