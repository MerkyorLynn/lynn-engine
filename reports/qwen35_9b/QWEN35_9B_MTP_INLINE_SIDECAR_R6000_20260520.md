# Qwen3.5-9B Official Inline MTP on R6000

## Summary

Qwen3.5-9B ships an official inline NextN/MTP head inside the BF16 HF shards. Lynn now extracts it into a sidecar and can load it through the resident runner.

## Artifacts

- Extractor: `scripts/qwen35_extract_inline_mtp_sidecar.py`
- Smoke harness: `scripts/r6000_qwen35_9b_mtp_spec_smoke.py`
- Sidecar on R6000: `/root/autodl-tmp/models/mtp_sidecars/qwen35-9b-official-inline-lynn/mtp.safetensors`
- Extract report: `reports/qwen35_9b/qwen35_9b_mtp_inline_extract_20260520_133430.json`
- K1 smoke: `reports/qwen35_9b/qwen35_9b_mtp_spec_smoke_20260520_135119.json`
- K2 smoke: `reports/qwen35_9b/qwen35_9b_mtp_spec_smoke_20260520_135604.json`

## Extract Result

- Tensor count: 15 official `mtp.*` tensors.
- Sidecar size: 486,582,952 bytes.
- SHA256 prefix: `2ac294fb7fd33031`.
- Missing expected tensors: none.
- Unexpected MTP tensors: none.
- Dense MTP layout is supported by `engine/mtp_sidecar.py` via `mlp.gate_proj/up_proj/down_proj`.

## R6000 Smoke Results

All runs used Qwen3.5-9B Lynn-native NVFP4 W4A16 on R6000 with graph disabled for the first correctness-oriented MTP serving smoke.

| Mode | Events | Accepted | Accept Rate | Tokens Committed | Effective MTP TPS | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Baseline | 0 | 0 | n/a | 0 | n/a | control |
| Sequential K1 | 148 | 112 | 75.68% | 252 | ~20.7-23.6 per-prompt | MTP head works; no net speedup |
| Batched K2 + full-attn `t1_loop` | 144 | 116 | 80.56% | 252 | 23.10 mean | K2 path works; still eager/fallback-bound |

## Interpretation

This is a good 9B result: the official head is real, contract-compatible after dense-sidecar support, and accept rate is high. The current serving path is not a speed win yet because it is running with graph disabled and full-attention K2 forced through the strict `t1_loop` fallback.

The next speed target is not sidecar extraction. It is reducing K2 verifier cost:

1. Keep full-attn strictness, but avoid broad eager fallback where possible.
2. Measure K2 with production 9B default profile (`LYNN_LINEAR_BLOCK_GRAPH=1`) once speculative is graph-compatible.
3. Make dense K2 full-attn projection path T=2-capable or isolate the few full-attn layers behind a cheaper strict bridge.
4. Only promote if P37/structured/tool-call gates stay green and TPS beats the 61.7 default NVFP4 baseline.

## Current Promotion State

- Sidecar extraction: GREEN.
- MTP load + sequential accept: GREEN.
- Batched K2 accept: GREEN.
- Performance promotion: CLOSED for now; eager/fallback path is slower than the current graph baseline.
