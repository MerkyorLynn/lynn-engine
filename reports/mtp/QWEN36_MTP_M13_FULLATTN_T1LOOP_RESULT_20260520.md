# Qwen3.6-35B-A3B MTP M13 Full-Attn T1-Loop Result

**Date:** 2026-05-20  
**Host:** Spark  
**Model:** `Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000`  
**MTP sidecar:** `qwen36-35b-a3b-mtp-official-lynn-fused/mtp.safetensors`  
**Env:** `LYNN_FULL_ATTN_K2_BACKEND=t1_loop`

## Result

| Path | Exact | Prefix mean | Accept | Effective TPS | Ratio vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 6/6 | 100.83 | - | 25.79 | 1.000x |
| shadow | 6/6 | 100.83 | 81.44% | - | - |
| spec_k1 sequential | 6/6 | 100.83 | 75.13% | 20.97 | 0.813x |
| spec_k1_batched | 2/6 | 65.00 | 75.17% | 19.75 | 0.766x |

## Interpretation

The official Qwen3.6 MTP head is now clearly alive in Lynn: shadow accept is
81.44%, and both sequential and batched speculative paths are around 75%
accept. This confirms the earlier concat-order fix and the official-to-Lynn
fused sidecar conversion.

The remaining blocker is not accept rate. Batched K=2 still fails exact parity
(2/6 exact) and is slower than baseline. The strict full-attention T1-loop
fallback removes the layer-level K2-vs-two-T1 drift in the standalone diff
probe, but the end-to-end smoke still has sequence-level divergence, so one
additional verifier mismatch remains.

## Promotion Status

`CLOSED_FOR_PROMOTION`: MTP is now a valid high-accept research path, but it
must not be counted as TPS credit until batched speculative decoding reaches
exact parity and exceeds baseline TPS.

## Artifacts

- JSON: `reports/mtp/mtp_smoke_m13_fullattn_t1loop_20260520_011344.json`
- Prior diff proof: `reports/mtp/QWEN36_MTP_K2_T1FULL_STRICT_DIFF_RESULT_20260520.md`
