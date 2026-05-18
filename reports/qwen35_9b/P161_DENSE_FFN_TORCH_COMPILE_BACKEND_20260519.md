# Qwen3.5-9B P161 Torch Compile Dense FFN Probe

**Date:** 2026-05-19  
**Model:** official Qwen3.5-9B Lynn-native W4A16 NVFP4  
**Candidate:** `BACKEND=torch_compile COMPILE_MODE=reduce-overhead COMPILE_FULLGRAPH=1`

## Result

Artifacts:

- `reports/qwen35_9b/p161_dense_ffn_candidate_outputs_20260519_0548_torch_compile.json`
- `reports/qwen35_9b/p161_dense_ffn_candidate_p160_contract_20260519_0548_torch_compile.json`

| Metric | Reference backend | torch_compile |
|---|---:|---:|
| candidate_ms_mean | 0.216 ms | 0.625 ms |
| p160 reference exact | 8 / 8 | 8 / 8 |
| p160 candidate exact | 8 / 8 | 0 / 8 |
| candidate max_abs range | 0 | 2.44e-4 to 9.77e-4 on sampled rows |
| Decision | GREEN | RED |

## Interpretation

`torch.compile` is not a useful 9B dense-FFN path here. It is slower than the
plain PyTorch reference and changes candidate outputs by small BF16-level
amounts, so it should not enter serving gates.

The 9B dense FFN speed work should move to a real packed/native/fused kernel
against the P159/P160/P161 fixture contract.
