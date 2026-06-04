# Stage 6 P4B Candidate Numeric Preflight Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4B candidate output matches P4A two-stage reference; speed still unbanked) |
| Decision | `PASS_P4B_SINGLE_CTA_NUMERIC_REFERENCE` |
| Symbol | `active_moe_fused_zero_shadow_single_kernel_contract` |
| Reference symbol | `active_moe_fused_zero_shadow_out_contract` |
| Candidate mode | `multi_cta` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Torch/CUDA | `2.9.1+cu130` / `13.0` |
| Build dir | `/tmp/lynn_engine_native_build/p4b_single_cta_numeric_20260604_011110` |
| Banked single-CTA numeric preflight | `True` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |
| Reference finite | `True` |
| Candidate finite | `True` |
| Numeric vs reference | `True` |
| Candidate rel L2 | `0.0` |
| Candidate max abs diff | `0.0` |
| Reference norm | `3.834905146504752e-05` |
| Candidate norm | `3.834905146504752e-05` |
| Zero-shadow candidate ABI | `True` |
| Packed byte budget | `True` |
| No inter_scratch candidate ABI | `True` |
| Packed/BF16 ratio | `0.3750001589457194` |
| Elapsed seconds | `25.514` |
