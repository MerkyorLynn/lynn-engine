# Stage 6 P4C Active-Reuse Microbench Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C active-reuse speed baseline recorded; speed/default promotion still closed) |
| Decision | `PASS_P4C_ACTIVE_REUSE_SPEED_BASELINE_RECORDED` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Torch/CUDA | `2.9.1+cu130` / `13.0` |
| Banked P4C speed baseline | `True` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |
| P4A two-stage median | `271.09758853912354` us |
| P4C active-reuse median | `271.06239795684814` us |
| P4C/P4A speedup | `1.0001298246549157` |
| P4C minus P4A | `-0.035190582275390625` us |
| Speed ratio floor | `0.8` |
| Output rel L2 | `0.0` |
| Output max abs | `0.0` |
| Scratch rel L2 | `0.0` |
| Scratch max abs | `0.0` |
| Active scratch bytes | `8192` |
| Zero BF16 shadow weight ABI | `True` |
| Packed/BF16 ratio | `0.3750001589457194` |
| Elapsed seconds | `26.751` |

## Boundary

- This banks only `banked_p4c_active_reuse_speed_baseline=true`.
- It is not a fused-kernel speed win and not a default-promotion gate.
- The next real implementation step is replacing the P4C symbol body with a faster active-reuse CUDA/CUTLASS-style candidate while preserving this ABI.
