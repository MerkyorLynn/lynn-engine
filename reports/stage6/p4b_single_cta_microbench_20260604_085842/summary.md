# Stage 6 P4B Single-CTA Microbench Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (measurement recorded; speed/default promotion still closed) |
| Decision | `PASS_P4B_SINGLE_CTA_MICROBENCH_RECORDED` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Torch/CUDA | `2.9.1+cu130` / `13.0` |
| Banked microbench | `True` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |
| P4A two-stage median | `282.51519203186035` us |
| P4B single-CTA median | `39539.166259765625` us |
| P4B/P4A speedup | `0.007145198514702697` |
| P4B minus P4A | `39256.651067733765` us |
| Numeric rel L2 | `0.0` |
| Numeric max abs | `0.0` |
| No inter_scratch candidate ABI | `True` |
| Packed/BF16 ratio | `0.3750001589457194` |
| Elapsed seconds | `30.817` |
