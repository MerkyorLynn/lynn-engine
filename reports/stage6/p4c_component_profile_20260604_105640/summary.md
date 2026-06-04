# Stage 6 P4C Component Profile Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C component profile recorded; promotion still closed) |
| Decision | `PASS_P4C_COMPONENT_PROFILE_RECORDED` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Banked component profile | `True` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |
| Full P4C median | `233.39200019836426` us |
| Gate/up component median | `151.6703963279724` us |
| Down component median | `86.21439933776855` us |
| Component sum | `237.88479566574097` us |
| Gate share | `0.6375791941788874` |
| Down share | `0.3624208058211126` |
| Component sum / full | `1.0192499977015417` |
| Gate rel L2 / max abs | `0.0` / `0.0` |
| Down rel L2 / max abs | `0.0` / `0.0` |
| Composed rel L2 / max abs | `0.0` / `0.0` |
| Caveat | `component symbols allocate output tensors; use split for bottleneck direction only` |

## Boundary

- This banks only `banked_p4c_component_profile=true`.
- Component timings use existing allocation-returning symbols and are diagnostic only.
- The next speed candidate should target the larger component first, then rerun the P4C speed baseline and RC gates.
