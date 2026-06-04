# Stage 6 P4C Gate/Up Shape Candidate Summary

| Field | Value |
|---|---|
| Verdict | **PASS** (P4C gate/up shape candidate recorded; default promotion still closed) |
| Decision | `PASS_P4C_GATEUP_SHAPE_CANDIDATE_RECORDED` |
| Device | `NVIDIA GB10` |
| Capability | `[12, 1]` |
| Current tile_inter | `8` |
| Candidate tile_inter | `2` |
| Current P4C median | `267.31839179992676` us |
| Candidate P4C median | `185.09119749069214` us |
| Candidate speedup vs current | `1.4442523222281798` |
| Candidate - current | `-82.22719430923462` us |
| P4A current tile median | `267.4207925796509` us |
| P4A candidate tile median | `185.043203830719` us |
| P4A candidate/current speedup | `1.445180298673884` |
| Candidate out rel L2 / max abs | `0.0` / `0.0` |
| Candidate scratch rel L2 / max abs | `0.0` / `0.0` |
| Banked P4C gate/up shape candidate | `True` |
| Banked fused kernel speed | `False` |
| Banked default promotion | `False` |

## Boundary

- This banks only `banked_p4c_gateup_shape_candidate=true`.
- It does not bank fused-kernel speed or default promotion.
- If this remains faster in server/RC context, wire it as an opt-in runtime default candidate and rerun quality gates.
