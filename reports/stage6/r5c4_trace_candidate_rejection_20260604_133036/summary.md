# Stage 6 R5-C4 Trace-Derived Candidate Rejection Summary

| Field | Value |
|---|---|
| Result | `reports/stage6/r5c4_trace_candidate_rejection_20260604_133036/result.json` |
| Decision | `PASS_R5C4_TRACE_DERIVED_CANDIDATE_REJECTED` |
| Validator decision | `FAIL_R5C4_FULL_ACTIVE_MOE_PREFILL_SPEED_AB` |
| Validator rejected trace candidate | `True` |
| Same-scope false | `True` |
| Real model weights false | `True` |
| Full boundary timed false | `True` |
| Decode/default not banked | `True` / `True` |

## Boundary

- This artifact banks only rejection behavior for a trace-derived bad candidate.
- It proves R5-C3A gate/up timing plus R5-C3C host composition parity cannot be promoted as R5-C4 speed.
- It does not bank full active-MoE prefill speed, decode TPS, server/RC behavior, or defaults.

