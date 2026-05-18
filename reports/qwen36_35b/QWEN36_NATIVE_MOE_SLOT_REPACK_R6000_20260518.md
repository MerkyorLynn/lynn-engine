# Qwen3.6 35B W4A16 Native MoE Slot Repack R6000 Probe

Date: 2026-05-18

## Scope

Validated the Claude/KIMI slot-repack MoE path on R6000 without merging the whole side branch:

- p135 exports slot-ordered BF16 gate/up and down weights from existing p133 W4A16 fixtures.
- p136 checks slot-only recomputation against the stored routed-output reference.
- `native_slot_output_owned_bf16` calls the native CUDA slot-output-owned kernel directly.

## Result

This is a useful research artifact, but it is not a strict promotion candidate.

| Stage | Result |
| --- | --- |
| p135 slot repack | 18/18 fixtures exported |
| p135 size | 865 MB total, about 49 MB per fixture |
| p136 strict contract | RED, 0/18 strict GREEN |
| p136 max_abs_max | 0.00390625 |
| p136 cosine_min | 0.9999865889549255 |
| p136 ref_ms_mean | 0.9001 ms |
| p136 slot_repack_ms_mean | 0.4555 ms |
| native slot candidate | RED under strict contract |
| native slot max_abs_max | 0.00390625 |
| native slot cosine_min | 0.9999802112579346 |
| native slot latency mean | 0.0520 ms |
| Triton active reference | 0.059 ms |

## Interpretation

Slot repack removes dynamic expert gather and unique-expert masking, and the native output-owned slot kernel is faster than the Triton active reference on the fixture microbench.

The blocker is numerical strictness. The p136 slot-only path and native slot candidate both differ from the stored routed reference at the BF16/accumulation-order level. Late layers amplify the absolute error, with L39 reaching 0.00390625. This is too large for the exact-greedy serving path, so it must not be promoted into the resident runner.

## Next Work

1. Keep p135/p136 and the native slot candidate as the fast MoE fixture harness.
2. Ask the next native-kernel worker to eliminate the BF16 intermediate round-trip or match the reference accumulation order exactly.
3. Only escalate a candidate to P37/P25/structured gates after p136 is either strict GREEN or explicitly classified as a relaxed research candidate with a documented drift budget.

Artifacts:

- `reports/qwen36_35b/p135_repacked_fixtures_manifest_20260518.json`
- `reports/qwen36_35b/p136_slot_repack_contract_report_20260518.json`
- `reports/qwen36_35b/native_slot_output_owned_bf16_report_20260518.json`
