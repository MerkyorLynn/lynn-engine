# Qwen3.6-35B W4A16 Native MoE P147 Triton Stage Contract

**Date:** 2026-05-19  
**Model line:** official Qwen3.6-35B-A3B Lynn-native W4A16 NVFP4  
**Fixture source:** `/root/autodl-tmp/reports/qwen36_35b/p138_packed_slot_fixtures_kimi_20260518`

## Verdict

P147 reference generation is **GREEN**.

It generated 18/18 Triton active-MoE stage references and wrote them to:

```text
/root/autodl-tmp/reports/qwen36_35b/p147_triton_stage_reference_20260519_0318
```

This gives Native MoE workers a stricter pre-P37 target: match the exact Triton
stage output before spending R6000 time on resident generation gates.

## Timing Signal

| Stage | Mean latency |
|---|---:|
| Triton gate/up | 0.0685 ms |
| Triton down weighted-sum | 0.0265 ms |
| Active routed MoE stage total | 0.0950 ms |

These are slot-packed fixture-stage numbers, not full resident service TPS. The
important signal is the exact stage boundary and the relative budget: any new
candidate should first show exact stage parity, then beat this stage budget
before P37 escalation.

## Contract Locked By P147

The P147 reference uses the production Triton active-MoE math order:

1. local slot order is preserved;
2. route weights stay FP32;
3. gate/up decodes FP4 and accumulates in FP32;
4. gate/up stores a BF16 intermediate;
5. down reloads the BF16 intermediate into FP32;
6. down accumulates each slot in FP32;
7. route multiplication happens after the per-slot down projection;
8. final routed output is stored as BF16.

This closes the ambiguity from earlier PyTorch slot-order and native-output
fixtures. A candidate can be close to PyTorch or cosine-clean and still drift
in resident decode if it does not match this contract.

## Next Admission Rule

For any new Native MoE candidate:

1. run P134/P136 if it is still a fixture-level kernel;
2. run P147 against candidate stage outputs;
3. only if P147 is exact, run P146/P37 resident exact-greedy;
4. only if P37 is exact, run P25 and structured gates.

Do not promote or service-test approximate native reductions that fail P147.

## Artifacts

- `benchmarks/p147_triton_contract_moe_stage_gate.py`
- `scripts/r6000_qwen36_moe_p147_triton_stage_gate.sh`
- `reports/qwen36_35b/p147_triton_stage_gate_20260519_0318.json`
