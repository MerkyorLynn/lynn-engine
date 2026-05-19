# P171 Linear Core Candidate Output Smoke

Date: 2026-05-19

## Purpose

P171 validates the candidate-output-dir plumbing for the Qwen3.6-35B-A3B W4A16 linear-core fixture contract.  It does not claim a speedup.  It proves that a future fused native `in_proj -> conv -> recurrent/GDN` candidate can emit safetensors without loading the full resident server and still be admitted by P169 before any P37/P25/structured gate.

## Artifacts

- Helper: `benchmarks/p171_qwen36_linear_core_candidate_output_smoke.py`
- R6000 wrapper: `scripts/r6000_qwen36_linear_core_candidate_output_smoke.sh`
- Full identity report: `reports/qwen36_35b/p171_linear_core_candidate_output_smoke_20260519_080056.json`
- Full identity P169 report: `reports/qwen36_35b/p171_linear_core_candidate_output_smoke_20260519_080056_p169_check.json`
- Only-final report: `reports/qwen36_35b/p171_linear_core_candidate_output_smoke_20260519_080056_onlyfinal.json`
- Only-final P169 report: `reports/qwen36_35b/p171_linear_core_candidate_output_smoke_20260519_080056_onlyfinal_p169_check.json`

## R6000 Results

| Mode | Fixtures | Tensors Written | P169 Passed | max_abs_max | cosine_min |
|---|---:|---:|---:|---:|---:|
| identity-reference | 20 | 320 | 20/20 | 0.0 | 0.999999463558197 |
| only-final | 20 | 60 | 20/20 | 0.0 | 0.9999998211860657 |

Both modes are GREEN.  The only-final mode writes only:

- `linear_core_out`
- `recurrent_state_out`
- `conv_state_out`

This is the intended minimum output set for the next real fused linear-core kernel candidate.

## Next Admission Contract

A real P171/P172 fused candidate should:

1. Read the P169 fixtures under `/root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750`.
2. Emit one safetensors file per fixture under a candidate output directory.
3. At minimum write `linear_core_out`, `recurrent_state_out`, and `conv_state_out`.
4. Pass:

```bash
python benchmarks/p169_qwen36_linear_core_fixture_contract.py \
  --fixtures /root/autodl-tmp/reports/qwen36_35b/p169_linear_core_fixtures_official_w4a16_20260519_0750 \
  --candidate-output-dir <candidate-output-dir> \
  --out <p169-candidate-report.json> \
  --check
```

Escalation to P37/P25/structured remains blocked until this fixture gate is GREEN.

