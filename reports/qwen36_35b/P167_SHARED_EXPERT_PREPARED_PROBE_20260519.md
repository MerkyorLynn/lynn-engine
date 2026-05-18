# P167 Qwen3.6-35B Shared Expert Prepared Probe

Date: 2026-05-19

## Purpose

P158 showed the shared expert plus shared gate/finalize path is a material MoE
component. P167 tests whether an exact caller-owned BF16 prepared boundary can
recover enough of that cost to become a resident candidate.

The probe compares the default BF16 shared expert against `torch.mm(..., out=)`
scratch variants over 18 real Qwen3.6 W4A16 fixtures.

## Result

| Path | Exact | Mean ms/layer | Delta |
|---|---:|---:|---:|
| Shared default | reference | 0.03489 | - |
| Shared `mm_out` | 18/18 | 0.03304 | -0.00185 |
| Shared inplace SiLU | 0/18 | 0.03430 | closed |
| Finalize default | reference | 0.02977 | - |
| Finalize with `mm_out` shared | 18/18 | 0.06270 | +0.03293 |
| Finalize prepared/in-place | 18/18 | 0.06515 | +0.03538 |

## Decision

Close as a promotion candidate. The shared expert `mm_out` variant is exact, but
the mean savings are about 0.00185 ms/layer, roughly 0.055 ms/token across 30
linear-attention MoE layers. That is too small to move the 35B service target by
itself. The broader finalize-prepared variants are exact but much slower.

Keep the result as evidence: exact BF16 scratch ownership works for the shared
expert body, but the next speed push needs a larger boundary than shared expert
alone.

## Artifacts

- `benchmarks/p167_qwen36_shared_expert_prepared_probe.py`
- `scripts/r6000_qwen36_shared_expert_prepared_probe.sh`
- `reports/qwen36_35b/p167_qwen36_shared_expert_prepared_probe_20260519_1240.json`
