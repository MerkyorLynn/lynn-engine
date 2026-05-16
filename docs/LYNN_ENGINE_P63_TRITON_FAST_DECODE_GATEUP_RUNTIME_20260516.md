# Lynn Engine P63: Triton fast-decode gate/up runtime probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P63

P62 showed that the rejected `cuda_tile_inter` backend fails through tiny early
hidden drift that later flips a low-margin token.  Before writing more native
CUDA, P63 tries a safer micro-change: use the existing Triton kernel shape but
swap only the E2M1 decode expression tested in P53.

Runtime opt-in:

```bash
export LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
```

This started as a non-default full-generate gate candidate. The P37 gate below
passed, so it is safe to keep as a promotable gate/up variant while the larger
grouped per-16 active-expert kernel remains the real 155 TPS path.

## What changes

Only gate/up active expert decode changes:

```text
production:        nvfp4_grouped_gate_up_silu
P63 candidate:     nvfp4_grouped_gate_up_silu_fast_decode
down/router/shared unchanged
```

The goal is to verify whether the P53 numerically exact local result remains
exact under full decode and whether performance is stable enough to keep as an
allowlisted kernel variant.

## Gate

Use the standard P37 generate gate:

```bash
python benchmarks/p37_moe_config_generate_gate.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p63_triton_fast_decode_gateup_generate_gate.json \
  --max-new 64 \
  --candidate LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
```

Promotion requires:

- `new_ids_all_match=true`;
- no `!` loop / no-think failure pattern;
- median speedup above noise.

If exact but slower, P63 remains a code-supported research variant. If exact and
faster on a wider layer/prompt suite, it may become an allowlisted variant.

## Result

Report:

```text
reports/p16_155/p63_triton_fast_decode_gateup_generate_gate.json
```

Summary:

| Metric | Baseline | Candidate |
|---|---:|---:|
| prompt count | 3 | 3 |
| decode TPS mean | 100.21 | 102.31 |
| decode TPS median | 100.38 | 102.41 |
| decode TPS min | 98.97 | 102.09 |
| decode TPS max | 101.28 | 102.42 |

Gate verdict:

```json
{
  "new_ids_all_match": true,
  "median_speedup": 1.0202271656155624,
  "promote_default": true
}
```

## Decision

P63 passes as a small, quality-safe gate/up decode improvement:

- greedy token IDs match the Triton baseline on all P37 prompts;
- no `!` loop or no-think failure pattern appeared in the completion text;
- median decode speed improved by about 2%.

This is not enough to solve the active-MoE bottleneck by itself. It does,
however, confirm that the fast E2M1 decode expression can survive full-token
generation when it stays inside the Triton accumulation shape. The next
performance-critical step remains a grouped per-16 native FP4 active expert FFN
kernel, with P63 kept as a safe default/allowlisted gate-up decode variant.
