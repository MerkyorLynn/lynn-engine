# Lynn Engine P64: R6000 tile sweep closed

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P64

Spark sm_121 found a useful SP-01/SP-01.6 Triton autotune gain by expanding
small MoE tile candidates. P64 repeats the idea on R6000, but with the stricter
Lynn engine promotion rule: a candidate must preserve greedy token IDs under
the full-generate gate, not just improve a production-bench average.

P64 also extends the P37 gate with baseline overrides, so candidates can be
compared against the current best safe P63 path:

```bash
--baseline LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
```

## Candidates

Baseline:

```bash
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
```

Candidate A:

```bash
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
LYNN_MOE_FAST_FIXED=0
LYNN_MOE_GATE_BLOCK_INTER=4
```

Candidate B:

```bash
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
LYNN_MOE_FAST_FIXED=0
LYNN_MOE_DOWN_BLOCK_HIDDEN=4
```

Reports:

```text
reports/p16_155/p64_gate_inter4_vs_p63_generate_gate.json
reports/p16_155/p64_down_hidden4_vs_p63_generate_gate.json
```

## Results

| Candidate | Greedy IDs | Median speedup | Candidate median TPS | Decision |
|---|---:|---:|---:|---|
| gate/up `BLOCK_INTER=4` | FAIL | 0.984x | 99.75 | reject |
| down `BLOCK_HIDDEN=4` | FAIL | 1.049x | 106.19 | reject |

The down candidate has a real speed signal, but all three P37 prompts diverged
from the P63 baseline. The generated text stayed broadly coherent in this small
sample, but strict Lynn runtime promotion requires exact greedy parity unless a
separate quality-retention track explicitly accepts drift. This is the same
lesson as P56/P58/P62: small accumulation-order changes can look harmless
locally and still flip later tokens.

## Decision

P64 closes the R6000 Triton tile-sweep path:

- Spark tile wins do not transfer directly to R6000.
- R6000 `gate/up BLOCK_INTER=8` remains the safe shape.
- R6000 `down BLOCK_HIDDEN=4` is a useful design signal, not a default path.
- P63 `triton_fast_decode` remains the only newly safe small MoE improvement.

The next 155 TPS step should not be more free-form Triton tile sweeping. It
should target the actual bottleneck: grouped per-16 native FP4 active expert
FFN, while preserving the P37/P62 full-generate gate discipline.
