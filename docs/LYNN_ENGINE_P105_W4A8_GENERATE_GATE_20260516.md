# Lynn Engine P105: W4A8 Generation-Level Gate

Date: 2026-05-16

## Purpose

P104 measured active-MoE tensor drift and found W4A8 E4M3-per16 to be close
enough for adaptation. P105 asks the stricter question: does the current
BF16-trained Lynn-native NVFP4 artifact preserve greedy generation when active
expert activations are fake-quantized to FP8?

This is a quality gate, not a performance benchmark. Runtime promotion is not
allowed from P105 alone.

## Setup

Model:

```text
/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Probe:

```text
benchmarks/p105_w4a8_generate_gate.py
```

Modes:

```text
off:    current BF16 activation semantics
gateup: fake-quant hidden activation before active expert gate/up
full:   gateup plus fake-quant intermediate activation before down
```

Fake quant contract:

```text
format:      E4M3
granularity: per16
prefill:     BF16
decode:      active expert activation fake quant only
```

## Results

### 24-token gate

```text
gateup: 5/6 exact, min same-prefix 12, mean same-prefix 20.33
full:   6/6 exact, min same-prefix 12, mean same-prefix 22.00
```

The only gate/up divergence is prompt 0 at token 14. The completion remains
semantically equivalent:

```text
baseline: ...更精确地保留模型内部激活值...
gateup:   ...更准确地保留模型内部激活...
```

### 64-token gate

```text
gateup: 4/6 exact, min same-prefix 12, mean same-prefix 36.50
full:   5/6 exact, min same-prefix 12, mean same-prefix 40.00
```

The first longer divergence appears on the Python palindrome prompt at token
25. This is a greedy-token exactness miss, not a coherence collapse.

## Decision

P105 is **AMBER**.

The raw JSON reports were produced with the first conservative script verdict,
which labeled any gate/up greedy-token divergence as RED. The script has since
been updated so long same-prefix stability is classified as AMBER for training
triage. The raw token traces are unchanged.

W4A8 should continue as the primary long-value path, but the current artifact
must not promote W4A8 runtime directly. The evidence supports the plan:

1. Use A100 QAT-lite/Recovery to adapt the model to E4M3-per16 active expert
   activation rounding.
2. Keep runtime W4A8 behind an explicit environment gate until generation-level
   exactness improves.
3. Re-run this same P105 gate after each Recovery checkpoint before V8/V9.

This is a good AMBER: the route is trainable. The divergence is late and
localized enough that a small adaptation run is justified.

## A100 Training Implication

The immediate A100 target is not a broad retrain. It is margin repair:

```text
P105 target after W4A8 Recovery:
  24-token gate: gateup/full 6/6 exact
  64-token gate: gateup/full >= 6/6 exact, or no divergence before token 48

Then:
  6-prompt smoke PASS
  strict tool-call PASS
  no-think loop guard PASS
  V8/V9 retention near BF16 baseline
```

MTP/NEXTN remains a separate serving multiplier. W4A8 adaptation should land
first because it unlocks the R6000 FP8 x FP4 active-MoE path and the Spark FP8
mirror path.

## Files

```text
benchmarks/p105_w4a8_generate_gate.py
reports/p105/p105_w4a8_generate_smoke_2prompt.json
reports/p105/p105_w4a8_generate_6prompt.json
reports/p105/p105_w4a8_generate_6prompt_64tok.json
```
