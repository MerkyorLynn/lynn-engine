# Lynn Engine P62: gate/up tile-inter first-divergence probe

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P62

P58/P61 proved:

```text
Triton + graph-off       exact, slow
cuda_tile_inter + graph-off  greedy mismatch
```

P62 narrows the remaining question: where does the gate/up tile-inter candidate
first diverge under a shared reference token stream?

## Probe

```bash
python benchmarks/p62_gateup_tile_first_divergence.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --out reports/p16_155/p62_gateup_tile_first_divergence.json
```

The probe:

1. disables linear-block graphs;
2. runs Triton gate/up/down as reference;
3. runs `cuda_tile_inter` gate/up + Triton down as candidate;
4. feeds both paths the same Triton/reference greedy token stream;
5. records first top-1 divergence and first layer whose hidden cosine drops
   below threshold.

## Expected use

P62 is not a speed benchmark.  It is a microscope for the next grouped per-16
kernel:

- if divergence happens immediately at the first active MoE layer, the gate/up
  accumulation contract is too loose;
- if hidden stays close but logits flip late, promotion gates need a wider
  margin or exact accumulation;
- if a layer type dominates, P63 can start with that layer family.

The output report is the handoff into P63 kernel design.

## R6000 result

Report:

```text
reports/p16_155/p62_gateup_tile_first_divergence.json
```

Summary:

```text
pass: false
first hidden divergence:
  step: 0
  layer: 4
  layer_type: linear_attention
  cosine: 0.99999857
  max_abs: 0.00024414
  rel_l2: 0.001655

first top-1 divergence:
  step: 5
  Triton top1: 1380
  cuda_tile_inter top1: 220
  Triton margin: 0.03125
  cuda_tile_inter margin: 0.0234375
```

Interpretation:

P62 does **not** show a catastrophic kernel bug. It shows tiny early hidden
drift that accumulates until a low-margin token flips. That is exactly why a
local `max_abs=0`-looking gate/up microbench is insufficient for promotion.

P63 should therefore preserve the current Triton-style accumulation/rounding
contract more carefully.  Chasing a faster scalar schedule without exact
decode-state gates will keep rediscovering the same failure.
