# Lynn Engine P52: grouped native-FP4 active expert contract

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Why P52

P16-P51 closed the shortcuts:

- block-size retunes and dispatch cleanup moved the exact path into the
  118-124 TPS band;
- `torch._scaled_mm` cross-expert wrappers overcompute and drift;
- fused atomic CUDA is slower than Triton;
- tile-hidden non-atomic down is locally faster but flips greedy in full decode;
- top-k limiting / skipping shared expert does not reach 155 TPS and degrades
  output quality before it gets close.

The remaining performance target is therefore the real kernel:

```text
active routed expert FFN:
  hidden[2048] bf16/fp4
  expert_ids[top_k=8]
  routing_weights[top_k=8]
  gate_up_packed[experts, 1024, 1024] uint8 E2M1
  gate_up_scale[experts, 1024, 128] fp32
  down_packed[experts, 2048, 256] uint8 E2M1
  down_scale[experts, 2048, 32] fp32
  -> active_out[2048] bf16/fp16
```

## Non-goals

P52 is not MTP/spec decode. Spec decode is a later serving multiplier; it does
not fix the base kernel efficiency and current Lynn 27B artifacts do not ship a
dedicated draft/MTP head.

P52 is not another quality-risking expert budget cut. P51 showed that
approximation does not reach 155 TPS and output breaks early.

## Two viable tracks

### Track A: exact-owned serving

Keep the current Triton active MoE math and reduce orchestration/graph overhead.

Promotion gate:

- greedy IDs match baseline across representative prompts;
- V8/tool-call smoke and V9/coding spike remain healthy;
- OpenAI server path gains, not only isolated replay.

Expected role: safe production incremental gains.

### Track B: grouped native-FP4 expert FFN

Replace the active expert math itself with a grouped/block-diagonal native-FP4
kernel. This can be CUTLASS/CuTe or custom CUDA, but it must express the
selected experts directly instead of computing top-k cross products and keeping
only the diagonal.

Promotion gate:

- first as research backend with explicit quality gates, not exact-greedy by
  assumption;
- then either restore exactness or prove quality retention on V8/V9/tool-call /
  long-context eval;
- no default promotion from microbench numbers alone.

Expected role: real 155+ TPS line.

## First implementation target

The most practical first target is a full active-expert FFN contract probe that
keeps the existing output order:

1. router/top-k unchanged;
2. native grouped gate/up computes `inter[top_k, 512]`;
3. native grouped down consumes the same `inter` and routing weights;
4. compare full MoE active output against current Triton active output on true
   decode states;
5. only after that wire full-generate gates.

P48 already contributed the down-side non-atomic tile kernel. P52 should avoid
promoting it alone; instead use its lessons when designing the full grouped
kernel so accumulation order and full-decode quality are validated from the
beginning.

## Current north star

```text
short-term stable:   118-124 TPS exact/replay band
next milestone:      130+ TPS with exact-owned serving or safe grouped kernel
target milestone:    155 TPS quality-gated grouped native-FP4 active experts
long target:         200+ TPS with broader native FP4 + serving optimizations
```
