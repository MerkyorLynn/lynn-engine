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

## P52-A/P52-B probes

P52-A tested the tempting bridge: keep router/down unchanged, replace only
selected gate/up with `torch._scaled_mm` native FP4.

```text
reports/p16_155/p52_native_gateup_active_moe_sensitivity.json

mean Triton active:        0.0581 ms
mean native-active hot:    0.1176 ms
mean native-active cold:   0.1957 ms
min gate/up cosine:        0.9720
min active cosine:         0.9763
max active rel_l2:         0.2208
```

Result: **do not compose `_scaled_mm` as the production active expert path**.
It is slower for this selected-expert shape and fails even relaxed quality.

P52-B decomposed the error:

```text
reports/p16_155/p52b_native_fp4_error_decomposition.json

activation FP4 QDQ vs Triton:              min cosine 0.9925
activation FP4 + FP8 weight-scale QDQ:     min cosine 0.9720
native `_scaled_mm` vs Triton:             min cosine 0.9720
native vs activation+FP8-scale simulation: min cosine 0.9956
```

Interpretation: the dominant loss is **not** mysterious tensor-core
accumulation. It appears when Lynn's FP32 per-16 weight scale contract is
compressed into the FP8 `scale_b` layout required by the generic
`torch._scaled_mm` path. The activation FP4 quantization alone is noisy but not
catastrophic; the scale contract is the blocker.

## Updated P52 decision

There are now only two serious routes:

1. **Lynn-native per-16 grouped kernel**: custom CUDA/CUTLASS/CuTe kernel that
   consumes Lynn's existing `uint8 E2M1 + fp32 per-16 scale` layout directly.
   This preserves the current NVFP4 artifact and is the cleanest route to
   exact/near-exact 155 TPS.
2. **Offline re-layout/re-calibration track**: produce a second artifact whose
   weights/scales are native-kernel-friendly (for example e8m0/group32-like).
   This may unlock vendor-style kernels faster, but it is a new quantization
   format and requires full V8/V9/tool-call/long-context retention gates.

Do **not** spend more time trying to make plain `_scaled_mm` selected-expert
composition the runtime path. It has now failed both speed and scale-contract
quality.

## Relation to NVIDIA ModelOpt NVFP4 checkpoints

NVIDIA's public
[`nvidia/Qwen3.5-397B-A17B-NVFP4`](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
checkpoint confirms the high-level strategy: Blackwell + NVFP4 + MoE linear
operators is a real vendor-supported serving path.  The model card states that
the checkpoint is produced with Model Optimizer, targets SGLang/vLLM on
Blackwell, and quantizes weights and activations of MoE transformer-block
linear operators.

That does **not** mean Lynn can drop in the same runtime kernel unchanged.  The
vendor path is optimized for its ModelOpt FP4 scale/layout contract.  Lynn 27B
currently preserves a different quality contract: grouped E2M1 packed weights
with FP32 per-16 scales recorded in `lynn_quant_manifest.json`.  P52-B shows
that compressing those scales into generic FP8 `scale_b` is the quality cliff.

Therefore the vendor checkpoint is best treated as proof that the destination
is valid, not as proof that our current artifact can use the vendor kernel
directly.  A compatibility track would require a new offline re-layout /
re-calibration artifact and full retention gates; the Lynn-native track keeps
the current artifact and writes a per-16 grouped kernel.
