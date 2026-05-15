# Lynn Engine P7 R6000 Progress - 2026-05-15

## Context

Target hardware is RTX PRO 6000 Blackwell 96GB. The short-term serving target is
50 TPS, mid-term target is 100 TPS, long-term target is native NVFP4/FP4 above
200 TPS.

The current model under active engine work is:

```text
/root/autodl-tmp/models/lynn-27b-variable-skeleton-v0-nvfp4
```

This is the Lynn-native variable-expert NVFP4 artifact, not the public
compressed-tensors v8-RTN variant.

## Completed Gates

### P7-E: Linear Block Graph Reuse

Change:

```text
LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
```

Effect:

```text
First request: captures 10 linear-attn block CUDA graphs.
Subsequent requests: capture cost = 0.0s, graph slot reused.
```

Measured:

```text
8-token repeated request:
  before reuse first request: ~4.3 request TPS
  after reuse subsequent requests: ~11.2 request TPS

32-token request:
  before reuse: ~27.1 request TPS
  after reuse:  ~29.8 request TPS
```

Result: PASS. Reuse is now opt-in and does not alter default behavior.

### P7-F: RMSNormGated Triton Dispatch

Change:

```text
LYNN_RMSNORM_GATED_BACKEND=triton
```

Measured on real generate path:

```text
decode TPS: ~65 -> ~68.3 TPS
32-token request TPS: ~29.6-30.5
128-token request TPS: ~37.0
```

Result: PASS. This is a real generate-path gain, not only a microbench gain.

### P7-G: Pair Q/K Norm+RoPE Kernel

Change:

```text
LYNN_QK_NORM_ROPE_BACKEND=triton_pair
```

Result:

```text
Correctness: real generate path did not fail.
Performance: steady-state TPS roughly unchanged (~68 TPS).
```

Conclusion: Keep as opt-in. It is not the next main speed lever.

### P7-H: Linear Graph Prewarm

Change:

```text
LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
```

Effect:

```text
Graph capture moved from first user request into runner warmup.
First 8-token request improved from ~4.3 TPS to ~8.2 TPS.
```

Result: PASS.

### P7-I: Prefill Warmup

Change:

```text
LYNN_PREFILL_WARMUP=1
```

Effect:

```text
Tiny dummy prefill runs during runner warmup to compile/cache prefill kernels.
First 8-token request improved again to ~10.6 TPS.
Subsequent 8-token requests remain ~11.1 TPS.
32-token request remains ~30.2 TPS.
```

Result: PASS.

## Current Best Serving Env

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYNN_PREFILL_WARMUP=1
export LYNN_LINEAR_ATTN_RECURRENT_BACKEND=triton_fused_prepare
export LYNN_MOE_IMPL=triton
export LYNN_QK_NORM_ROPE_BACKEND=triton
export LYNN_RMSNORM_GATED_BACKEND=triton
export LYNN_LINEAR_ATTN_INPROJ_FUSED=1
export LYNN_LINEAR_BLOCK_GRAPH=1
export LYNN_LINEAR_BLOCK_GRAPH_REUSE=1
export LYNN_LINEAR_BLOCK_GRAPH_PREWARM=1
export LYNN_LINEAR_STATE_UPDATE=inplace
```

Measured smoke:

```text
6 prompt coherence smoke: PASS
decode TPS per prompt: ~66.6-68.7 TPS
graph capture per request: 0.0s
prefill per request: ~0.60-0.64s after warmup
```

## Ceiling And Bottlenecks

Full-token CUDA graph ceiling with current kernels:

```text
eager full token: 29.84ms = 33.5 TPS
CUDA graph full token: 12.70ms = 78.7 TPS
```

Re-run confirmation after P8 changes:

```text
eager full token:      30.32ms = 32.98 TPS
CUDA graph full token: 12.68ms = 78.85 TPS
```

This is stable and reproducible. It is a ceiling probe, not the current default
server loop, but it is much more trustworthy than the less stable
`torch.compile` 81 TPS experiment.

This means scheduling/graphing alone does not reach 100 TPS. The next 100 TPS
work must attack kernel internals and resident memory layout.

Current full-attn layer profile, layer 31:

```text
layer.full_recomposed:       1.152ms
moe.active_expert_loop:      0.405ms
attn.qk_norm:                0.125ms
attn.rope_apply:             0.116ms
moe.shared_expert:           0.062ms
input/post RMSNorm:          ~0.061ms each
```

The 30 linear-attention layers are the biggest eager-path cost:

```text
linear-attn sum eager: ~21.65ms
full-attn sum eager:   ~7.76ms
```

But graphing the linear-attn blocks has already removed most of that cost from
the serving decode loop. The remaining route to 100 TPS is:

1. keep graph slot reuse and warmups for productized serving;
2. improve full-attn/MoE hot path;
3. introduce packed-resident NVFP4 to remove slow-dequant BF16 resident memory;
4. only then revisit native FP4 kernels for larger fused blocks.

## Memory Direction

The current runtime still slow-dequants Lynn-native NVFP4 into BF16 resident
weights. The manifest math says:

```text
Original BF16-equivalent quantized matrices: ~57.5 GiB
Packed NVFP4 payload:                       ~18.0 GiB
Potential resident release:                 ~35-40 GiB
```

Native/packed resident therefore matters for both memory and future speed. It
is also the real path toward the long-term >200 TPS target.

## Artifact Transfer

A100 final step5000 artifacts:

```text
BF16 final:  /mnt/data4/lynn-27b-work/merged/lynn-27b-variable-recovery-step5000-bf16-final
NVFP4 final: /mnt/data5/lynn-27b-variable-recovery-step5000-nvfp4-final
```

Validation:

```text
BF16 structural validation: PASS
BF16 greedy 2-token sanity: PASS
NVFP4 paired tensor validation: PASS, failures=0
NVFP4 greedy 2-token sanity: PASS, generated "MoE"
```

Transfer A100 -> R6000 is running via rsync with resume:

```text
NVFP4 first, then BF16.
Observed speed: ~3.5-4.0 MB/s.
Stable but slow; safe because rsync uses --partial --inplace.
```

## Next Steps

Immediate:

1. Let A100 -> R6000 final artifact transfer continue.
2. Add packed-resident loader path that keeps selected decode matrices as
   `PackedNVFP4Linear` instead of dequantized BF16 tensors.
3. Start with decode-only selected projections, preserving BF16 prefill path.
4. Re-run 6-prompt smoke and P7 request amortization after every opt-in switch.

Do not globally enable native_scaled_mm yet. P7-G showed the current native FP4
single-matmul route is not the main speed lever. Packed resident is the correct
next structural step.

## P8-A/B/C/D Packed Decode Alias Gate

2026-05-15 late afternoon update: implemented decode-only packed aliases behind
opt-in env flags:

```text
LYNN_PACKED_DECODE=1                 # all decode projections
LYNN_PACKED_DECODE_LINEAR_ATTN=1     # linear-attn projections only
LYNN_PACKED_DECODE_FULL_ATTN=1       # full-attn projections only
LYNN_PACKED_DECODE_BACKEND=scalar_bridge|native_fast_2d
```

The key contract is that BF16/dequantized weights remain under the original
`.weight` keys for prefill. Decode may look up `.weight.packed` aliases via
`PackedNVFP4Linear`. This avoids the multi-token prefill failure mode while
letting us validate packed NVFP4 wiring end-to-end.

Results on R6000, 27B variable skeleton:

```text
baseline P7-I decode TPS:                  ~66-68 tok/s
P8-A all packed, scalar_bridge:             23.8 tok/s (4-token)
P8-B all packed, native_fast_2d:            19.5 tok/s (4-token)
P8-C linear-attn packed only, scalar_bridge 23.7 tok/s (4-token)
P8-D full-attn packed only, scalar_bridge   27.4 tok/s (4-token)
```

Verdict:

1. Packed decode alias plumbing is correct enough to generate coherent text.
2. Scalar bridge and current PyTorch `_scaled_mm` fastpath are not production
   speed paths.
3. Packed resident remains a memory-architecture milestone, not a speed win
   until we replace per-projection scalar/native calls with larger fused kernels.
4. The 100 TPS route should not flip these env flags by default. Keep P7-I as
   the current best serving path, and use P8 aliases as the integration harness
   for future fused native kernels.

P8-F also ruled out the final projection as the hidden 100 TPS blocker:

```text
embedding lookup: 0.008ms
final RMSNorm:    0.063ms
lm_head:          0.666ms
argmax:           0.011ms
```

The lm_head is visible but not dominant. The remaining gap from ~68 TPS to
100 TPS sits in the per-layer decode kernels / launch structure, not the final
vocab projection.

## P8-G OpenAI HTTP Smoke

Started the Lynn OpenAI-compatible server with the current best R6000 env and
ran one `/v1/chat/completions` request:

```text
endpoint: http://127.0.0.1:18099/v1/chat/completions
model:    lynn-27b-nvfp4-engine
prompt:   用一句话解释 MoE active parameters
result:   coherent Chinese answer
request tokens/s: 28.86 including ~0.64s prefill
decode TPS:       66.69
graph capture:    0.0s, reused=true
```

The server was stopped after the smoke to free the R6000 GPU. This proves the
current 66-68 decode TPS path is reachable through the OpenAI-compatible entry,
not only benchmark scripts.

## P8-H OpenAI Tool-Call Smoke

Patched `server/openai_http.py` to accept OpenAI-compatible `tools` and parse
Qwen XML tool calls into `message.tool_calls`.

Tool smoke:

```text
prompt: 北京今天天气怎么样？
tool:   get_weather(location: string)
result: tool_calls[0].function.name      = get_weather
        tool_calls[0].function.arguments = {"location": "北京"}
finish_reason: tool_calls
decode TPS: 66.31
```

This closes an important serving gap: Lynn engine can now expose a real
OpenAI-compatible tool-call response on the R6000 path. The server was stopped
after the test to keep the GPU free for ongoing engine work.

## P8-I Torch Compile Full-Token Probe

Ran a one-off `torch.compile(..., mode="reduce-overhead")` probe around the
single-token decode wrapper. This is not enabled in normal serving yet; the goal
was to measure whether PyTorch compiler capture can beat the manual CUDA graph
ceiling from P7-F.

Measured on R6000, 27B variable skeleton:

```text
eager full token:       32.82ms = 30.47 TPS
compiled full token:    12.33ms = 81.10 TPS
speedup:                 2.66x
compiled first call:     53.47s
compile wrapper cost:   616.22ms
verdict: PASS
```

The compiler warned that CUDA graphs were skipped for mutated inputs:

```text
skipping cudagraphs due to mutated inputs
state.conv_state[layer_idx].copy_(new_conv)
```

Interpretation:

1. `torch.compile` is a valid resident-service experiment and slightly beats the
   current manual full-token graph ceiling (~78.7 TPS -> ~81.1 TPS).
2. The first-call compile cost is large, so this is not a cold-start path.
3. This still does not reach the 100 TPS target. The remaining gap requires
   fused per-layer decode kernels / packed-resident native kernels, not only
   scheduler or compiler wrapping.
4. Keep the P7-I env as the default stable serving path. Treat P8-I as an
   opt-in benchmark branch until it is integrated into server warmup with
   deterministic cache behavior.

Follow-up script hardening:

```text
Inductor cudagraph trees ON:  saw 81.10 TPS once, but can trip CUDA allocator
                              assertions with mutable Triton state.
Inductor cudagraph trees OFF: stable script run, but compiled path regressed to
                              35.30ms = 28.33 TPS.
```

So P8-I is a useful ceiling signal, not a production switch. The stable serving
branch remains P7-I, and the stable graph ceiling remains the manual CUDA graph
probe (~78.7 TPS) until full-token graph replay is integrated deliberately.

## P8-J Recurrent State In-Place Update Probe

Implemented an opt-in recurrent state update path:

```text
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
```

The Triton recurrent gated-delta kernel can now write `s_new` directly back
into the resident recurrent state tensor. `_decode_layer` skips the redundant
`copy_` when the returned tensor already aliases the resident state.

Measured on R6000 with the current best serving env plus the new flag:

```text
6-prompt coherence smoke: PASS
decode TPS range:          67.18 - 68.81 tok/s

8-token request #1:        10.76 request TPS, 62.55 decode TPS
8-token request #2:        11.35 request TPS, 68.69 decode TPS
32-token request:          30.45 request TPS, 68.72 decode TPS
```

Verdict:

1. Correctness passes; the in-place recurrent state path is safe as an opt-in.
2. It does not move the throughput needle under the graph-reuse serving path.
3. This rules out recurrent state copy-back as the main 68 -> 100 TPS blocker.
   The remaining work is still fused per-layer decode kernels / packed-resident
   native kernels rather than state-copy cleanup.
