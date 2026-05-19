# Qwen3.6 / Qwen3.5 Closed Speed Levers Guardrail

Date: 2026-05-19  
Scope: speed levers closed today or not worth repeating without a materially
different boundary. This is a guardrail for future R6000 serving work.

## Summary

The common pattern is clear: small output-buffer, scratch, prepared-wrapper, and
block-shape tweaks are exact but flat, or they preserve local speed while losing
service-level promotion. The next useful work should move to larger exact
boundaries: 35B linear/GDN core, larger MoE/GDN launch boundaries, or offline
layout/repack work. Do not keep spending P37/P25 time on the closed knobs below.

## Closed / Do-Not-Repeat Matrix

| Lever | Status | Key numbers | Closed reason | Replacement direction |
|---|---|---:|---|---|
| Router softmax scratch | closed / flat | 18/18 exact; router mean 0.039006 -> 0.039062 ms | Full router boundary is flat once `F.linear` and promoted top-k out-buffer are included; isolated softmax win is below measurement noise | Only bundle softmax scratch into a larger prepared router wrapper with a P25-visible gain |
| Triton active-MoE prepared boundary | closed for default | P37 exact; P25 512 106.64 TPS; hard structured 40/40 mean 107.41 TPS | Numerically safe, but below current safe default line | Keep Triton math authoritative; remove larger MoE/GDN launch boundaries or change offline layout |
| Triton active-MoE block-shape sweep | closed | 40 configs tested; only current 8x256 gate / 8x512 down, warps 4/8 is 18/18 exact | Near misses are not exact and timing deltas are service-noise scale | Do not retune block sizes; focus on larger exact boundary or layout/repack |
| Shared prepared / shared `mm_out` | closed as promotion | shared `mm_out` 18/18 exact, 0.03489 -> 0.03304 ms/layer; finalize prepared exact but ~0.065 ms vs 0.02977 ms default | Shared body saves only ~0.00185 ms/layer; prepared finalize variants are much slower | Fold shared work only as part of a larger MoE finalize boundary |
| Native FP4 activation scratch | closed for default | P169 fixtures 20/20 exact; P37 exact; P25 512 107.07 TPS; hard structured 40/40 mean 107.43 TPS | Local in-proj win did not convert into service promotion; below safe default 108/109 bar | Keep as opt-in plumbing for a larger 35B linear-core boundary |
| Recurrent from `out_conv` | closed for default | P169 20/20 exact; P37 exact; P37 speedup 1.0042x; P25 512 107.996 TPS; structured 40/40 mean 107.865 TPS | Fixture boundary improved to 0.309 ms, but q/k/v split elimination alone is too small once linear-block graph/service overhead is included | Fuse a larger boundary such as `conv + gate + recurrent`, or remove allocation/launches beyond q/k/v views |
| Recurrent from `out_conv` + A/B gate | closed / not exact | P176 total 0.286 ms but P169 0/20, max_abs 0.01534, cosine_min 0.999995 | Moving sigmoid/softplus beta/g prep into Triton is faster but violates exactness | Keep PyTorch-produced `beta`/`g` explicit unless the default quality contract changes |
| 9B act-scratch stacked service gate | closed / flat | P175 stack 128/256/512 decode TPS 61.82 / 62.42 / 62.52 vs P173 62.55 at 512 | Adding `LYNN_NATIVE_FP4_ACT_SCRATCH=1` on top of dense gate/up + RoPE cache does not move 9B service TPS | Continue 9B work on larger dense FFN/TensorCore repack or server batching, not act scratch |
| Native packed MoE gate/up replacement | research-only; no resident promotion | P160 partial exact 0/18, max diff 4.768e-7; P161 terms exact; P162 simple trees fail Triton `tl.sum`; P146 resident backends fail P37 | Drift is Triton reduction-tree mismatch, not FP4 decode/scale math. Approximate native output cannot enter exact-first P37/P25 | Keep Triton active-MoE as exact authority; either reproduce Triton lowering deliberately or fuse around Triton boundaries |

## Per-Lever Notes

### Router Softmax Scratch

Candidate: `LYNN_ROUTER_SOFTMAX_OUT_BUFFER=1`.

P164 was exact on all 18 fixtures, but the full router boundary moved from
0.039006 ms to 0.039062 ms. The softmax-only microbench improved by about
0.001920 ms, but that does not survive the surrounding router work.

Guardrail: do not rerun this as a standalone default candidate. It is only
worth revisiting inside a larger router wrapper that already removes a
P25-visible boundary.

### Triton Active-MoE Prepared Boundary

Candidate stack:

```text
LYNN_MOE_ACTIVE_SCRATCH=1
LYNN_MOE_TRITON_PREPARED=1
LYNN_NATIVE_GATEUP_BACKEND=triton_fast_decode
LYNN_NATIVE_DOWN_BACKEND=triton
```

P165 passed exactness gates, but service speed was not enough:

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0068x |
| P25 512 decode TPS | 106.64 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 107.41 |

Guardrail: keep the prepared path as an exact artifact, but do not promote it
or repeat wrapper-only variants. The next useful MoE-side move must remove real
launches or memory traffic.

### Triton Active-MoE Block-Shape Sweep

P166 tested 40 block-shape configurations against the P157 Triton-stage
contract. Only the current default shape was 18/18 exact:

```text
gate block_inter=8, block_hidden=256, warps=4
down block_hidden=8, block_inter=512, warps=8
```

Fast near misses lost exactness, for example 18/18 inter but only 13/18 output
exact with down warps 16. Timing differences were too small to justify further
exactness risk.

Guardrail: block-shape retuning is closed for the current Triton active-MoE
authority.

### Shared Prepared / Shared `mm_out`

P167 showed the shared expert body can use caller-owned BF16 output exactly,
but the service-scale win is too small:

| Path | Exact | Mean ms/layer | Delta |
|---|---:|---:|---:|
| shared default | reference | 0.03489 | - |
| shared `mm_out` | 18/18 | 0.03304 | -0.00185 |
| shared inplace SiLU | 0/18 | 0.03430 | closed |
| finalize default | reference | 0.02977 | - |
| finalize with `mm_out` shared | 18/18 | 0.06270 | +0.03293 |
| finalize prepared/in-place | 18/18 | 0.06515 | +0.03538 |

Guardrail: do not promote shared prepared variants alone. Reuse the finding only
inside a larger MoE finalize boundary.

### Native FP4 Activation Scratch

Candidate: `LYNN_NATIVE_FP4_ACT_SCRATCH=1`.

P170 closed the standalone promotion. It is exact but below the current serving
bar:

| Gate | Result |
|---|---:|
| P169 linear-core fixtures | 20/20 exact |
| P169 max_abs_max | 0.0 |
| P37 exact | true |
| P37 median speedup | 1.0125x |
| P25 512 decode TPS | 107.07 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 107.43 |

The local P168-style census did show a small in-proj win:

| Segment | Default | Scratch | Delta |
|---|---:|---:|---:|
| fused native FP4 in-proj sum | 2.107 ms/token | 1.906 ms/token | -0.201 ms |
| full linear core | 8.993 ms/token | 8.799 ms/token | -0.194 ms |

Guardrail: keep as opt-in plumbing. Do not promote or rerun standalone; use it
only as part of the larger P168/P169 linear-core focus.

### Recurrent From `out_conv`

Candidate: `LYNN_LINEAR_ATTN_RECURRENT_FROM_OUTCONV=1`.

P175 added an exact Triton recurrent/GDN path that reads q/k/v directly from
`out_conv` instead of materializing q/k/v views in Python. Fixture-level results
were clean:

| Gate | Result |
|---|---:|
| P169 boundary fixtures | 20/20 exact |
| P169 max_abs_max | 0.0 |
| P172 diagnostics hash match | 20/20 for `core_attn_out`, `conv_state_out`, `recurrent_state_out` |
| Fixture total | 0.309 ms vs P173 serving-like 0.336 ms |

Resident serving remained below promotion:

| Gate | Result |
|---|---:|
| P37 exact | true |
| P37 median speedup | 1.0042x |
| P25 512 decode TPS | 107.996 |
| hard structured | 40/40 |
| hard structured mean decode TPS | 107.865 |

Guardrail: keep the opt-in path for future larger fused-boundary experiments,
but do not promote or rerun it alone. It proves q/k/v view removal is exact but
too small.

### Recurrent From `out_conv` + A/B Gate

P176 moved beta/g computation into the Triton recurrent kernel:

```text
out_conv + a_raw + b_raw + A/dt -> recurrent/GDN
```

It was faster at the fixture boundary but not exact:

| Metric | Result |
|---|---:|
| P169 passed | 0/20 |
| max_abs_max | 0.01534 |
| cosine_min | 0.999995 |
| fixture total | 0.286 ms |

Guardrail: do not escalate this candidate to resident serving. The current
default-quality contract requires PyTorch-equivalent sigmoid/softplus behavior,
so `beta` and `g` should remain explicit inputs unless a future gate explicitly
allows relaxed math.

### 9B Act-Scratch Stacked Service Gate

P175 stacked act scratch on top of the already-clean 9B P173 stack:

```text
LYNN_DENSE_FFN_GATE_UP_FUSED=1
LYNN_FULL_ATTN_ROPE_CACHE=1
LYNN_FULL_ATTN_ROPE_CACHE_MAX_SEQ=65536
LYNN_NATIVE_FP4_ACT_SCRATCH=1
linear block graph reuse/prewarm
```

Result:

| Max tokens | Decode TPS |
|---:|---:|
| 128 | 61.82 |
| 256 | 62.42 |
| 512 | 62.52 |

This is flat versus P173's 62.55 TPS at 512 tokens.

Guardrail: do not spend more 9B service-gate time on act scratch stacking. The
9B path needs larger dense FFN/TensorCore repack work or server batching.

### Native Packed MoE Reduction-Tree Drift

P160-P162 close the native packed MoE exactness mystery:

| Probe | Key result |
|---|---|
| P160 partial trace | partial exact 0/18; max partial diff 4.768e-7; native partial -> P147 inter max 2.441e-4 |
| P161 term trace | 256 FP32 products are bit-exact; term max abs 0.0 |
| P162 tree search | native tree identified as pairwise halving; simple FP32 trees do not reproduce Triton `tl.sum` |

This means the drift is not FP4 decode, scale/global arithmetic, or BF16 input
conversion. It is the exact 256-term FP32 reduction tree used by Triton.

Guardrail: no resident P37/P25 escalation for approximate native packed MoE
gate/up. Either reproduce Triton's lowering deliberately, or keep Triton as the
exact active-MoE authority and optimize around it.

## Replacement Focus

Use P168 as the main 35B next-step compass:

1. Build and keep a fixture-style linear-core contract for representative
   linear-attention layers.
2. Target `in_proj -> conv -> recurrent/GDN` as one larger exact boundary.
3. Use native FP4 act scratch only inside that larger boundary, not as a
   standalone service candidate.
4. Keep all candidates behind local exactness, P37, P25, and hard structured
   gates before any default discussion.

## Source Anchors

- `reports/qwen36_35b/P164_ROUTER_SOFTMAX_BOUNDARY_20260519.md`
- `reports/qwen36_35b/P165_TRITON_PREPARED_ACTIVE_MOE_BOUNDARY_20260519.md`
- `reports/qwen36_35b/P166_TRITON_MOE_BLOCK_SWEEP_20260519.md`
- `reports/qwen36_35b/P167_SHARED_EXPERT_PREPARED_PROBE_20260519.md`
- `reports/qwen36_35b/P170_NATIVE_FP4_ACT_SCRATCH_CLOSED_20260519.md`
- `reports/qwen36_35b/P175_RECURRENT_FROM_OUTCONV_CANDIDATE_20260519.md`
- `reports/qwen36_35b/P176_RECURRENT_FROM_OUTCONV_AB_CANDIDATE_20260519.md`
- `reports/qwen35_9b/P175_ACT_SCRATCH_STACKED_SERVICE_GATE_20260519.md`
- `reports/qwen36_35b/P160_NATIVE_MOE_GATEUP_PARTIAL_TRACE_20260519.md`
- `reports/qwen36_35b/P161_NATIVE_MOE_GATEUP_TERM_TRACE_20260519.md`
- `reports/qwen36_35b/P162_NATIVE_MOE_REDUCTION_TREE_SEARCH_20260519.md`
- `reports/qwen36_35b/P168_LINEAR_CORE_SEGMENT_CENSUS_20260519.md`
