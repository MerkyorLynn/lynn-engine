# Lynn Engine P11 Packed-Resident Memory Plan (2026-05-15)

P10 proved the 27B NVFP4 runtime can reach the 100 TPS class on R6000 when the
decode path uses the right native FP4 + graph contract. P11 starts the memory
side of the same story: the artifact is ~20 GiB packed, but the current runner
still keeps BF16 shadows for safe prefill.

## Ground Truth

Model:

`/root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final`

Runtime env:

```bash
LYNN_MOE_IMPL=packed_nvfp4
LYNN_PACKED_DECODE=1
LYNN_PACKED_DECODE_BACKEND=native_fast_2d
LYNN_PACKED_DECODE_PREPARE_NATIVE=1
LYNN_LINEAR_ATTN_INPROJ_FUSED_NATIVE_FP4=1
LYNN_NATIVE_FP4_LM_HEAD=1
```

Read-only gate:

```bash
benchmarks/p11_decode_shadow_release_report.py \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final
```

## Current Memory

After load:

| Metric | GiB |
|---|---:|
| CUDA allocated | 80.85 |
| CUDA reserved | 81.51 |
| Max allocated | 81.32 |

This confirms the known caveat: Lynn-native NVFP4 is packed on disk, but the
current resident runner still carries BF16 shadows.

## Decode Shadow Release Candidates

The runner has packed decode aliases for 270 tensors, representing **56.47 GiB**
of BF16 shadows that could be released in a decode-only/session-scoped lifecycle.

| Bucket | Count | BF16 Shadow GiB |
|---|---:|---:|
| `moe.experts.gate_up` | 40 | 36.05 |
| `moe.experts.down` | 40 | 18.03 |
| `linear_attn.in_proj` | 120 | 1.41 |
| `linear_attn.out_proj` | 30 | 0.47 |
| `full_attn.qkv_proj` | 30 | 0.35 |
| `full_attn.o_proj` | 10 | 0.16 |

## Interpretation

The real packed-memory prize is MoE expert ownership, not attention projection
ownership. Projection shadows matter for cleanliness, but only reclaim ~2.39
GiB. Routed expert shadows reclaim ~54.08 GiB.

## Safety Boundary

Do not globally delete BF16 shadows in the default multi-request server yet.
The current prefill path still uses BF16 tensors. A permanent deletion would
make the first decode path look good and then break the next request.

Safe next steps:

1. Add an explicit session-scoped mode:
   - prefill with BF16 shadows,
   - switch to packed decode,
   - release BF16 expert/projection shadows for that session,
   - reject new requests or reload shadows on demand.
2. Build a packed prefill path for MoE/linear attention so shadows are never
   needed for normal serving.
3. Only after either path passes smoke + tool-call + longctx should packed-only
   resident mode become a production option.

## Target

Short term:

```text
80.85 GiB current resident
  - 54.08 GiB MoE expert BF16 shadow
  - 2.39 GiB projection BF16 shadow
  + packed/native scale overhead
≈ 25-35 GiB realistic packed-resident target
```

That is the path toward simultaneously beating framework memory footprint and
serving latency.

## P11-B Session-Scoped Release Smoke

We added `LynnIncrementalRunner.release_decode_bf16_shadows()` and verified the
first safe lifecycle:

```text
load BF16-shadow resident runner
prefill prompt
snapshot prefill state
run baseline decode
restore prefill state
release decode-covered BF16 shadows
run packed decode
compare greedy ids
```

Required env is recorded and enforced by the smoke:

```bash
LYNN_MOE_IMPL=packed_nvfp4
LYNN_LINEAR_ATTN_RECURRENT_INPLACE=1
LYNN_NATIVE_FP4_LM_HEAD=1
```

R6000 result:

| Metric | Result |
|---|---:|
| Released tensors | 270 |
| Released BF16 shadow | 56.47 GiB |
| Allocated before release | 81.06 GiB |
| Allocated after release | 24.59 GiB |
| Reserved before release | 81.53 GiB |
| Reserved after release | 25.44 GiB |
| Greedy ids after release | exact match |
| Verdict | PASS |

This confirms the packed-memory thesis in a real runner lifecycle. The default
HTTP server should still stay on the non-release path until we add either:

- packed prefill, or
- a single-session mode that refuses/reloads subsequent prefill requests.
