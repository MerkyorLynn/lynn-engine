# Lynn Engine Phase 4 P3 — Native FP4 Roadmap

Date: 2026-05-14

Reference workload:

- `Lynn-V4-Pro-Distill-Qwen-35B-A3B-BF16-merged`
- `Lynn-V4-Pro-Distill-Qwen-35B-A3B-NVFP4-v8-RTN`
- R6000 / RTX PRO 6000 Blackwell Server Edition / sm_120

## Current State

P2 established independent Lynn Engine execution:

- BF16 prefill/decode works.
- NVFP4 v8-RTN slow-dequant prefill/decode works.
- BF16 and NVFP4 greedy generation match on the minimal prompt.
- BF16 and NVFP4 OpenAI-compatible HTTP smoke both pass.

P3 has now established native packed-weight execution:

- A Triton kernel consumes `weight_packed + weight_scale + weight_global_scale`
  directly.
- QKV and MoE expert gate/up/down representative tensors pass with cosine ~1.0
  versus resident dequantized reference.
- `PackedNVFP4Linear` exposes this as a runtime object.
- `decode_linear_attn` can route selected projections through packed NVFP4.
- Replacing all five linear-attention decode projections passes:
  `cosine=0.999981`, `rel_l2=0.00620`.

## What This Is And Is Not

This is **native packed-weight execution**, not yet final Blackwell tensor-core
FP4 GEMM.

It is still a major engineering unlock:

- the loader no longer has to materialize full BF16 weights for every Linear,
- decode code now has a stable `PackedNVFP4Linear` runtime contract,
- future kernels can replace the implementation behind that contract without
  touching model semantics.

Current performance is slower than resident BF16 for single-token decode because
the scalar bridge launches several small kernels. That is expected.

## Milestones

### P3-E — Batched Projection Probe

Goal:

- Reduce launch overhead for linear-attention decode projections.
- Probe whether QKV/Z/B/A/out can be grouped or scheduled with fewer launches.

Deliverables:

- `benchmarks/p3_nvfp4_projection_batch_probe.py`
- JSON timing report under `/root/autodl-tmp/reports/lynn-engine-p1/`

Pass criteria:

- output cosine >= 0.999 versus existing P3-D path,
- runtime no worse than P3-D,
- clear timing breakdown for each projection.

Status:

- Initial projection timing probe completed.
- Report: `/root/autodl-tmp/reports/lynn-engine-p1/p3_nvfp4_projection_timing_probe.json`.

| Projection | Packed ms | Resident ms | Delta ms |
|---|---:|---:|---:|
| in_proj_qkv | `0.0530` | `0.0259` | `+0.0271` |
| in_proj_z | `0.0390` | `0.0253` | `+0.0137` |
| in_proj_b | `0.0360` | `0.0269` | `+0.0091` |
| in_proj_a | `0.0364` | `0.0267` | `+0.0098` |
| out_proj | `0.0549` | `0.0244` | `+0.0305` |
| total projections | `0.2193` | `0.1292` | `+0.0901` |

Interpretation:

- `in_proj_qkv` and `out_proj` are the largest per-projection deltas.
- `in_proj_a` and `in_proj_b` are tiny 32-output projections; their cost is
  mostly launch overhead and should not remain as separate kernels.
- P3-E next should try either `a+b` fusion or a larger `qkv+z` grouped
  projection probe before attempting full-layer native packed decode.

`a+b` dual projection fusion is now validated:

| Probe | Value |
|---|---:|
| separate `in_proj_a` then `in_proj_b` | `0.1015 ms` |
| fused dual `a+b` kernel | `0.0502 ms` |
| speedup | `2.02x` |
| a output max_abs vs separate | `0.0` |
| b output max_abs vs separate | `0.0` |

Report:

```text
/root/autodl-tmp/reports/lynn-engine-p1/p3_nvfp4_dual_ab_probe.json
```

Conclusion: tiny projections are launch-overhead dominated. Same-shaped packed
matvec fusion is worth generalizing.

### P3-F — Native MoE Expert Probe

Goal:

- Apply `PackedNVFP4Linear` to selected MoE expert gate/up/down projections.
- Confirm router-selected expert decode path can run from packed weights.

Pass criteria:

- expert FFN output cosine >= 0.999 versus resident dequantized expert path,
- top-k expert routing unchanged,
- timing breakdown for gate/up/down.

Status:

- First expert gate/up dual fusion probe completed.
- Report: `/root/autodl-tmp/reports/lynn-engine-p1/p3_nvfp4_dual_expert_gate_up_probe.json`.

| Probe | Value |
|---|---:|
| separate expert gate then up | `0.0802 ms` |
| fused dual expert gate/up | `0.0511 ms` |
| speedup | `1.57x` |
| gate max_abs vs separate | `0.0` |
| up max_abs vs separate | `0.0` |

Next: compute `silu(gate) * up`, then run packed down projection, and compare
the full single-expert FFN output against the resident dequantized expert path.

Single expert FFN packed path is now validated:

| Probe | Value |
|---|---:|
| expert | layer 0 expert 0 |
| packed FFN vs resident cosine | `0.999994755` |
| packed FFN vs resident rel_l2 | `0.00328` |
| resident expert FFN | `0.0650 ms` |
| packed gate/up + packed down FFN | `0.0976 ms` |

Report:

```text
/root/autodl-tmp/reports/lynn-engine-p1/p3_nvfp4_single_expert_ffn_probe.json
```

Conclusion: MoE expert correctness is stable through gate/up/down in packed
form. Performance is still slower because the path uses two scalar kernels
(fused gate/up, then down). The next useful kernel step is a fused expert FFN
or a true FP4 GEMM path for gate/up/down.

### P3-G — Decode Layer With Packed Linear-Attention + Packed MoE

Goal:

- Run one full layer decode where linear-attention projections and at least the
  selected experts use packed NVFP4 runtime objects.

Pass criteria:

- layer output cosine >= 0.999 versus resident dequantized layer,
- recurrent/conv state consistency remains stable,
- no dependency on SGLang/vLLM/TRT/llama.cpp.

### P4/P5 — True FP4 GEMM

Goal:

- Replace scalar unpack-dot with Blackwell-native FP4 GEMM.

Candidate paths:

- CUTLASS/CUDA extension when `nvcc`/toolchain is available,
- PyTorch `_scaled_mm` if it can represent the needed packed FP4 layout,
- flashinfer/cuDNN FP4 path if exposed outside SGLang in a reusable form,
- custom Triton block GEMM if Triton gains usable FP4 lowering for sm_120.

Pass criteria:

- token-for-token parity against P3 slow packed path,
- measurable decode speedup over resident BF16,
- no loss of the P2/P3 correctness gates.

## Non-Negotiables

- Correctness first: no silent fallback, no random-init tolerance.
- Every new kernel gets a JSON report and a documented comparison against the
  resident dequantized reference.
- P2 resident slow path remains intact as the oracle.
- Performance claims must cite measured R6000/Spark reports, not estimates.
