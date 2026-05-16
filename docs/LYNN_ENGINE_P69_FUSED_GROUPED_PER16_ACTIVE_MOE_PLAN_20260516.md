# Lynn Engine P69: fused grouped per-16 active-MoE plan

Date: 2026-05-16
Branch: `codex/p16-r6000-155-tps`

## Context

P66-P68 moved the grouped per-16 active expert route from a vague target into a
measurable native ABI:

| Phase | Signal | Decision |
|---|---:|---|
| P66 down scalar reference | 0.856x vs Triton | ABI only |
| P67 down tile reference | 1.266x vs Triton down | fast sub-kernel, no runtime promote |
| P68 active tile reference | 1.108x vs Triton active | complete reference candidate, no runtime promote |

P68 is the most useful measurement because it composes gate/up and down. It
also shows why another two-stage scalar wrapper is not enough: down-only wins
1.266x, but the full active route drops to 1.108x once gate/up, launch overhead,
and the intermediate tensor are included.

## What P69 Targets

The next target is a **single active-MoE boundary**:

```text
hidden[2048]
  + expert_ids[top_k=8]
  + routing_weights[top_k=8]
  + Lynn per-16 packed gate/up/down weights
  -> hidden[2048]
```

This boundary should eventually be implemented by a custom CUDA/CUTLASS/CuTe
kernel family. The kernel must preserve Lynn's per-16 scale contract instead of
forcing the artifact into NVIDIA/Triton e8m0 group32 scaling.

## Why Not Just Use The 200TPS Nemotron Result

The Nemotron screenshot is a valuable north star, but it is not the same
runtime problem:

| Item | Nemotron result | Lynn native path |
|---|---|---|
| Artifact | Q4_K_M / GGUF-style path | Lynn-native NVFP4 per-16 artifact |
| Runtime | llama.cpp C++ tight loop + Blackwell native FP4 | Python/Triton/native extension today |
| Benchmark | aggregate N=1..8 throughput table | strict Lynn 27B decode / graph / OpenAI gates |
| Kernel contract | llama.cpp/ggml quant contract | Lynn per-16 FP32 scale contract |

The 200TPS result says the hardware has headroom and that production C++/CUDA
loops matter. It does **not** solve Lynn's current active expert kernel, because
P18 showed that simple e8m0/group32 bridges fail quality.

The strategic split remains:

1. **Public production route**: GGUF/Q4_K_M + llama.cpp is worth pursuing for
   end-user speed where model architecture support exists.
2. **Lynn-native route**: grouped per-16 active expert FFN is the hard engine
   moat and the path to native NVFP4 serving.

## Acceptance Gates

No future active-MoE kernel should be promoted by microbench alone. Promotion
requires:

1. **Sub-kernel gate**
   - representative layers: 2, 8, 14, 20, 28, 36;
   - min cosine vs current Triton active path >= 0.999999;
   - max relative L2 <= 0.01;
   - active-MoE boundary speedup >= 1.25x before full-runtime testing.
2. **Full-generate gate**
   - representative prompt suite;
   - no `<think>` / `!` / repetition fail pattern;
   - greedy IDs match for exact-safe candidates, or explicit quality gates
     replace exact-match if accumulation order intentionally changes.
3. **Quality gate**
   - 6-prompt coherent smoke;
   - strict tool-call;
   - V8/V9 retention;
   - long-context smoke.
4. **Server gate**
   - OpenAI server decode TPS;
   - no memory regression;
   - no graph-capture regression.

## P69 Immediate Engineering Step

Before writing the tensor-core kernel, keep one machine-readable gate:

- freeze the `active_moe_grouped_per16_*` extension ABI;
- record P66/P67/P68 as baseline measurements;
- require any future candidate to write a JSON report with the same fields:
  latency, cosine, relative L2, runtime promotion decision, and quality gate
  status.

The first gate is:

```bash
python benchmarks/p69_grouped_kernel_acceptance_gate.py \
  --report reports/p16_155/p68_grouped_per16_active_tile_reference_probe.json \
  --out reports/p16_155/p69_acceptance_on_p68_active_tile.json
```

Result:

| Check | Value |
|---|---:|
| P68 sub-kernel contract | pass |
| observed speedup | 1.108x |
| required speedup | 1.25x |
| min cosine | 0.99999988 |
| max relative L2 | 4.98e-4 |
| P69 acceptance | fail |

This is the desired outcome for P68: it stays as kernel signal only. The next
candidate must beat 1.25x at the active-MoE boundary before full-generate or
server promotion work is worth burning GPU time.

This prevents the project from repeating the P48/P56 pattern where a locally
fast tile shape looked tempting but failed full-generate correctness.

## Expected Impact

P16 showed the upper bound:

| Path | Replay TPS |
|---|---:|
| current correct path | 107.13 |
| skip active routed experts | 173.84 |
| skip active + shared | 208.78 |

The active routed expert implementation is therefore the real 155TPS blocker.
P68's 1.108x two-stage reference is not enough. A fused grouped per-16 kernel
needs to remove intermediate materialization and reduce launch/scheduling
overhead at the active-MoE boundary.

If the fused kernel reaches a conservative 1.4-1.6x active-MoE speedup while
holding quality gates, the 155TPS target becomes plausible without relying on
MTP/spec decode.
