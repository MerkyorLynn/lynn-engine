# Lynn Engine Active Research Status

Date: 2026-05-27 Asia/Shanghai

Update 2026-05-28: Spark production fallback #2 is now the llama.cpp
APEX-MTP I-Balanced route with `--spec-type draft-mtp --spec-draft-n-max 4`.
It reaches **77.01 wall TPS** in the short A/B run and **76.19 median wall TPS**
in a live sanity check. Existing 32K thinking-on quality anchors are
**MMLU500 90.00%**, **GPQA198 78.79% naive / 83.87% excl parse fail**, and
**tool-call 12/15**. A fresh serialized 32K quality refresh is running on Spark
under `/home/merkyor/eval/reports/apex_quality32k_20260528_121431`.

This note updates the 2026-05-20 product pivot. The pivot is still true:
Lynn client should use llama.cpp/GGUF as the default local inference backend
until Lynn engine clears the same-model same-hardware speed bar.

It does **not** mean Lynn engine is abandoned. The active engine track is now
narrower and more concrete:

```text
Qwen3.6-35B-A3B / Qwen3.5-9B
  + Lynn APEX-MTP draft sidecar
  + Nemotron-style verify -> accept -> crop/repair runtime
  + T=1-equivalent K=N verifier kernels
```

## Current TL;DR

| Item | Status |
|---|---|
| Nemotron 8B as a ship model | Rejected for Lynn product use; useful as algorithm reference only. |
| Nemotron algorithm control flow | Portable. The useful pieces are verify/accept/crop, full-accept fast commit, prefix repair, and K=N verifier structure. |
| Qwen35 APEX-MTP K=2 runtime | Token-exact on Spark W4A16 + official MTP sidecar with conservative row-wise bridge. |
| Throughput today | Not yet a win in Lynn Python runner. K2 verifier cost still dominates. |
| Main blocker | Batched full-attention verifier numerics: PyTorch batched attention and batched `o_proj` are not T=1-equivalent. |
| Active next work | Build T=1-equivalent dual-row attention and `o_proj` kernels, then re-run real 35B maintenance smoke. |
| Production guardrail | Brain V2 MIMO stays first; llama.cpp APEX-MTP `n_max=4` I-Balanced service stays restored as second fallback after experiments. |

## What Nemotron Contributed

The Nemotron-Labs-Diffusion-8B work separated two things that are easy to
confuse:

1. The **model-specific draft source**: bidirectional diffusion/self-spec draft
   requires joint AR-diffusion training plus a linear-spec LoRA. This does not
   transfer directly to Qwen/Llama causal checkpoints.
2. The **runtime skeleton**: draft, verify, accept, crop/repair, and move the
   decode state forward. This is model-agnostic and directly useful for Lynn
   APEX-MTP.

The practical conclusion is:

```text
Do not copy Nemotron's bidirectional draft path into Qwen base weights.
Do copy the runtime structure into Lynn's existing APEX-MTP path.
```

Nemotron 8B itself also failed Lynn product gates in SGLang chat/eval mode:
tool calling did not produce usable `tool_calls`, MCQ eval emitted repeated
`</think>` spam, and the attractive raw TPS numbers were inflated by repetitive
generation. That does not invalidate the algorithm extraction; it confirms this
should be a Lynn-native APEX-MTP effort, not a Nemotron model switch.

## What Is Already In Lynn Engine

The active branch is:

```text
codex/qwen35-mtp-block-verify
```

Implemented or prototyped:

| Area | Result |
|---|---|
| K=N speculative skeleton | `decode_block_to_logits_and_hidden` and `speculative_step_kn_batched` can verify multi-token draft blocks. |
| Reject/crop semantics | Conservative restore-and-replay path protects token exactness for partial rejects. |
| Full-accept fast commit | K=2 fully accepted blocks can keep verifier state directly. |
| Prefix repair | K=2 one-token partial accept can repair the committed prefix with a smaller exact block. |
| Safe full-attention bridge | `LYNN_FULL_ATTN_K2_BACKEND=rowwise_gate_bridge` keeps K2 token-exact by row-wise QKV/RoPE, attention, and `o_proj`, while batching only the safe gate multiply. |
| Row-wise `o_proj` prototype | `triton_kernels/rowwise_linear.py` proves one Triton launch can match two T1 row-wise launches bit-for-bit under the same backend. |
| Row-wise prefix attention prototype | `triton_kernels/rowwise_attention.py` proves K2 prefix attention can keep independent T1-equivalent online-softmax accumulators in one launch. |
| Safe maintenance wrapper | `scripts/spark_run_qwen35_mtp_maintenance_smoke.sh` masks/stops/restores APEX service around heavy 35B experiments. |

Primary reports:

```text
reports/mtp/QWEN35_MTP_BLOCK_VERIFY_OVERNIGHT_20260526.md
reports/mtp/qwen35_mtp_safe_k2_bridge_20260526_230739.json
reports/mtp/qwen35_mtp_kn_full_accept_fast_20260526_233834.json
reports/mtp/qwen35_mtp_k2_prefix_repair_20260526_235627.json
reports/mtp/qwen35_k2_rowwise_linear_kernel_20260527_020517.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_20260527_023349.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_stride_20260527_023632.json
reports/mtp/qwen35_k2_rowwise_attention_kernel_dynamicn_20260527_025345.json
reports/mtp/qwen35_mtp_k2_rowwise_gate_real2_20260527_021806.json
reports/mtp/qwen35_mtp_k2_rowwise_attention_kernel_warm_20260527_024429.json
reports/mtp/qwen35_mtp_k2_rowwise_attention_dynamicn_warm_20260527_025502.json
reports/mtp/qwen35_mtp_k2_profile_20260527_093355.json
reports/mtp/qwen35_mtp_k2_moe_batched_optimized_20260527_094126.json
```

## Spark Evidence So Far

| Experiment | Exactness | Key Number | Meaning |
|---|---:|---:|---|
| Strict K2 bridge | 6/6 | 16.27 TPS | Correct but too much replay/row-wise cost. |
| K2 full-accept fast commit | 6/6 | 24.36 TPS | +49.5% over strict K2; direct state commit works for K=2. |
| K2 prefix repair | 6/6 | 24.79 TPS | Partial-accept repair is exact; speed gain is small but direction is right. |
| Manual GQA fast path | 5/6 | 25.90 TPS | Approximate fast path breaks parity and is still below baseline. |
| Row-wise `o_proj` kernel probe | 8/8 | 0.18 ms vs 0.75 ms | One dual-row Triton launch can replace two T1 launches under the same backend. |
| Row-wise prefix-attention kernel probe | 8/8 | 0.077 ms vs 0.123 ms | K2 prefix attention can be exact and about 1.6x faster than two T1 launches in the micro fixture. |
| Dynamic-N prefix-attention probe | 8/8 | 0.115 ms vs 0.162 ms | Runtime `N` removes per-seq-len specialization risk, at the cost of a slower micro-kernel. |
| Real 35B rowwise-gate smoke | 6/6 | K2 21.28 TPS | Correct end-to-end, but verifier attention remains too expensive. |
| Real 35B rowwise-attention smoke | 6/6 | K2 22.99 TPS, 0.83x warmed baseline | Attention kernel improves the exact K2 bridge by about 8%, but still does not beat the warmed T1 baseline. |
| Real 35B dynamic-N smoke | 6/6 | K2 23.00 TPS, 0.62x warmed baseline | Current code path is exact and avoids baseline length-JIT artifacts, but K2 remains below baseline. |
| MTP profiler smoke | 2/2 | K2 23.79 TPS, profile-sync run | Highest cost is block/layer verification; K1-batched shows K2 MoE compatibility is a major local bottleneck. |
| Batched optimized MoE shortcut | fails | K1-batched exact gate false | BF16 optimized batched MoE is not a safe shortcut; a real packed NVFP4 T=2 exact MoE kernel is required. |
| llama.cpp APEX-MTP service loop | live | single 72.76 wall TPS, 60.99% accept; 4-way 81.28 aggregate TPS, 60.95% accept | Production fallback already gets useful APEX-MTP accept rate; service-loop work should focus on draft overhead and request-level A/B. |
| llama.cpp request-level `n_max` A/B | live | single MTP n=4 77.01 TPS vs AR 60.65 TPS; 4-way MTP n=4 85.62 TPS vs AR 124.80 TPS | MTP helps single-stream by ~27%, but hurts current 4-slot concurrency; next production policy is dynamic MTP admission. |

Current Python runner baseline on this short smoke is about 31-34 TPS, while the
active llama.cpp APEX-MTP fallback has been observed around 66.7 tok/s with
draft acceptance around 0.47. So the Lynn Python K2 path is a correctness and
research result today, not a production speed win yet.

The 2026-05-27 warmed rowwise-attention smoke makes this sharper. Under the
same experimental rowwise attention/`o_proj` contract, K2 stayed 6/6 exact and
rose from 21.28 TPS to 22.99 TPS, but the warmed baseline was 27.82 TPS. The
kernel direction is useful; it has not yet crossed the production speed bar.

The current committed kernel uses runtime `N` rather than specializing on
sequence length. This removes a misleading per-prompt JIT artifact and gives a
cleaner comparison: warmed baseline 37.03 TPS, K2 23.00 TPS, both 6/6 exact.
So the active branch is now correctness-clean and less benchmark-fragile, but
the K2 verifier still costs too much.

The next ROI pass added an opt-in profiler:

```text
LYNN_MTP_PROFILE=1
LYNN_MTP_PROFILE_SYNC=1
```

The 2-prompt profile smoke showed:

- `spec_k2_batched` is dominated by generic block verification:
  `block_verify.layers_total` averaged about 70.8 ms per event. This path is a
  3-token verifier (`pending + 2 drafts`) and still falls back to mostly T1
  full-attention work.
- `spec_k1_batched` is the path that actually exercises `_decode_layer_k2`.
  There, the biggest steady cost is not the rowwise attention micro-kernel; it
  is the per-token MoE compatibility bridge. The packed NVFP4 decode MoE is
  T=1-only, so K2 currently calls MoE once per token.
- A diagnostic shortcut,
  `LYNN_MTP_K2_MOE_MODE=batched_optimized`, failed the exactness gate and was
  slower. So BF16 optimized batched MoE cannot be promoted.

This changes the next implementation target. The highest-ROI engine work is no
longer QKV/RoPE or attention. It is either:

1. A packed NVFP4 T=2 MoE verifier kernel that preserves the T1 accumulation
   contract, or
2. Moving the already-correct verify/accept/crop flow into the production
   llama.cpp/APEX service loop where Python dispatch is gone.

The first Path A service-loop benchmark is now committed as:

```text
benchmarks/llamacpp_apex_mtp_service_bench.py
reports/mtp/LLAMA_CPP_APEX_MTP_SERVICE_LOOP_PLAN_20260527.md
reports/mtp/llamacpp_apex_mtp_service_bench_20260527_1219.json
```

It runs against the already-loaded Spark service over HTTP and does not start a
second 35B model. The 2026-05-27 run showed:

| Mode | Requests | Completion tokens | Wall TPS | Server TPS | Draft accept |
|---|---:|---:|---:|---:|---:|
| single | 3 | 384 | 72.76 | 80.30 median | 60.99% |
| 4-way concurrent | 4 | 512 | 81.28 aggregate | 22.77 median/request | 60.95% |

This confirms APEX-MTP is already active and useful in the production fallback
loop. It also shows the next bottleneck: concurrency scaling is weak, and
`draft-mtp` still drafts autoregressively inside `common/speculative.cpp`.
Request-level `speculative.n_max` knobs are present in server parsing but
disabled under `#if 0`, so the next small llama.cpp patch should re-enable only
`n_max` for safe same-service A/B.

That A/B is now complete:

```text
reports/mtp/LLAMA_CPP_APEX_MTP_SERVICE_AB_20260528.md
reports/mtp/service_ab_20260528_115837/summary.json
reports/mtp/llama_cpp_apex_mtp_request_nmax_20260528.patch
```

Result: `n_max=4` remains the best single-stream setting and beats AR by about
27% on the short Spark run. But under 4-way concurrency, AR wins by about 46%.
So the immediate production change is not lowering draft depth; it is dynamic
MTP admission: use `n_max=4` for single/low-queue requests and clamp to `n_max=0`
when multiple slots are active.

## Why It Still Matters

The runtime result changes the problem from "can this algorithm work on Qwen?"
to a sharper engineering target:

```text
Make the K2/K4 verifier as cheap as the accept rate deserves.
```

That is exactly the kind of engine problem Lynn can own. The APEX sidecar
already gives a real trained draft source. The extracted Nemotron runtime tells
us how to spend accepted draft tokens safely. The remaining gap is kernel and
service-loop cost, not conceptual feasibility.

## Current Blocker

Fast K2 parity was split into components:

| Component | Finding |
|---|---|
| Batched gate multiply | Safe. |
| Batched `o_proj` matmul | Not T=1-equivalent enough for deterministic verifier parity. |
| Batched attention / SDPA | Not T=1-equivalent, and in the tested GQA prefix shape was not faster either. |
| Row-wise bridge | Exact, but too slow. |

Therefore the next kernels are not "generic faster matmul" work. They need to
preserve the T=1 accumulation contract while sharing launch/weight-load cost
across two rows:

1. Dual-row prefix attention with independent online-softmax accumulators.
2. Dual-row `o_proj` with independent row accumulators.
3. Both enabled for baseline T1 and K2 verifier when checking token exactness.

## Training Direction

For Qwen-family diffusion/self-spec training, the conclusion is separate:

| Route | Recommendation |
|---|---|
| Qwen3.5 9B continued AR-diffusion pretrain | Technically correct, but requires real A100/H100 budget. |
| Qwen3.5 4B LoRA-only diffusion PoC | Good cheap gate to estimate accept ceiling before renting larger hardware. |
| Qwen3.6-35B-A3B MoE diffusion training | Park for now; MoE plus bidirectional diffusion has too many first-dollar unknowns. |
| Lynn APEX-MTP K expansion | Mainline now; it uses an already-trained draft source and the new runtime skeleton. |

If the runtime kernels become cheap and K=2/K=4 accept is still insufficient,
then the right next spend is draft-side training, not another verifier rewrite.

## Next 72 Hours

1. Do not pursue BF16 optimized batched MoE; it failed exactness.
2. Decide between two remaining high-ROI paths:
   packed NVFP4 T=2 MoE exact kernel inside Lynn Python, or port the correct
   K2 runtime into llama.cpp/APEX.
3. If staying in Python, start with a tiny MoE parity fixture before touching
   full 35B: one layer, two tokens, packed T1x2 reference vs candidate T2
   packed kernel.
4. Keep production restored after every smoke:

```bash
systemctl is-active lynn-apex-mtp-llamacpp.service
```

## Product Framing

Lynn engine remains an R&D engine, but it is not idle. The current active
framing is:

```text
llama.cpp ships product fallback today.
Lynn engine develops the next self-spec / APEX-MTP acceleration path.
```

That is a narrower promise than the pre-5/20 engine plan, and a stronger one.
It keeps the product safe while preserving the path toward a Lynn-owned speed
advantage on Qwen3.6-35B-A3B and Qwen3.5-9B.
