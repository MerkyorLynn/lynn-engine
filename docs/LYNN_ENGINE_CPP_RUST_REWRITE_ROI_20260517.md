# Lynn Engine C++/Rust Rewrite ROI

Date: 2026-05-17

## Decision

Long-term C++/Rust is feasible, but the ROI is sharply staged:

```text
Do not start a full Rust rewrite now.
Phase A: CUDA kernel + MTP integration inside the current Python oracle.
Phase B: C++ decode/service hot path only after Phase A exposes Python overhead.
Phase C: Rust rewrite only if product/commercial/runtime strategy needs it.
```

The reason is simple: Lynn's current miss to 155 TPS is not only Python
overhead. P112/P113 show the expensive MTP part is the one-layer MTP decoder
and active MoE execution. A language rewrite without a better native verify
and MoE contract will spend months and still miss the target.

This supersedes the earlier "Rust server soon after native core" framing. The
server shell is not the scarce part yet. The scarce part is a quality-preserving
native decode core that can cash out W4A8 and MTP.

## External Evidence

Atlas is the strongest current validation for the direction.

Primary references:

- Atlas site: <https://atlasinference.io/>
- Atlas repo: <https://github.com/Avarok-Cybersecurity/atlas>
- Atlas NVIDIA forum thread: <https://forums.developer.nvidia.com/t/atlas-open-source-inference-engine-for-dgx-spark-2minute-cold-start-100-tok-s-on-qwen3-6-35b-fp8-13-supported-models/369263>
- Chimere repo: <https://github.com/AIdevsmartdata/chimere>
- TriAttention paper: <https://arxiv.org/abs/2604.04921>
- TriAttention repo: <https://github.com/WeianMao/triattention>
- Lynn MTP verify ABI: <./LYNN_ENGINE_MTP_VERIFY_ABI_20260517.md>

Atlas claims and source code line up with a Rust + CUDA architecture:

| Evidence | Meaning for Lynn |
|---|---|
| Atlas publishes Qwen3.5-35B-A3B MTP K=2 at about 130 tok/s peak / 111 tok/s average on GB10 | Lynn's 155 TPS target is aggressive but not fantasy for a smaller 27B-A3B class model if MTP and kernels cash out. |
| Atlas hot path is Rust + CUDA, no Python/PyTorch | The framework ceiling is real once kernels and state ownership are mature. |
| Current Atlas source has MTP bootstrap plus K=2/K=3/K=4 verify steps | The useful speculative path is batched verify with state commit/rollback, not a serial "sidecar then normal decode" loop. |
| Atlas exposes MTP quantization policy: NVFP4, FP8, BF16 | Lynn should make MTP precision and label source a serving policy, not a hidden training artifact. |
| Atlas has request/scheduler awareness for grammar and speculative boundaries | Lynn's new format-guard MTP disable knob is aligned with the right product behavior. |
| Atlas implements HSS / NVMe KV swap through an io_uring path | Useful for Lynn long-context serving later; not a short-decode 155 TPS lever. |
| Atlas has Turbo3/4/8 KV formats using Walsh-Hadamard rotation and Lloyd-Max codebooks | Worth tracking for long context and KV quality; separate from Lynn W4A8 weight/activation path. |
| Atlas is AGPL-3.0-only | Use clean-room architecture lessons only. Do not copy code into Lynn without an explicit license decision. |

TriAttention is useful, but it is not the next 155 TPS lever. It targets long
reasoning KV-cache compression and reports 2.5x throughput or 10.7x KV-memory
reduction on long outputs. Lynn's immediate bottleneck is short/medium decode,
MTP acceptance, and active-MoE native runtime. TriAttention belongs in a later
long-context workstream.

Chimere is a useful adjacent reference, not a replacement route. It is
Apache-2.0 and Rust-facing, but its production path is built on a customized
`ik_llama.cpp` backend and GGUF/RAMP quantization, with MTP infrastructure still
gated in the current README. The actionable ideas are:

- entropy-based routing should be considered for future MTP enable/disable
  policy;
- Engram-style n-gram logit bias is a product feature idea, not a 155 TPS
  runtime fix;
- its 16 GB consumer-card framing validates that specialized single-model
  runtimes can beat generic stacks, but it does not support Lynn's variable
  expert NVFP4/W4A8 artifact directly.

## Product Decisions

The project decision is:

```text
1. Continue Lynn-specific W4A8 + variable-expert 27B-A3B.
2. Continue self-owned MTP/NEXTN sidecar training.
3. Do not switch to standard Qwen3.5-35B-A3B as the main product just because
   Atlas benchmarks it well.
4. Use Atlas as an architecture benchmark, not as code or as the model target.
```

Rationale:

| Decision | Why |
|---|---|
| Keep W4A8 | It is Lynn's differentiator and unlocks R6000 FP8 x FP4 / Spark FP8 mirror work. Atlas does not solve this path. |
| Keep variable experts | Lynn 27B's custom pruning/distill is the product IP. Standard Qwen3.5-35B is easier to benchmark but easier for Atlas/llama.cpp to commoditize. |
| Keep MTP self-training | Atlas proves MTP K=2 is a real production multiplier, but it does not provide a Lynn-compatible head. |
| Add C++/CUDA core before Rust server | The 155 miss lives in native verify, active-MoE, and state ownership. Rust API polish comes after the core moves. |
| Treat Atlas benchmark reproduction as optional | Running Atlas 35B on Spark is useful for external baseline confidence, but it should not consume the main A100/R6000 loop or delay Lynn core work. |

Atlas also exposes a competitive warning: Lynn's current Spark/R6000 runtime is
behind a purpose-built engine on raw tok/s. The answer is not to abandon Lynn's
custom model path; it is to carry the custom path into an equally owned runtime.

## Lynn Reality

Current repo shape:

```text
Python files: 261
CUDA files:   2
C++ files:    1
Rust files:   0
```

The existing native path is still an opt-in PyTorch extension under
`csrc/lynn_native/`. The hot path remains Python/Torch/Triton with some native
CUDA bridges. That is good for fast research, but it limits the next production
step in three ways:

1. The token loop, state moves, and graph lifecycle are not owned by a native
   runtime.
2. K=2/K=3 speculative verify cannot be made cheap enough with serial Python
   orchestration.
3. PyTorch CUDA graph slots have already shown state-portability limits in the
   P9U/P9V/P9W branch.

Confirmed bottleneck facts:

| Probe | Result | Rewrite implication |
|---|---|---|
| P112 MTP component profile | v34 draft median 7.47 ms; MTP decoder layer 6.64 ms | MTP layer/runtime, not lm_head, is the expensive part. |
| P113 slot-sorted MTP layer | exact path cuts draft to about 2.24 ms | Good first cut, still above the P115 budget for 155. |
| P115 continuous-credit sim | top1 depth2 needs <=1.65 ms per iteration; top8 depth2 needs near-constant cost | Need batched verify/overlap, not serial recursion. |
| P9U/P9V/P9W graph branch | PyTorch graph state is not portable across fresh requests | Need a native/static state ABI before relying on whole-decode graphs. |
| P50 down tile | speed improves, quality drifts | Native kernels must be quality-gated, not promoted by TPS alone. |

## ROI By Layer

Tier-level estimate from today's evidence:

| Tier | Scope | Timebox | Expected TPS effect from 49 tok/s Spark-class baseline | ROI |
|---|---|---:|---:|---|
| 0 | Current Python/Triton/Config D class | now | 49 tok/s class | baseline |
| 1 | top hot CUDA kernels, W4A8 quality, MTP integration, runtime policy | 3-6 months, 1-2 engineers | 80-100 tok/s class | highest |
| 2 | C++ decode/service hot path, buffer arena, more native kernels | +6-9 months, 2 engineers | 110-140 tok/s class | medium |
| 3 | full Rust+CUDA Atlas-style rewrite | 18+ months, larger team | 130-180 tok/s class if kernels/MTP already work | lowest near-term |

The main expected gain is Tier 1. Tier 2 is justified only if Phase A shows the
Python/service loop has become the bottleneck. Tier 3 is a product/runtime
strategy, not a near-term speed fix.

| Layer | What changes | Expected TPS ROI | Risk | Recommendation |
|---|---|---:|---|---|
| Python control cleanup | Fewer Python dispatches, better env policy, current Triton kernels | 3-8% | Low | Keep doing only when it directly supports a gate. |
| More Triton-only retuning | Tile/search around current kernels | 0-10% | Medium | Low ROI now; most easy wins have been tried. |
| C++/CUDA kernel island | Native active-MoE, transposed NVFP4 decode layout, MTP layer kernels | 10-25% base, plus lower draft cost | Medium | Highest near-term runtime ROI. |
| C++/CUDA decode core | Native token loop, buffer arena, K=2/K=3 verify, state commit/rollback | 15-35% base/realization, enables MTP multiplier | Medium-high | Required before 155 is credible. |
| Rust server/scheduler | HTTP/API, queueing, grammar/MTP policy, telemetry around native core | Small raw TPS, high production ROI | Medium | Defer until the runtime is a commercial product surface. |
| Full Rust/CUDA rewrite from scratch | Own everything from loader to server | 1.3-2.0x only if kernels and MTP also land | High | Not a 2026-H1 speed plan. |

The full rewrite ROI is high only after the core speed path is known. Otherwise
it risks becoming a slower duplicate of the current Python runner.

## Phase A Targets

The first native work should be five focused replacements, not a full engine:

| Priority | Target | Evidence | Exit gate |
|---|---|---|---|
| 1 | MTP K=2/K=3 verify ABI and MTP decoder layer | P112 says MTP decoder layer is the draft cost; P115 says serial one-token MTP misses 155. | P107/P116 parity, state commit/rollback exact, K=2 cost <= 1.65 ms or overlap hides it. |
| 2 | Variable-expert active-MoE fused boundary | P50/P97 show local speed but quality drift; P69 says one active-MoE boundary is the right ABI. | active-MoE speedup >=1.25x and P105/serving preview stays AMBER/GREEN. |
| 3 | Transposed NVFP4/W4A8 decode layout probe | Atlas uses transposed expert layouts and batch2/3 kernels for verify; Lynn needs clean-room per-16 version. | one-layer parity plus measurable reduction in active-MoE read/launch overhead. |
| 4 | Native/static full-attn layer boundary | P9H/P9I show 4x graph replay, but PyTorch graph slots are not portable across fresh requests. | explicit-state full-attn boundary with strict parity and reusable capture. |
| 5 | Runtime MTP policy surface | P117 says global MTP is unsafe; format guard disable is already proven. | request-level disabled/shadow/allowlist policy with metadata and no structured guard regression. |

Native FP4 lm_head stays a small steady-state lever. It is already useful, but
P112/P26 show it is not where the missing 155 TPS budget lives.

## Recommended Architecture

Stage the native runtime as three layers, but only promote a layer when the
previous one has passed a speed and quality gate:

```text
Python research/oracle
  - quality gates
  - training/calibration
  - report generation
  - differential testing

C++/CUDA decode core
  - resident buffer arena
  - packed NVFP4/W4A8 tensor views
  - active-MoE native kernels
  - MTP propose and K=2/K=3 verify
  - state commit/rollback ABI
  - C ABI callable from Python and Rust

Rust serving shell (deferred)
  - OpenAI/Anthropic compatible API
  - scheduler and request policy
  - grammar/format guard integration
  - observability and crash isolation
```

Why this split:

- C++ is the shortest path for CUDA kernels, tensor pointers, and driver-level
  graph/state work.
- Rust is better for long-lived server code, request scheduling, and policy
  plumbing, but only after the C++/CUDA core is worth packaging as a product.
- Python remains the fastest way to run A100 quality experiments and compare
  against the existing runner.

## Go/No-Go Gates

Do not start a broad Rust server rewrite until these are true:

| Gate | Required result |
|---|---|
| Native K=2 verify ABI | exact greedy parity on P107/P116 traces; state commit/rollback proven. |
| Verify cost gate | K=2 iteration cost <= 1.65 ms on R6000, or overlap hides the excess. |
| Base runtime gate | native decode core reaches >115 tok/s without MTP on 256/512-token serving probes. |
| MTP serving gate | route-allowlisted MTP effective TPS clears 155 while structured guard keeps MTP disabled. |
| Quality gate | W4A8 structured/tool-call generation remains AMBER/GREEN. |
| Spark gate | same core beats local llama.cpp on identical prompt/token budget after warmup. |
| Product gate | Lynn engine needs a lightweight commercial runtime surface, not only a research runner. |
| Team gate | at least several engineers can maintain kernels, server, and model work in parallel. |

If the K=2 verify ABI or cost gate fails, stay in Python+C++ island mode and
keep optimizing kernels. If it passes, the Rust server becomes worth doing.

## Immediate Engineering Plan

1. Keep A100 focused on W4A8 quality/MTP label construction; do not use MTP to
   mask a RED structured/tool generation gate.
2. Write a clean Lynn K=2/K=3 verify ABI spec from the current Python state
   objects: token input, hidden output, recurrent state, conv state, KV state,
   and commit/rollback semantics.
3. Build a C++/CUDA extension prototype for K=2 verify on one sequence, using
   Python tensors as the initial memory owner. This avoids a full loader rewrite
   while proving the state machine.
4. Add a transposed NVFP4 active-MoE decode layout probe. Atlas's source points
   at transposed decode weights as a core runtime pillar; Lynn needs a clean-room
   implementation against its own per-16 manifest format.
5. Move MTP policy from report-only to runtime shape: disabled, shadow-only,
   route-allowlisted, and forced-off during format guard.
6. Only after those pass, design `liblynn_decode_core` as a standalone C ABI
   library. Do not start the Rust server wrapper until the product/team gates
   above are true.

## Expected Timeline

| Timebox | Deliverable | Exit criterion |
|---|---|---|
| 1-2 days | ABI spec plus Python/C++ boundary smoke | one token decode state roundtrip is exact. |
| 1-2 weeks | K=2/K=3 verify prototype | P107/P116 parity and cost report. |
| 1-2 months | native active-MoE/transposed layout prototype | P69/P105/P25 quality gates survive and base TPS rises. |
| 3-6 months | Phase A kernel+MTP stack | W4A8 quality AMBER/GREEN and 80-100 tok/s class serving. |
| 6-12 months | Phase B C++ decode/service hot path if needed | >115 tok/s without MTP or clear evidence Python is no longer the bottleneck. |
| 18+ months | Phase C Rust product runtime, conditional | product/team gates pass; not required for the current 155 push. |

This is a realistic path to a production engine. It is also the fastest way to
learn whether the full rewrite is worth expanding.

## Bottom Line

C++/CUDA is not optional long-term if Lynn wants to beat llama.cpp/SGLang/vLLM
on its own model class. Full Rust is optional and conditional. The next
high-ROI step is Phase A: W4A8 quality, MTP K=2/K=3 verify, and native
active-MoE/layout work under the Python oracle, not a blank-page Rust server.
