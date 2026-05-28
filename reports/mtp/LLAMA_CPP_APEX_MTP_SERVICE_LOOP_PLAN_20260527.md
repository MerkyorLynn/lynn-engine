# llama.cpp APEX-MTP Service Loop Plan (2026-05-27)

## Decision

Use **Path A: move the Nemotron-style runtime learnings into the llama.cpp/APEX service loop** as the next Lynn Engine track.

The Python runner already proved exact K2 control flow, but its current verifier path is slower than baseline. The Spark production service already runs `draft-mtp` inside llama.cpp, so the highest-ROI work is to improve and measure the real service loop instead of building another Python-side runtime.

## Current Source Facts

- Spark llama.cpp already has `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` in `common/speculative.cpp`.
- Server integration is in `tools/server/server-context.cpp`.
- KV rollback/crop is already represented by `common_context_seq_rm(...)` / `llama_memory_seq_rm(...)`.
- Verification already batches `[sampled, draft_0, ..., draft_n]` into the target context and samples/accepts through `common_sampler_sample_and_accept_n(...)`.
- The `draft-mtp` implementation still drafts autoregressively: each draft token calls `llama_decode(ctx_dft, ...)` before the next draft token can be sampled.
- Per-request speculative knobs are parsed in `tools/server/server-task.cpp`, but the adjustment block is currently under `#if 0`, so online A/B of `speculative.n_max=0/2/4` needs a code patch or a second service instance.

## Production Observation

`lynn-apex-mtp-llamacpp.service` is active as Brain v2 fallback #2.

Recent service log sample:

- 457 generated tokens
- 76.20 tok/s
- draft acceptance 0.60714 (323 accepted / 532 generated)
- `draft-mtp` draft time 1452.913 ms across 135 calls

This means APEX-MTP is not abandoned: it is already producing useful accept rate in the production loop. The problem is engineering ROI: draft/verify overhead still limits speedup.

## Work Plan

1. Add safe HTTP benchmark runner for the active service.
   - No new model process.
   - No systemd restart.
   - Capture single-stream TPS, concurrent aggregate TPS, `draft_n`, and `draft_n_accepted`.

2. Patch llama.cpp research branch for request-level speculative knobs.
   - Re-enable a narrow subset: `speculative.n_max`, `speculative.n_min`, `speculative.p_min`.
   - Keep defaults unchanged.
   - Use this to A/B `n_max=0/1/2/4` on the same loaded service.

3. Add MTP runtime counters if the existing response timings are insufficient.
   - Draft calls.
   - Generated draft tokens.
   - Accepted draft tokens.
   - Draft wall time.
   - Rollback count.

4. Gate any production change.
   - Exactness/quality smoke unchanged.
   - Single-stream TPS improves over current service.
   - Concurrent aggregate TPS does not regress.
   - Brain v2 fallback ordering remains unchanged.

## Likely Next Patch

The first actual llama.cpp patch should be small:

- In `tools/server/server-task.cpp`, expose only `speculative.n_max`, `speculative.n_min`, and `speculative.p_min`.
- Do not expose `speculative.type` yet.
- Run a same-service A/B matrix:
  - `n_max=0` (AR equivalent, speculation disabled per request)
  - `n_max=1`
  - `n_max=2`
  - `n_max=4` (current production)

If `n_max=4` wins, the next engineering target is draft-side overhead. If `n_max=1/2` wins, production should lower the default before deeper kernel work.

## Lynn Engine Portability Conclusion

Nemotron's bidirectional diffusion draft source is not directly reusable for Qwen/Llama without training, but its service-loop mechanics are directly reusable:

- batched verification,
- accept/reject accounting,
- KV crop on rejection,
- draft-depth sweeps,
- service-visible speculative metrics.

For Qwen3.6-35B-A3B and Qwen3.5-9B, the practical path is still: use Lynn/APEX draft heads as the draft source, and move the runtime mechanics into the serving loop where CUDA graphs, KV cache, batching, and sampling are already real.
