# Lynn Engine — FP8 9B Spark Revival Progress (2026-06-01)

**Goal:** a Spark-native NVFP4-efficient inference engine (model-agnostic). R6000
(sm_120a, native FP4 MMA) and A100 leases ended (5/24, 5/19) → **DGX Spark GB10
sm_121 is the only GPU**. sm_121 has **no FP4 MMA** (ptxas rejects f8f6f4) but
usable **FP8 E4M3 MMA** → the path is: store NVFP4, execute **W4A8 via FP8 MMA**.
Test vehicle: Qwen3.5-9B dense (`Qwen3.5-9B-lynn-native-w4a8-fp8`).

## Validated this session (measured on Spark)

| # | Finding | Data |
|---|---|---|
| M1 | FP8 9B decode floor (conservative cfg) | **~15 TPS**; FP8 +9% over W4A16 at M=1; output coherent (numerically sound) |
| M1 | Per-token budget | 67 ms = **~33 ms memory** (FP8 = 8-bit, ~9 GB/tok) + **~34 ms Python dispatch** |
| M2 | Env-only CUDA graph | **No help.** w4a16+linear_block_graph 13.7; FP8 eager 14.9; **FP8+full_token_graph_slot 6.3 (2.4× slower)** — the slot is position-bound → re-captures 40 layers every token |
| M4 | MTP K=1 batched (dense FP8) | **Works, beats baseline at high accept: 16.9 TPS @ 0.90** (14.4 @ 0.76, 10.1 @ 0.45) vs 15.3 baseline |
| M4 | MTP K≥2 batched | **Broken: accept=0** (all drafts rejected → 5–8 TPS). Dense K≥2 block-verify drift — separate fix |

**Reference bars (llama.cpp on same Spark, 9B Q4_K_M):** 36.8 TPS AR / 60.95 TPS
MTP n_max=4 @ 64% accept. (4-bit reads ~5.3 GB/tok vs FP8's ~9 GB/tok.)

### The key conclusion
FP8 at M=1 has a **hard memory ceiling ~30 TPS < llama.cpp 36.8** — it cannot win
on single-token decode. **MTP (M>1) is mandatory** (amortizes weight reads +
engages FP8 MMA), and **the ~34 ms Python dispatch must be removed**. Both levers
were proven necessary, not optional.

## Shipped (engine was 35B-MoE-only; now dense-capable)
- `engine/mtp_sidecar.py` — `has_embedded_mtp` + `load_mtp_embedded` (load the
  embedded `mtp.*` head from the model dir, dequant FP8→BF16 since proj weights
  are quant-only and attention reads BF16 via `F.linear`); dense-aware
  `mtp_layer_weights` / `mtp_layer_config` / `mtp_layer_forward`.
- `engine/resident_runner.py` — wire embedded-MTP load (gated by
  `LYNN_MTP_SPECULATIVE`/`SHADOW`/`EMBEDDED`).
- `engine/full_forward.py` — dense-FFN branch in `_decode_layer_k2` +
  `_decode_layer_block` (were MoE-hardcoded → KeyError `mlp.gate.weight` on dense).
- `engine/incremental_decode.py` — **M3 fixed-shape full-attn path**
  (`LYNN_FULL_ATTN_FIXED_SHAPE=1`): `index_copy_` KV write + full-cache
  position-masked SDPA, so no tensor shape depends on `cached_seq_len` (prereq
  for capture-once/replay-many decode graph). Parity vs variable-slice: **coherent but not bit-exact** — 1/3 prompts
  token-exact (code 128/128), others diverge at token 7/52 (SDPA picks a
  different backend under the bool mask; not corruption). Fixed-shape eager 13.4
  TPS (full-window cost). Decisive test = graphed-replay speedup, quality after.
- Probes: `scripts/spark_9b_fp8_mtp_k1_probe.py`, `scripts/spark_9b_fixed_attn_parity.py`.

## The lever — M3: full-decode CUDA-graph replay (NOT a full C++ rewrite)
Linear-attn blocks already graph-replay (`_get_reusable_linear_block_graphs`,
fixed-shape recurrent state). Only **full-attn layers stay eager** (KV slice grows
with seq_len). Fix = fixed-max KV + position-buffer + masked attention (standard
vLLM/TRT-LLM decode-graph technique) → entire decode (linear + full-attn +
lm_head) replays from one captured graph → kills the ~34 ms dispatch → target
**2–3× → pass llama.cpp 36.8**. Then MTP (K≥2 fixed) compounds.

Step 1 (fixed-shape attn, bit-exact) → Step 2 (capture-once/replay-many in
`generate()`) → Step 3 (measure). A native C++ decode loop is a later optimization.

## Next
1. M3 graph capture + replay (the dispatch-killer).
2. Fix K≥2 batched MTP accept=0.
3. Full 500/198 W4A8-vs-W4A16 quality regression (canonical).
4. Migrate the proven runtime to 35B-A3B MoE.
