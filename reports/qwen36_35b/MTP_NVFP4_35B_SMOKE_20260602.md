# MTP K≥2 on NVFP4 35B-A3B with the trained sidecar — measured (2026-06-02)

**Milestone:** task #11 moves from paper-direction to **measured**. The trained
35B MTP sidecar drafts *well* on Lynn's NVFP4 W4A16 path and the verify is
token-exact — but effective TPS is a **slowdown**, and this run pinpoints the
exact kernel that blocks the speedup.

## Setup
- Model: `Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-bf16-20260526`
  (`nvfp4_e2m1_rowwise_per_16`, Lynn-native packed). NVFP4 fast path
  (`packed_nvfp4` + `native_fast_2d` + native FP4 lm_head), resident BF16 shadow.
- Sidecar: `mtp_sidecars/qwen36-35b-a3b-mtp/mtp.safetensors` (1.6 GB, the trained
  MoE NEXTN head; same head llama.cpp's APEX `draft-mtp` uses).
- Tool: `scripts/spark_mtp_speculative_smoke.py --spec-k-list 2,4 --max-new 128`,
  6 configs on one prompt set, single load. Raw: `mtp_nvfp4_smoke.json` / `.log`.
- Spark GB10 sm_121, APEX stopped during the run for memory headroom (NVFP4 35B
  resident ≈ 90 GB with the BF16 shadow), restarted after.

## Results
| config | accept | eff. TPS | vs baseline | token-exact vs baseline |
|---|---|---:|---:|:--:|
| `baseline` (NVFP4 W4A16 prod) | — | **36.10** | 1.00× | — |
| `spec_k1` (P118 sequential verify) | 80.4% | 30.96 | 0.858× | ✅ **True** |
| `spec_k1_batched` | 79.0% | 29.98 | 0.830× | ✗ (batched-K1 path bug) |
| `spec_k2_batched` | **65.0%** draft (429/672) | 16.18 | 0.448× | ✅ **True** |
| `spec_k4_batched` | 44.0% draft (481/1129) | 13.42 | 0.372× | ✗ |

## Finding 1 — the lever is REAL on Lynn (not just llama.cpp)
- `spec_k1` **80.4% accept**, `spec_k2_batched` **65.0% draft-accept** — the latter
  **matches llama.cpp APEX's 63%** on the identical 35B. The trained sidecar
  produces high-quality drafts through Lynn's NVFP4 numerics.
- `spec_k1` (sequential) and `spec_k2_batched` are **token-exact vs baseline** —
  the K=1 sequential verify and the K=2 block verify are wire-correct. (The
  `spec_k1_batched` and `spec_k4_batched` paths are NOT exact yet — separate
  block-verify bugs, not the sidecar.)

So MTP head quality **and** verify correctness are both proven on Lynn. This was
the open question; it's now answered yes.

## Finding 2 — why it's still a slowdown (the exact blocker)
Effective TPS is **below baseline** at every K (0.86× → 0.45× → 0.37×), and it
degrades monotonically with K. Cause: the **NVFP4 MoE verify is per-position
T=1** (the long-standing *T=1-only kernel contract* — `packed_nvfp4` /
`native_fast_2d` hardcode `shape[0]==1`). The K+1 batched-verify positions
therefore re-run the MoE **K+1 times** instead of amortizing it. At M=1 decode
the MoE is memory-bound, so each verify position **re-reads every active
expert's weights** — K+1× the weight traffic. MTP adds forwards without
amortizing them, so more speculation = slower (k4 < k2 < k1), exactly as
observed.

llama.cpp wins on the same sidecar (63% accept → real speedup) precisely because
its MoE verify is genuinely batched — it reads each expert once for all the
verify positions.

## The unlock (precisely scoped)
**A batched T≥2 NVFP4 MoE verify kernel.** Group the K+1 verify positions'
expert GEMMs so each active expert's packed weights are read **once** for all
positions routing to it (grouped/segmented MoE over the M=K+1 rows), instead of
the current per-position T=1 loop. This is the NVFP4 analogue of P2's grouped
GEMM, applied to the *verify* path. With 65% draft-accept already in hand, a
batched verify that costs ~1 MoE pass (not K+1) flips spec_k2 from 0.45× to a
net win → the path to ~60 TPS.

Secondary: fix the `spec_k1_batched` / `spec_k4_batched` block-verify exactness
bugs (k1-seq and k2-batched are already exact, so the verifier core is sound).

## Status
- task #11: **lever proven + verify correct + bottleneck pinpointed + next kernel
  scoped.** Not yet a TPS win — that needs the batched T≥2 NVFP4 MoE verify.
- This supersedes the earlier "graph +10%" lever: graph doesn't help (dispatch
  isn't the bottleneck); the batched-verify MoE kernel is the real lever for MTP.

## Next (high-ROI, next focused session)
1. Batched T≥2 NVFP4 MoE verify kernel (grouped over K+1 positions) — the win.
2. Re-run this smoke; target `spec_k2_batched` > 36.1 → toward llama.cpp 69.77.
3. Fix `spec_k1_batched` / `spec_k4_batched` exactness, then sweep K for the knee.

## Implementation anchors (turnkey for the next session)
The diagnosis is corroborated by the code's own comments — the build is scoped:
- **`engine/full_forward.py:797-828`** (`_decode_layer_k2`): the MoE-K=2 path. The
  per-position T=1 loop at **lines 823-828** (`base_moe_fn(h_norm[:, t:t+1, :])`
  per `t`, then `torch.cat`) is the deliberate slowdown — "trades a tiny
  per-position launch for backend consistency". Replace with a grouped T≥2 call.
- The comment at **797-808** already names the plan: *"before writing a real
  packed T=2 kernel"*; a `LYNN_MTP_K2_MOE_MODE=batched_optimized` hook (line 815)
  exists but calls the **BF16** `optimized` path once → numerically drifts from
  the packed_nvfp4 production numerics → accept collapsed (11% vs 77%). So the
  real kernel must stay in **packed_nvfp4 numerics**, just batched.
- **The kernel to extend:** the `packed_nvfp4` fused gate/up Triton kernel
  **hard-codes `h.shape[1]==1`** (the T=1-only contract). Extend it to take M
  rows per expert (group the K+1 verify rows by expert, read each expert's
  packed weight once, GEMM all its rows) — the NVFP4 analogue of P2's grouped
  GEMM, on the verify path.
- **`_decode_layer_block`** (`full_forward.py:832`) is the K>2 counterpart and
  carries the same per-position issue + the `spec_k4_batched` exactness bug.
- **Validate** token-exact vs the current per-position path (already a passing
  gate for k2), then re-run `scripts/spark_mtp_speculative_smoke.py`.

## Session checkpoint (2026-06-02)
Banked + pushed: lever proven (80%/65% accept = llama.cpp parity), verify
correct (k1-seq + k2-batched token-exact), blocker pinpointed and code-confirmed,
next kernel scoped with anchors above. APEX (:18098) restarted as 3rd-priority
fallback. The Triton T≥2 kernel is deferred to a fresh session (clean context →
safe to build + validate + commit without risking a broken packed path).
