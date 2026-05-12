# Phase 4 — Reference workload + correctness-oracle positioning

> **One-liner**: After 2026-05-13 ships `Lynn-V4-Distill-Qwen-35B-A3B`, Lynn engine finally has its own dogfood model. Phase 4 makes that dogfood the reference workload for P1–P3, defers all native kernel work until correctness is proven, and positions Lynn engine as a **second-source verifier / correctness oracle**, not an SGLang replacement.

## ⭐ Strategic reframe (2026-05-12 night)

**P1+P2 dequant-to-BF16 on R6000 is itself a shippable artifact, not test scaffolding for P4.**

The R6000 sm_120 ecosystem gap is real and 4-8 weeks from closing (memory `feedback_modelopt_fp4_scale_naming_mismatch.md` traps #7-9):

- SGLang stable / nightly `sgl_kernel` PyPI wheel ships **only `sm_90 + sm_100`** — no `sm_120/`
- vLLM 0.20.2 hard-requires `flash-attn` with **no `cu130 + torch 2.11 + py3.12` wheel** on PyPI
- TRT-LLM 1.2.x pinned to transformers 4.57 → `qwen3_5_moe` not recognized → cascade

**Result**: as of 2026-05-12, **no production-grade inference engine can serve Qwen3.6-35B-A3B NVFP4 on R6000 sm_120 Workstation**.

If Lynn engine P1+P2 lands a slow-but-correct dequant-to-BF16 path (NVFP4 packed → BF16 unpack → existing BF16 forward), the comparison is NOT "Lynn vs FP4 native kernel" — the comparison is **"Lynn vs nothing"**. That makes the dequant path itself a deliverable artifact:

- **First inference engine to run `modelopt_fp4` / `compressed-tensors nvfp4` on R6000 sm_120 single card**
- Slow (BF16 forward path, ~10-20x slower than hypothetical FP4 GEMM) but **correct** (verified by N≥20 parity vs SGLang Spark sm_121 production)
- Closes the ecosystem gap for R6000 / 5090 / 5060 Ti / future 27B-A3B pruned single-card deployments (memory `project_lynn_27b_pruning_plan_0509.md`)
- P4 native NVFP4 GEMM becomes **performance optimization**, not a gate for ship

This reframe matters for prioritization. P1+P2 is no longer "boring scaffolding before the real work" — it is the **shortest path to the first deployable artifact**. If we can load + dequant + forward, ship it; native kernel comes when P3 parity is rock solid.

## Strategic positioning (2026-05-13)

Upstream engines (vLLM / SGLang / TRT-LLM) compete on **tokens/sec**.
Lynn engine competes on **never silently wrong**.

These two axes are orthogonal. Lynn engine's pitch:

- **Fail-loud loader** (`engine/loader.py` L4 guard, committed 2026-05-11)
- **N ≥ 20 parity gate** — single-prompt PASS does not endorse
- **Token-for-token reference against production SGLang** — daily silent regression detector
- **No silent fallback** anywhere: format unknown → raise, layer mismatch → log + abort, quant suffix unknown → raise

Lynn engine does NOT try to outrun SGLang. It runs alongside and says "this token disagrees".

## Reference workload: `Lynn-V4-Distill-Qwen-35B-A3B`

Lynn-trained, Lynn-owned, 4 checkpoints public on HF after 2026-05-13 cutover:

| Repo | Format | Role for Lynn engine |
|---|---|---|
| `nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-bf16-merged` (if published) | BF16 merged | **Ground truth** for parity tests |
| `nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-v8-RTN` | compressed-tensors `nvfp4-pack-quantized` | NVFP4 path A |
| `nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-modelopt` | `modelopt_fp4` (HF native) | NVFP4 path B (different schema) |
| `nerkyor/Lynn-V4-Distill-Qwen-35B-A3B-FP8` | `compressed-tensors` FP8 | Compatibility fallback path |

Why this is the right reference:

1. **Same model class as Spark production** (Qwen3.6-35B-A3B MoE on SGLang dev-cu13 nightly) → SGLang nightly is the natural oracle
2. **Lynn team-owned** → not subject to upstream deprecation
3. **4 quant formats of same weights** → cross-format parity testable
4. **Reproducible** → public HF download, anyone can re-run
5. **Already passed 4-Gate eval** → if Lynn engine output diverges from SGLang, the model is not the suspect

## Phase 4 sprint sequencing (boundary: NO kernel work in P1–P3)

### P1 — Canonical tensor spec + fail-loud loader (2026-05-13 → 05-15, ~2 days)

```python
@dataclass
class NVFP4TensorSpec:
    packed_weight_uint8: torch.Tensor    # FP4 packed, 2x 4-bit/byte
    weight_scale_fp8: torch.Tensor       # F8_E4M3 per-group, group_size=16
    global_scale_fp32: torch.Tensor      # F32 per-tensor
    input_scale_fp32: torch.Tensor       # F32 per-tensor
    group_size: int = 16
    source_format: Literal["compressed_tensors", "modelopt_fp4"]
```

**Three input formats normalized into one canonical spec**:

- `BF16 merged` — passthrough (no NVFP4)
- `FP8 (compressed-tensors)` — separate spec (FP8TensorSpec, similar pattern)
- `compressed-tensors nvfp4-pack-quantized` (v8-RTN) — `*.weight_packed` / `*.weight_global_scale` / `*.input_global_scale` per-expert
- `modelopt_fp4` (HF native) — `experts.N.{down,up,gate}_proj.weight + .weight_scale + .weight_scale_2 + .input_scale` per-expert

**Loader fail-loud contract** (already enforced by L4 guard 2026-05-11 commit `6e61d29`):

- Unknown `quant_method` → `NotImplementedError`
- Unknown weight key suffix → `NotImplementedError`
- Missing scale for packed weight → `ValueError`
- All errors carry zhihu postmortem link for downstream debug

**Deliverables**:
- `engine/quant_spec.py` — `NVFP4TensorSpec` + `FP8TensorSpec` dataclasses
- `engine/loader.py` — extend L4 guard to actually parse + normalize when format is supported
- Unit tests: 4 ckpts × `load_and_inspect` → spec dump

### P2 — Dequant-to-BF16 correctness (2026-05-15 → 05-20, ~3-5 days)

**Target: speed irrelevant, correctness mandatory.**

1. Load `Lynn-V4-Distill-Qwen-35B-A3B-NVFP4-v8-RTN` via Lynn engine
2. Normalize each packed weight tensor → `NVFP4TensorSpec`
3. Dequant to BF16 (`packed_uint8` × `weight_scale_fp8` × `global_scale_fp32`)
4. Substitute into model's `Linear` modules at load time
5. Run BF16 forward via existing Lynn engine BF16 path

**Parity gates** (each step has explicit checksum):

| Gate | Check | Pass criterion |
|---|---|---|
| **G2.1 single tensor unpack** | One Linear's packed → fp4 values → BF16, compare to `Lynn-V4-Distill BF16 merged` same Linear's weight | `atol ≤ 1e-3` (single tensor) |
| **G2.2 one linear layer forward** | Same input through dequantized Linear vs BF16 reference Linear | `atol ≤ 1e-3` on output |
| **G2.3 one transformer layer forward** | Layer N forward, including attention + MoE + RoPE, dequant vs BF16 | `atol ≤ 1e-2` (accumulated error) |
| **G2.4 5-token greedy decode** | 5-token greedy from prompt, compare token IDs | exact match for first 5 tokens |

Failure at any G2.X → triage: which layer / which tensor / scale broadcast off by what factor? **No `gate.skip()`.** No `try: ... except: pass`. No silent fallback to BF16 — if dequant path is wrong, we **want** the test to fail loudly.

**Deliverables**:
- `engine/dequant.py` — pure-Python NVFP4 unpack + scale broadcast → BF16
- `benchmarks/reference_workloads/lynn_v4_distill.py` — workload config
- `benchmarks/parity_g2.py` — G2.1 → G2.4 runner

### P3 — N ≥ 20 multi-prompt parity gate (2026-05-20 → 05-27, ~5-7 days)

**Goal**: prove Lynn engine produces **token-identical** output to SGLang dev-cu13 nightly on N ≥ 20 prompts, for BOTH `BF16 merged` and `dequant-from-v8-RTN` paths.

Test composition:
- 10 short prompts (≤ 64 tokens output)
- 5 medium prompts (≤ 256 tokens output)
- 5 long prompts (≤ 1024 tokens output)
- Mix of zh / en, single-turn / multi-turn, with tool-call cases included

Per prompt:
1. Lynn engine greedy decode → token IDs
2. SGLang dev-cu13 nightly greedy decode → token IDs
3. **Token-by-token compare**: any divergence within first 256 tokens fails the prompt

**Pass criteria**:
- `BF16 path`: 20/20 prompts agree token-for-token
- `dequant v8-RTN path`: 18+/20 prompts agree (2 prompts tolerance for FP4 quantization noise at deep layers)
- Any single-prompt divergence > 5 tokens → flagged + saved for investigation

**Daily regression mode**:
After P3 passes once, the same test runs daily as `benchmarks/parity_daily.py`. Output:
```
lynn_engine_parity_YYYYMMDD.json
```
Schema in `benchmarks/parity_schema.py`. Any prompt that flipped from PASS to FAIL since yesterday → alert.

This is Lynn engine's **silent regression detector** product. The single most differentiated artifact in the LLM inference engine space.

### P4 — Native NVFP4 kernel (defer until P3 passes, ~2026-06)

**Not now — and not blocking ship.** Per memory `project_lynn_engine_nvfp4_native_path.md`:

> "P4 native NVFP4 GEMM: 不在 P1-P3 都通之前碰这层 — 同时面对格式 / scale / router / kernel 四层不确定性,会调到怀疑人生。"

After P3 N≥20 parity passes (2026-05-27+), Lynn engine has a **shipping artifact** (the slow dequant path). P4 is performance work *on top of* an already deployable engine:

- expert FFN `Linear` → NVFP4 grouped GEMM
- Triton implementation first (most accessible, sm_120 ABI stable)
- CUTLASS path optional for performance bake-off
- **Target hardware**: R6000 sm_120 single-card (Spark sm_121 already has SGLang nightly, so sm_120 is where Lynn engine has the strongest niche)
- **Success criterion**: kernel path produces identical token IDs to dequant path on N≥20 parity gate, while runtime drops 5-10x

Critical: **P4 must not regress P3 parity.** Same gate runner, same N≥20 test set, same SGLang oracle — kernel passes only if token-for-token identical to dequant baseline (we already proved dequant matches BF16 ground truth in P2). This is why P3 must lock first.

## Hard rules (Phase 4)

1. **No native kernel before P3 passes.** Re-stating because it's the single most common temptation.
2. **No silent fallback to BF16** when NVFP4 fails. Raise.
3. **Single-prompt PASS does not endorse.** N ≥ 20 minimum.
4. **Token-for-token compare**, not embeddings / cosine sim / "vibe check".
5. **Ground truth is published HF ckpt**, not "my local file" — reproducibility is mandatory.
6. **Parity report JSON in `benchmarks/results/` is canonical**. README screenshots / blog claims must match the JSON.
7. **Lynn engine is not an SGLang replacement** in any 5/15 ship doc, blog, or HF model card. It is a verifier.

## Tonight's prep layer (2026-05-12)

- `docs/PHASE4_REFERENCE_WORKLOAD.md` — this doc
- `benchmarks/reference_workloads/lynn_v4_distill.py` — placeholder workload config
- `benchmarks/parity_schema.py` — JSON schema for parity reports

When V Pro Distill is published (2026-05-13 morning), P1 starts.
