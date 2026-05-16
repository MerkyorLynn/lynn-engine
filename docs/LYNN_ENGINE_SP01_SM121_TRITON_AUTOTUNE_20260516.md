# Lynn Engine SP-01 · Spark sm_121 Triton Autotune for MoE Kernels

Date: 2026-05-16
Branch: `spark/sm121-port` (Spark-only; does NOT touch R6000 main line `codex/p16-r6000-155-tps`)

## Goal

Beat SGLang FP8+MTP on Spark sm_121 with Lynn 27B NVFP4 in the same single-stream
and 20-prompt mixed-stability bench harness.

```text
Baseline   Lynn 27B NVFP4 packed   42.85 mean / 44 peak TPS
Baseline   SGLang FP8+MTP 35B      49.97 mean / 62.51 peak TPS
Gap        +17% mean / +42% peak to overtake
Target     50+ mean / 55+ peak (SP-01 alone)
           65+ peak (SP-01 + SP-02 N-gram spec combined)
```

## Why SP-01 First

SGLang's own server logs on Spark sm_121 print:

```text
Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!
Config file not found at .../device_name=NVIDIA_GB10
```

That is the same gap Lynn's `nvfp4_moe.py` has: kernels ship a single hardcoded
`(BLOCK_INTER=8, BLOCK_HIDDEN=64, num_warps=4)` config for gate/up and
`(BLOCK_HIDDEN=8, BLOCK_INTER=256, num_warps=4)` for down. Those values were
chosen for R6000 sm_120 and never re-tuned for sm_121. Triton autotune sweeping
17 candidates per kernel removes that gap with zero math changes.

## What Changed

### `triton_kernels/nvfp4_moe.py`

Added two new kernels:

- `_grouped_gate_up_silu_kernel_sp01_autotuned`
- `_grouped_down_weighted_sum_kernel_sp01_autotuned`

Both bodies are byte-identical to the existing `_grouped_gate_up_silu_kernel`
and `_grouped_down_weighted_sum_kernel`. The only delta is a `@triton.autotune`
decorator with:

```text
gate_up: 17 configs, key=("HIDDEN", "INTERMEDIATE")
         BLOCK_INTER  ∈ {8, 16, 32, 64, 128}
         BLOCK_HIDDEN ∈ {64, 128, 256}
         num_warps    ∈ {2, 4, 8}
         num_stages   ∈ {2, 3}

down:    17 configs, key=("TOP_K", "HIDDEN", "INTERMEDIATE")
         BLOCK_HIDDEN ∈ {8, 16, 32, 64, 128}
         BLOCK_INTER  ∈ {64, 128, 256, 512}
         num_warps    ∈ {2, 4, 8}
         num_stages   ∈ {2, 3}
```

Two Python wrappers exposed:

- `nvfp4_grouped_gate_up_silu_sp01_autotuned(x, expert_ids, packed, scale, gscale)`
- `nvfp4_grouped_down_weighted_sum_sp01_autotuned(inter, expert_ids, weights, packed, scale, gscale)`

The wrappers do NOT accept `block_inter` / `block_hidden` kwargs — autotune
supplies them.

### `engine/moe_packed_nvfp4.py`

Added env-gated dispatch:

```python
_SP01_TRITON_AUTOTUNE = os.environ.get("LYNN_SP_TRITON_AUTOTUNE", "0") == "1"
```

When set, `moe_forward_decode_packed_nvfp4` calls the autotuned variants. Default
off, fully reversible by unsetting the env var.

## Risk Surface

1. **Tiny numeric drift across BLOCK_HIDDEN choices.** BF16 reductions are not
   strictly associative; different inner-loop tile sizes change addition order
   and can perturb the last few mantissa bits. P50 on R6000 already showed this
   can flip low-margin greedy choices after ~5 steps.

   *Mitigation:* SP-01 is gated by quality (V8 / V9 / tool-call / 6-prompt
   coherence), NOT by exact-greedy parity. Spark Lynn 27B currently scores V8
   = 77.1% (beats V Flash 35B by 1.17×) and tool-call 80%, so there is V8
   headroom to absorb sub-percent drift in exchange for material TPS.

2. **First-call autotune cost.** Triton sweeps all 17 configs on first
   invocation per kernel per process. Expect ~50-300 ms one-time penalty at
   server warm-up. Already paid before steady-state TPS sampling begins.

3. **Cache lifecycle.** Triton's autotune cache is in-process only. Restarting
   the server re-tunes. Acceptable since the server is long-lived; if it
   becomes a problem, a future SP-01-B will persist the best config to disk.

## Promotion Gate

SP-01 stays opt-in until ALL of the following pass on Spark:

1. **Microbench parity:** `benchmarks/sp01_sm121_autotune_microbench.py`
   reports min cosine ≥ 0.9999 vs the existing static kernel on a real layer.
2. **TPS win:** mean TPS over 20-prompt mixed-stability bench is ≥ +5% over
   the current 42.85 baseline (so ≥ 45 mean), AND peak ≥ +5% over 44 (so ≥ 46
   peak).
3. **Quality gates pass:**
   - 6-prompt coherent smoke (V8-style prompts) all coherent
   - V8 stage4 stays ≥ 70% (we currently sit at 77.1%, so up to -7pp drift OK)
   - tool-call 15-stage1 stays ≥ 75% (currently 80%, so up to -5pp OK)
   - no `<think>` loop in greedy decode
4. **No regression in long-ctx:** 16k coherent smoke still passes

If any gate fails, the env var stays off, kernel stays opt-in, and we move to
SP-01-B (persistent kernel) or SP-02 (N-gram spec) before promoting.

## Bench Harness

`benchmarks/sp01_sm121_autotune_microbench.py` performs:

1. Loads a single layer of packed NVFP4 expert weights from the production 27B
   shards.
2. Constructs a representative active expert input (top_k=8, hidden=2048).
3. Times static kernel (current production path) over N=1000 launches.
4. Times autotuned kernel — first call triggers Triton sweep, then times
   N=1000 steady-state launches.
5. Records the autotune-picked config from Triton's internal cache.
6. Verifies cosine ≥ 0.9999 between static and autotuned outputs.
7. Emits a JSON report under `reports/sp01_autotune/`.

## Spark Run Commands

After pulling this branch on Spark inside the engine venv:

```bash
# 1) Microbench (validates kernel parity + isolated speedup)
python benchmarks/sp01_sm121_autotune_microbench.py \
    --model-dir /path/to/27B-nvfp4 \
    --layer 6 --iters 1000 \
    --out reports/sp01_autotune/sp01_microbench_$(date +%Y%m%d_%H%M).json

# 2) End-to-end TPS — restart the server with SP01 enabled
LYNN_SP_TRITON_AUTOTUNE=1 \
  python server/openai_http.py --model-dir /path/to/27B-nvfp4

# 3) Same 20-prompt mixed-stability bench used for SGLang baseline
python benchmarks/lynn_27b_vs_35b.py \
    --target lynn-27b-sp01 \
    --runs-single 3 --tokens-single 300 \
    --runs-mixed 20 --tokens-mixed 200 \
    --out reports/sp01_autotune/sp01_tps_$(date +%Y%m%d_%H%M).json

# 4) V8 / V9 / tool-call quality gates (use existing harnesses)
python benchmarks/run_v8_stage4.py --endpoint http://localhost:8080
python benchmarks/run_v9_strict.py --endpoint http://localhost:8080
python benchmarks/run_tool_call_15_stage1.py --endpoint http://localhost:8080
```

## What Comes Next

SP-02 (N-gram lookahead speculative decoding) — closes the +42% peak gap
that SGLang gets from its built-in MTP NEXTN head. Lynn 27B has no internal
MTP head (distilled), so we use prompt n-gram lookahead with batched
verification. Plan in `LYNN_ENGINE_SP02_NGRAM_SPEC_DECODE_PLAN_20260516.md`
(forthcoming).

SP-03 (custom CUDA grouped FP4 kernel) — only if SP-01 + SP-02 combined
still fall short of 65+ peak.

## Discipline

- All SP-N changes ship on `spark/sm121-port`. Never `codex/p16-r6000-155-tps`.
- All TPS numbers reported here are **Spark sm_121 only**. They are NOT
  comparable to R6000 sm_120 numbers (R6000 stable 68-69 / breakthrough 103.44
  is a different machine). Always quote with the device tag.
- R6000 stays Codex's lane (P47/P48 grouped non-atomic kernel work). SP-01
  does not compete with that workstream and does not merge into it.
