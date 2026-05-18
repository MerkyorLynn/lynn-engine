# Spark Optimization Branch Plan (2026-05-15)

This is the recommended split for running a Spark-specific Lynn Engine branch
in parallel with the R6000 main optimization line.

## Branch

Use a separate branch:

```bash
git checkout -b codex/spark-sm121-engine-path
```

Suggested upstream name:

```text
feat/spark-sm121-engine-path
```

Do not mix Spark-only experiments into the R6000 production branch until a gate
passes on both machines.

## Ownership Split

| Track | Owner | Goal |
|---|---|---|
| R6000 mainline | Codex | First-class production path, 100 TPS then 200 TPS |
| Spark branch | Claude / Spark worker | sm_121 compatibility, launch scripts, eval harness, native path validation |

## Spark Baseline

Current Spark result for Lynn 27B NVFP4:

| Item | Result |
|---|---:|
| Backend | scalar_bridge |
| Server memory | ~79G used / ~39G free |
| 6-prompt smoke | 6/6 pass |
| Tool-call strict | pass |
| Long-ctx 16K Chinese | 5/5 pass |
| Decode speed | ~24 tok/s |

This is a quality and serving baseline, not the target performance path.

## Spark Branch Tasks

1. Keep a known-good scalar bridge launch script.
2. Add a second launch script for native-path experiments.
3. Verify sm_121 support for:
   - packed NVFP4 MoE aliases,
   - fused native FP4 linear-attn in-proj,
   - native FP4 lm_head,
   - any CUDA graph path being tested.
4. Run a fixed master eval after every performance change:
   - health,
   - 6-prompt smoke,
   - strict tool call,
   - V9 holdout,
   - long-ctx,
   - TPS x3.
5. Keep Spark-only reports under:

```text
reports/spark-sm121/
```

## Performance Expectations

Spark at 24 tok/s was the scalar bridge baseline, not the final product target.
The official Qwen3.6-35B-A3B pivot and the 2026-05-18 three-way benchmark now
change the Spark framing:

| Quant | Stack | Single-stream TPS |
|---|---|---:|
| BF16 official | SGLang dev-cu13 | 30.14 |
| Q4_K_M-imatrix GGUF | llama.cpp server-cuda | 69.77 |
| W4A16 NVFP4 Lynn-native | lynn-engine Config D | 38.96 |

Q4_K_M/llama.cpp is the current Spark single-stream serving default. It is the
fastest tested Spark route and pairs with the checked quality result
(`83.00%` MMLU / `50.00%` GPQA). Lynn-native W4A16 is highly stable on Spark
(`0.09` TPS stddev) but much slower because `sm_121` does not expose the same
native FP4 MMA path that makes R6000 fast.

Corrected Spark target ladder:

| Stage | Target | Meaning |
|---|---:|---|
| Scalar fallback | 24 tok/s | Known-good quality baseline. |
Current Lynn-native W4A16 | 39 tok/s | Stable fallback and compatibility path. |
Spark serving default | 70 tok/s | Q4_K_M/llama.cpp user-facing route. |
True Spark-native mirror | >70 tok/s | Only worth pursuing with a real FP8-friendly path. |
Lynn stretch | > llama.cpp on same prompt/token budget | Acceptance gate for a Spark Lynn-native promotion. |

Spark should not start from a full Rust rewrite. It should only consume native
kernel work if the kernel is compatible with `sm_121` and can beat the local
llama.cpp Q4_K_M baseline on identical prompts. Otherwise, keep Spark serving on
Q4_K_M and spend the main kernel budget on R6000.

2026-05-17/18 Atlas correction:

The widely repeated `130 TPS` Atlas number is for Qwen3.5-35B-A3B with MTP, not
the official Qwen3.6-35B-A3B hybrid SSM route. Treat Atlas as an external
reference, not as proof that Qwen3.6 Spark MTP is a solved 130 TPS path.

Atlas reproduction on Spark is useful but not blocking. If Spark is idle, run
the official Atlas image as an external baseline and record:

```text
image digest
model id
exact command
max context
MTP on/off
warmup tokens
decode benchmark prompt/token budget
single-stream and 4-concurrent results
tool-call smoke result
```

Do not use Atlas reproduction as a substitute for Lynn's own gate. The Spark
acceptance criterion remains Lynn engine > llama.cpp on the same prompt/token
budget with Lynn's model/artifact.

## Guardrails

- Do not publish Spark TPS as R6000 TPS.
- Do not promote a Spark-only launch flag into README defaults until R6000 also
  passes.
- Do not use simplified chat templates for tool-call eval. Use the full
  no-think template with the tools block preserved.
- Keep `scalar_bridge` as the fallback path so quality validation is never
  blocked by native-kernel experiments.

## Suggested Deliverables

```text
scripts/run_spark_scalar_bridge_server.sh
scripts/run_spark_native_probe_server.sh
reports/spark-sm121/master_eval_*.json
reports/spark-sm121/tps_*.json
docs/SPARK_SM121_STATUS_*.md
```
