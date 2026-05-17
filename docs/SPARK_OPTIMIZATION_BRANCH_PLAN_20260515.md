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

Spark at 24 tok/s is not an architectural ceiling. It is the scalar bridge
baseline. If Spark can consume the same native packed path as R6000, a 50+
tok/s target is reasonable. Treat 50+ as the Spark mid-term goal, not the
current branch acceptance gate.

2026-05-17 Atlas update:

Atlas changes the long-term Spark expectation. Its public GB10 numbers show
Qwen3.5-35B-A3B around `~130` tok/s peak / `~111` tok/s average with Rust+CUDA
and MTP K=2, and about `70` tok/s without speculative decoding in its own README.
Those are different model/runtime conditions and should not be pasted onto Lynn,
but they prove Spark-class hardware can run far above the current Lynn Config D
class when the engine owns CUDA kernels, buffer state, and speculative verify.

Updated Spark target ladder:

| Stage | Target | Meaning |
|---|---:|---|
| Scalar fallback | 24 tok/s | Known-good quality baseline. |
| Current Config D class | ~49 tok/s | Stable production-like floor. |
| Native packed + W4A8 mirror | 60-90 tok/s | First serious llama.cpp comparison band. |
| Native core + K=2 MTP verify | 90-130 tok/s | Atlas-like architecture band, not guaranteed. |
| Lynn stretch | > llama.cpp on same prompt/token budget | Actual acceptance gate. |

Spark should not start from a full Rust rewrite. It should consume the same
C++/CUDA decode-core gates as R6000, then wrap them in a cleaner server if the
core beats llama.cpp on identical benchmarks.

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
