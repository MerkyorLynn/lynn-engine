# Qwen3.5-9B Local Agent Endpoint Plan

Date: 2026-05-19

## Decision

Make the 9B dense model the first end-user local-agent endpoint.

The release shape is two compatible runtime tracks behind the same
OpenAI-compatible API contract:

| Track | Default hardware | Artifact | Runtime | Purpose |
|---|---|---|---|---|
| Mac / generic | Apple Silicon, CPU, generic CUDA | Q4_K_M GGUF | llama.cpp | Fastest path to a usable local agent endpoint |
| NVIDIA native | R6000 / Blackwell | Lynn-native NVFP4 | Lynn Engine | Native FP4/FP8 kernel path and future acceleration |

The user-facing agent config should not care which track is underneath:

```text
base_url=http://127.0.0.1:18099/v1
api_key=local
model=qwen35-9b-q4km
```

For NVIDIA native:

```text
base_url=http://127.0.0.1:18099/v1
api_key=local
model=qwen35-9b-nvfp4
```

## Why Q4_K_M First

Q4_K_M / llama.cpp is already mature enough to provide the first complete
local-agent story:

- no Python/CUDA extension dependency for Mac users;
- small model package, around the 5 GB class;
- OpenAI-compatible `llama-server`;
- works with Claude Code, Cline, OpenCode, Continue, and any agent that can
  point at `base_url`;
- gives Lynn a stable product baseline while the NVIDIA NVFP4 kernel path keeps
  improving.

This is not a retreat from Lynn Engine. It splits the product correctly:

- **Q4_K_M**: compatibility and immediate endpoint adoption;
- **NVFP4**: NVIDIA-specific performance moat and custom kernel work.

## Local Launcher

The local launcher lives at:

```bash
scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

It discovers:

1. `llama-server` from PATH/Homebrew/common build paths;
2. `Qwen3.5-9B Q4_K_M` GGUF from `$GGUF`, `$MODEL_ROOT`, `~/Models`,
   `~/Downloads`, and HuggingFace cache.

Example:

```bash
GGUF=/Users/lynn/Models/Qwen3.5-9B-Q4_K_M.gguf \
  bash scripts/local_qwen35_9b_q4km_llamacpp_server.sh
```

Endpoint:

```text
http://127.0.0.1:18099/v1
```

Smoke:

```bash
bash scripts/local_qwen35_9b_q4km_smoke.sh
```

## Agent CLI Config

All agent integrations should be expressed as this minimal contract:

```text
provider=openai-compatible
base_url=http://127.0.0.1:18099/v1
api_key=local
model=qwen35-9b-q4km
```

The install helper can later emit concrete snippets for:

- Claude Code;
- Cline;
- OpenCode;
- Continue;
- LM Studio compatible clients.

## Release Gates

Before publishing the Q4_K_M endpoint track:

| Gate | Requirement |
|---|---|
| Server smoke | `/v1/chat/completions` returns non-empty text |
| Structured smoke | JSON/code/YAML mini-set passes |
| 32K context smoke | long prompt does not crash |
| MMLU / GPQA record | published as model-card quality numbers |
| Agent config | at least one CLI agent can send a request |

The NVIDIA NVFP4 track keeps separate gates:

| Gate | Requirement |
|---|---|
| W4A16 safe default | no structured regressions |
| W4A8 route | content gate passes before speed claims |
| Native FP4/FP8 kernels | fixture + resident gates before promotion |

## Current Product Framing

9B gives the first complete local-agent product:

```text
Mac users:      Q4_K_M + llama.cpp endpoint
NVIDIA users:   Lynn-native NVFP4 + Lynn Engine endpoint
35B users:      quality/reference branch, not the day-one endpoint
```

Once this endpoint is smooth, optimization work can continue underneath without
blocking the product story.
