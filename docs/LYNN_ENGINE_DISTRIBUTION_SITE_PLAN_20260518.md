# Lynn Engine Distribution Site Plan

Date: 2026-05-18

## Canonical Domain

Use `engine.merkyorlynn.com` as the canonical Lynn Engine entry point.

Purpose:

- product landing page;
- install instructions;
- release notes;
- model/runtime compatibility matrix;
- OpenAI-compatible server examples;
- CLI-agent setup docs.

Keep large binary traffic separate from the canonical site:

| Domain | Purpose |
|---|---|
| `engine.merkyorlynn.com` | docs, install page, release index |
| `dl.merkyorlynn.com` | resolved download domain for binaries, model bundles, and release artifacts |
| `mirror.merkyorlynn.com` | domestic mirror endpoint if needed |

The root install page should treat GitHub as source control and
`dl.merkyorlynn.com` as delivery infrastructure. Do not make GitHub the only
install path for China users.

Recommended URL shape:

| URL | Use |
|---|---|
| `https://engine.merkyorlynn.com/install.sh` | small bootstrap installer |
| `https://engine.merkyorlynn.com/docs` | docs and compatibility matrix |
| `https://dl.merkyorlynn.com/releases/` | wheels, native binaries, Docker metadata |
| `https://dl.merkyorlynn.com/models/` | quantized model bundles and manifests |
| `https://dl.merkyorlynn.com/checksums/` | checksums and signatures |

## Target User Install Shape

The public install target should hide the current developer dependency surface.
Users should not need to understand CUDA/Triton/native-kernel details before
they can start the OpenAI-compatible server.

Desired flow:

```bash
curl -fsSL https://engine.merkyorlynn.com/install.sh | bash
lynn-engine pull qwen36-35b-a3b-w4a16
lynn-engine serve --model qwen36-35b-a3b-w4a16 --api openai
```

Agent setup should be a separate one-command helper:

```bash
lynn-engine agents setup
```

That helper can write local configuration snippets for Claude Code, Cline,
OpenCode, Continue, or any OpenAI-compatible CLI that only needs:

```text
base_url=http://127.0.0.1:18099/v1
model=qwen36-35b-a3b-w4a16
api_key=local
```

## Packaging Principle

Developer dependencies are allowed to be heavy. User dependencies should be one
of:

1. Docker image with CUDA runtime and Lynn Engine included.
2. Prebuilt wheel plus a model/runtime bundle.
3. Native binary bundle after the kernel islands stabilize.

Near term, Docker is the safest public route because it can pin:

- CUDA runtime;
- PyTorch;
- Triton;
- compiled Lynn native extension;
- server entry point;
- known-good runtime env flags.

The CLI should expose profiles rather than raw env variables:

| Profile | Meaning |
|---|---|
| `safe` | official Qwen3.6-35B-A3B W4A16, exact-greedy default |
| `structured-fast` | opt-in AMBER structured-serving profile |
| `research` | local kernel/MTP experiments, never default |

## Current Release Candidate

The current publishable technical anchor is:

```text
official Qwen/Qwen3.6-35B-A3B
  -> Lynn-native W4A16 NVFP4
  -> safe default R6000 profile
```

Current R6000 gate anchor:

| Gate | Result |
|---|---:|
| P37 exact greedy | GREEN |
| P25 512-token decode TPS | 107.43 |
| Hard structured gate | GREEN, 40/40 |
| Hard structured mean decode TPS | 107.86 |

The AMBER structured-fast profile is useful but should not be the default until
its exact-greedy drift is accepted or fixed.
