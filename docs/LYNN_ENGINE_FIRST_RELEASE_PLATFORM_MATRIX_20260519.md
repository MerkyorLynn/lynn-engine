# Lynn Engine First Release Platform Matrix

Date: 2026-05-19

## Release Principle

The first public release must cover both major local-inference audiences:

1. **Mac / Apple Silicon users** who want a simple local agent endpoint.
2. **NVIDIA users** who want the Lynn-native NVFP4 accelerated path.

Do not force every platform through the same runtime. The product contract is
the same OpenAI-compatible endpoint; the runtime underneath can differ by
hardware.

## V1 Platform Targets

| Platform | V1 Status | Runtime | Model Artifact | Endpoint | Notes |
|---|---|---|---|---|---|
| macOS Apple Silicon | **Required / stable** | llama.cpp Metal | Qwen3.5-9B Q4_K_M GGUF | `http://127.0.0.1:18099/v1` | Fastest path to a polished local-agent experience |
| Linux + NVIDIA CUDA | **Required / stable** | Lynn Engine | Qwen3.5-9B Lynn-native NVFP4 W4A16, plus opt-in W4A8 research | `http://127.0.0.1:18099/v1` | Main native-kernel performance track |
| Windows + NVIDIA | **Beta** | WSL2 + Docker / CUDA | Same as Linux NVIDIA | `http://127.0.0.1:18099/v1` | Support through WSL2 first; native Windows is not V1 |
| Linux CPU / generic GPU | **Compatible** | llama.cpp | Qwen3.5-9B Q4_K_M GGUF | `http://127.0.0.1:18099/v1` | Good fallback; not the performance headline |
| Windows native llama.cpp | **Community / later** | llama.cpp Vulkan/CUDA builds | Q4_K_M GGUF | `http://127.0.0.1:18099/v1` | Useful, but too many install variants for V1 support burden |

## Why Windows Is Beta In V1

Windows users matter, especially NVIDIA laptop/desktop users, but native Windows
support multiplies the support matrix:

- CUDA driver versions;
- Visual Studio build toolchain;
- PowerShell path and permission differences;
- llama.cpp binary variants;
- Docker Desktop / WSL2 configuration.

For V1, support the Windows NVIDIA route through:

```text
Windows 11 + WSL2 Ubuntu + NVIDIA driver + Docker or Python wheel
```

That keeps the runtime identical to Linux while still serving Windows users.

## Install UX

The user-facing command should eventually choose the right runtime:

```bash
lynn-engine doctor
lynn-engine pull qwen35-9b
lynn-engine serve qwen35-9b --api openai
```

Under the hood:

```text
macOS arm64        -> Q4_K_M GGUF + llama.cpp Metal
Linux NVIDIA      -> Lynn-native NVFP4 + Lynn Engine CUDA
Windows WSL2 CUDA -> Lynn-native NVFP4 + Lynn Engine CUDA
fallback          -> Q4_K_M GGUF + llama.cpp
```

## Model Names

Expose stable names instead of implementation details:

| Public name | Runtime target |
|---|---|
| `qwen35-9b-local` | Auto-select best runtime |
| `qwen35-9b-q4km` | Force GGUF / llama.cpp |
| `qwen35-9b-nvfp4` | Force Lynn-native NVFP4 |
| `qwen36-35b-a3b-nvfp4` | 35B research / high-quality NVIDIA track |

## V1 Acceptance Gates

### Mac Stable

| Gate | Required |
|---|---|
| llama.cpp server starts from launcher | Yes |
| OpenAI chat smoke | Yes |
| Structured mini-gate | Yes |
| 32K context smoke | Yes |
| MMLU / GPQA values documented | Yes |
| Agent CLI config snippet | Yes |

### NVIDIA Stable

| Gate | Required |
|---|---|
| Lynn server starts from Docker/wheel | Yes |
| OpenAI chat smoke | Yes |
| Structured mini-gate | Yes |
| 9B NVFP4 speed reported on R6000 | Yes |
| W4A16 remains safe fallback | Yes |
| W4A8 only opt-in unless content gate passes | Yes |

### Windows Beta

| Gate | Required |
|---|---|
| WSL2 instructions | Yes |
| Driver/Docker checklist | Yes |
| Same Linux command works inside WSL2 | Yes |
| Marked beta in docs | Yes |

## Distribution Domains

Use:

| Domain | Purpose |
|---|---|
| `engine.merkyorlynn.com` | install page, docs, compatibility matrix |
| `dl.merkyorlynn.com` | model bundles, GGUF, NVFP4 packs, checksums |

Suggested download layout:

```text
https://dl.merkyorlynn.com/models/qwen35-9b/q4km/
https://dl.merkyorlynn.com/models/qwen35-9b/nvfp4/
https://dl.merkyorlynn.com/releases/lynn-engine/
```

## Summary

V1 should not be "Lynn Engine only." It should be:

```text
Mac:      Q4_K_M + llama.cpp, stable
NVIDIA:   NVFP4 + Lynn Engine, stable on Linux, beta on Windows WSL2
35B:      quality/reference branch, not the first endpoint promise
```

That gives users a working local agent immediately while preserving the native
NVIDIA performance path as Lynn's technical moat.
