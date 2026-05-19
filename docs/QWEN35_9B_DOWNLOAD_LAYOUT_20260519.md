# Qwen3.5-9B 下载页布局 · 2026-05-19

下载域名：`dl.merkyorlynn.com`

文档域名：`engine.merkyorlynn.com`

## 目录结构

```text
dl.merkyorlynn.com/
└── models/
    └── qwen35-9b/
        ├── README.md
        ├── model-card.md
        ├── release-gate.json
        ├── checksums.sha256
        ├── q4_k_m/
        │   ├── Qwen3.5-9B-Q4_K_M-imatrix.gguf
        │   └── README.md
        ├── nvfp4-w4a16/
        │   ├── qwen35-9b-nvfp4-w4a16.tar.zst
        │   ├── qwen35-9b-nvfp4-w4a16.manifest.json
        │   └── README.md
        └── nvfp4-compact-candidate/
            ├── qwen35-9b-nvfp4-compact-candidate.tar.zst
            ├── qwen35-9b-nvfp4-compact-candidate.manifest.json
            └── README.md
```

## 下载卡片

| 卡片 | 文件 | 状态 | 用户 | Runtime |
|---|---|---|---|---|
| Mac Q4_K_M GGUF | `q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf` | stable | Mac / general local agents | llama.cpp / LM Studio / CLI |
| NVIDIA NVFP4 W4A16 | `nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.tar.zst` | stable | NVIDIA Linux / Blackwell | Lynn Engine |
| NVIDIA compact NVFP4 candidate | `nvfp4-compact-candidate/qwen35-9b-nvfp4-compact-candidate.tar.zst` | candidate | NVIDIA Linux testers | Lynn Engine |
| Checksums | `checksums.sha256` | required | all users | checksum tools |
| Model card | `model-card.md` | docs | all users | browser |
| Release gate JSON | `release-gate.json` | evidence | reviewers / automation | JSON |

## checksums.sha256

`checksums.sha256` should cover every public artifact and metadata file:

```text
<sha256>  <size_bytes>  q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf
<sha256>  <size_bytes>  nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.tar.zst
<sha256>  <size_bytes>  nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.manifest.json
<sha256>  <size_bytes>  nvfp4-compact-candidate/qwen35-9b-nvfp4-compact-candidate.tar.zst
<sha256>  <size_bytes>  nvfp4-compact-candidate/qwen35-9b-nvfp4-compact-candidate.manifest.json
<sha256>  <size_bytes>  model-card.md
<sha256>  <size_bytes>  release-gate.json
```

Verification copy:

```bash
python3 scripts/qwen35_9b_release_checksums.py verify \
  --manifest checksums.sha256 \
  --root ~/Models/Lynn/Qwen3.5-9B \
  --out reports/qwen35_9b/local_checksum_verify.json
```

## README.md 页面文案

### 标题

Qwen3.5-9B for Lynn Engine Local Inference

### 简介

Choose the stable Mac path if you want the simplest local setup. Choose the
NVIDIA path if you want the Lynn Engine NVFP4 runtime. Always download
`checksums.sha256` and verify artifacts before first run.

### 推荐选择

| If you have... | Download this |
|---|---|
| MacBook / Mac Studio | `q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf` |
| NVIDIA Blackwell Linux box | `nvfp4-w4a16/qwen35-9b-nvfp4-w4a16.tar.zst` |
| NVIDIA test box and want compact package trial | `nvfp4-compact-candidate/` candidate only |

## Artifact notes

### Mac Q4_K_M GGUF

- Stable first-release Mac path.
- Works with llama.cpp, LM Studio, and CLI wrappers.
- Best user-facing path for fast setup.

### NVIDIA NVFP4 W4A16

- Stable first-release NVIDIA path.
- Intended for Lynn Engine.
- Current package may keep BF16 `embed_tokens` and `lm_head`; publish package size and checksum together.

### NVIDIA compact NVFP4 candidate

- Candidate only.
- Do not mark stable until release gates pass.
- Page copy should say: "Try this only if you are validating the compact package candidate. Use W4A16 for the stable NVIDIA path."

## Release gate JSON

`release-gate.json` should include machine-readable status for:

- Mac Q4_K_M GGUF smoke;
- NVIDIA NVFP4 W4A16 smoke;
- checksum status;
- W4A8 / FP4xFP8 resident status as experimental;
- 9B thinking-on 32K GPQA as in-progress if referenced;
- 35B track as side-track, not first-release path;
- MTP excluded from TPS credit.

## Guardrails for download page

- Do not claim final 9B thinking-on 32K GPQA full score while the run is still active.
- Do not present 35B as part of this 9B first-release download path.
- Do not count MTP as TPS credit.
- Do not mark compact NVFP4 candidate as stable.
- Do not hide checksum verification behind optional text; show it as the first post-download step.
