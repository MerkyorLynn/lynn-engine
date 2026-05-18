# Lynn Engine Install + CLI Spec

Date: 2026-05-18

Status: spec only. No implementation yet.

## Goal

Compress the current open-engineering footprint (CUDA, PyTorch 2.9.1+cu130,
Triton 3.5.1, lynn-engine source mount, `lmsysorg/sglang:dev-cu13` docker, env
var grid the size of an R6000 production config) into a two-layer end-user
surface:

1. **普通用户 / 服务部署**: one-liner curl install → `lynn-engine pull` → `lynn-engine serve`.
2. **CLI 智能体用户**: same install, plus `lynn-engine agents setup` that writes
   ready-to-use config blocks for Claude Code, Cline, OpenCode, Continue.

Target experience:

```sh
curl -fsSL https://engine.merkyorlynn.com/install.sh | bash
lynn-engine pull qwen36-35b-a3b-w4a16
lynn-engine serve --model qwen36-35b-a3b-w4a16 --api openai
lynn-engine agents setup
```

Open-engineering surface (CUDA toolchain, nvcc, CMake, Triton autotune,
benchmarks, parity gates) stays intact for `lynn-engine dev` and the source
repo. Nothing in this spec breaks the current Spark / R6000 dev flow.

## Domains (per 2026-05-18 hand-off)

| Domain | Role |
|---|---|
| `engine.merkyorlynn.com` | install script + CLI manifest + OpenAI shim docs |
| `dl.merkyorlynn.com` | large-file mirror for model packages + wheels + images (Tencent Cloud, GitHub mirror) |

## Layer 1 — install.sh

Bash, idempotent, no Anaconda channel, Apache/MIT-clean dependencies only
(see `feedback_dont_fork_agpl_inference_engines_20260517` for the AGPL
exclusion). Resolves the right runtime stack for the host:

| Host | Default runtime |
|---|---|
| DGX Spark sm_121 (GB10) | docker `lmsysorg/sglang:dev-cu13` w/ Lynn-engine mount |
| RTX PRO 6000 sm_120a | same docker, native FP4 backends opt-in via env |
| Generic CUDA 13 + Hopper/Blackwell | docker image fallback |
| CPU / no-GPU | refuse with a clear message; suggest `lynn-engine pull` for offline transfer only |

Detection order:

1. `nvidia-smi --query-gpu=name,compute_cap` → resolve sm_arch.
2. `docker version --format '{{.Server.Version}}'` → bail if < 25.0.
3. `python3 -c 'import sys; print(sys.version_info[:2])'` → require ≥3.10.

Install steps:

1. `mkdir -p ~/.local/share/lynn-engine/{bin,cache,models,configs}`.
2. Download `lynn-engine` CLI wrapper script (~3KB Bash) to `~/.local/bin/lynn-engine`,
   chmod 755.
3. Pull pinned docker image (e.g. `lmsysorg/sglang:dev-cu13`) — log size + hash.
4. Write `~/.config/lynn-engine/runtime.toml` with detected arch / image / cuda
   version / model cache path.
5. Print a one-line success banner with what was installed and what next.

Failure modes are surfaced and never silently fall back. Hooks for telemetry
opt-in (`LYNN_ENGINE_TELEMETRY=on`) only after the install finishes — never
during.

## Layer 2 — `lynn-engine` CLI

Single Bash wrapper that dispatches to subcommands. Each subcommand is a
self-contained Python module under `cli/lynn_engine/<cmd>.py` so they can be
swapped to compiled binaries later without breaking UX.

### `lynn-engine pull <package>`

Pulls a published Lynn model package from `dl.merkyorlynn.com`.

Packages (initial catalogue, mirrors `~/models/Qwen3.6-35B-A3B-*` naming):

| Package id | Size | Quality | Speed (Spark single-stream) |
|---|---:|---|---:|
| `qwen36-35b-a3b-bf16` | 67G | MMLU 86.40 / GPQA 45.45 | 30 TPS |
| `qwen36-35b-a3b-q4km-imatrix` | 20G | MMLU 83.00 / GPQA 50.00 | 70 TPS |
| `qwen36-35b-a3b-w4a16-nvfp4` | 23G | MMLU 84.40 / GPQA 49.49 | 39 TPS |
| `qwen36-35b-a3b-mtp-sidecar` | 1.6G | warm-start head | — |

Flags:

- `--mirror <url>` override `dl.merkyorlynn.com` (e.g. for offline LAN cache).
- `--verify-sha256` (default on); manifest pinned in `dl.merkyorlynn.com/manifests/<package>.json`.
- `--parallel <N>` (default 16 streams; mirrors the 16-curl pattern that beats
  single-stream MS download on the Spark wire-speed LAN).
- `--out <path>` (default `~/.local/share/lynn-engine/models/<package>`).

Resolves to a stable on-disk layout the CLI knows how to feed `serve`. Each
package has a `lynn_quant_manifest.json` describing
`quant_method=lynn_native`, `format`, weight/activation contract, and a sha256
chain so `serve` can fail loudly on a partial download.

### `lynn-engine serve [flags]`

Spins the OpenAI-compatible serving endpoint via the pinned docker image, with
env vars chosen for the host arch + chosen package.

Required:

- `--model <package|path>` package id (resolved through `lynn-engine pull`) or
  absolute path to a Lynn-native package directory.

Recommended:

- `--port <N>` default 18099 (production γ default; see Spark history).
- `--api openai` (default) → exposes `/v1/chat/completions` + `/v1/models` +
  `/health`. Compatible with Claude Code, Cline, OpenCode, Continue.
- `--concurrency <N>` default 1; warns if `>1` on hybrid-SSM model because Atlas
  itself documents `--speculative` regression and Lynn's W4A16 path has the
  same shape.
- `--profile {safe-default|amber|custom}` selects the env block.
  - `safe-default` mirrors the safe Config D variant (see refactor plan §Stream B/C).
  - `amber` enables the structured AMBER variant for explicit benchmarking.
  - `custom` reads `LYNN_ENGINE_PROFILE_FILE=...toml` for arbitrary env.
- `--kv-high-precision-layers auto` (default; mirrors Atlas CLI shape so
  Qwen3.6 users have one fewer surprise).
- `--mtp-quantization {off,bf16}` default off until iterative accept is non-zero.
- `--tool-call-parser qwen3_coder` (default; injects the qwen3_coder
  system-prompt format Qwen3.6 ships with).

Output: prints OpenAI base URL + cURL example + suggested next command:
`lynn-engine agents setup`.

### `lynn-engine agents setup [--client <name>]`

Writes ready-to-paste config snippets for the major CLI agent clients. Pulls
the live `/v1/models` + `/health` from the local serve to discover model id +
endpoint, then templates per-client config files.

| Client | Target file |
|---|---|
| Claude Code | `~/.claude/settings.json` (env block) or printed snippet |
| Cline | `~/.config/cline/openai-compat.json` |
| OpenCode | `~/.opencode/providers/lynn-engine.toml` |
| Continue | `~/.continue/config.json` (block) |

By default the command prints the snippet and asks before writing. `--write`
applies the changes. `--client all` walks every supported client. The command
NEVER touches API keys — it sets endpoint + model id only, and reuses
existing keys if the user already has them stored.

### `lynn-engine dev [...]`

Pass-through into the repo's existing dev surface (benchmarks, kernel parity
harness, promotion gate). Only shipped when the CLI is installed in
"developer mode" (env `LYNN_ENGINE_DEV=1` or `--dev` install flag).
Subcommands include:

- `lynn-engine dev bench p25 ...`  → wrap `benchmarks/p25_server_decode_tps_probe.py`.
- `lynn-engine dev gate promote ...` → wrap `scripts/r6000_qwen36_candidate_promotion_gate.sh`.
- `lynn-engine dev parity ...`     → wrap `benchmarks/kernel_parity_harness.py`.

Dev mode preserves the engineering-grade flow (CUDA/Triton/CMake build, env
toggles, candidate gates) without bloating the default UX.

### `lynn-engine status`

Single shot health + version banner:

- detected arch + cuda + docker image
- running container if any (port, image, env profile, uptime)
- pulled packages (size, sha verified)
- last 3 promotion-gate reports

## Backend isolation

CLI does NOT call into `lynn-engine` Python directly. It always shells out to:

- `docker run lmsysorg/sglang:dev-cu13 python3 -m server.openai_http ...` for serving
- `curl` against the local serve for everything else

So a future `lynn-engine` wheel (PyPI, with `pyproject.toml`) that wraps the
core engine API can be added without changing the CLI surface. This also
keeps the source repo decoupled from CLI evolution.

## Telemetry + privacy

Default opt-in only. `LYNN_ENGINE_TELEMETRY` controls a single anonymous
heartbeat that records (a) arch sm_x.y (b) chosen profile (c) CLI exit code.
No prompts, no model outputs, no tokens. Stored at
`https://engine.merkyorlynn.com/telemetry/`; the endpoint refuses payloads
> 256 bytes.

## Non-goals (for this spec)

- Multi-tenant scheduling. Lynn engine is currently single-stream first;
  scheduling work lands after MTP is real.
- Web UI. CLI only. A separate `lynn-desktop` repo handles the agent app.
- Auto-update for the docker image. Image hash is pinned per CLI version;
  upgrading requires `lynn-engine self-update`.
- Windows native. WSL is fine; native Windows is best-effort via docker.

## Open spec questions (to resolve before implementation)

1. Should `lynn-engine pull` use BitTorrent or just multi-stream HTTPS from
   `dl.merkyorlynn.com`? The 67G BF16 package is the limiting factor; LAN
   wire-speed users prefer HTTPS, WAN users may benefit from torrent fallback.
2. How does `lynn-engine serve` discover the safe default + AMBER profiles for
   a non-R6000 host? Proposal: ship per-arch `profiles/<arch>/safe_default.env`
   in the CLI package, mirror the Spark Config D for `sm_121` and the R6000
   P15 profile for `sm_120a`. Audit before ship.
3. Should `agents setup` rewrite an existing config or only append? Proposal:
   append a `lynn-engine` block under an explicit `providers` key and never
   touch other providers' blocks.
4. Should `dl.merkyorlynn.com` host docker images too, or only model
   packages + wheels? Tencent Cloud egress vs Docker Hub: depends on user
   geography. Spec'd separately in `docs/DL_MIRROR_SPEC` (to be written).

## Wire to existing pieces

| Existing | CLI mapping |
|---|---|
| `~/lynn-engine/scripts/spark/run_27b_nvfp4_server.sh` | `lynn-engine serve --profile spark-config-d` |
| `~/lynn-engine/benchmarks/sp01_tps_bench.py` | `lynn-engine dev bench sp01` |
| `~/lynn-engine/benchmarks/p25_server_decode_tps_probe.py` | `lynn-engine dev bench p25` |
| `scripts/r6000_qwen36_candidate_promotion_gate.sh` | `lynn-engine dev gate promote` |
| `~/models/Qwen3.6-35B-A3B-lynn-native-w4a16-nvfp4-from-r6000` | `lynn-engine pull qwen36-35b-a3b-w4a16` |
| existing `lynn_quant_manifest.json` | already CLI-compatible; reused as-is |

## Release plan

| Phase | Scope | Acceptance |
|---|---|---|
| α | install.sh + `serve` + `pull` (Spark sm_121 only) | One-liner installs, serves Qwen3.6 W4A16 W4A16 W4A16 package, OpenAI-compatible /v1/chat/completions returns valid Qwen3.6 chat completion |
| β | `agents setup` for Claude Code + Cline | Two clients work end-to-end against the local serve |
| 1.0 | full client matrix + status + telemetry + R6000 profile | All four clients work; promotion-gate `dev` subcommands round-trip; install passes on Spark + R6000 + generic CUDA 13 box |

Each phase ends with a clean dl.merkyorlynn.com release manifest + an
`engine.merkyorlynn.com/install.sh` build that pulls only that phase's
artifacts.
