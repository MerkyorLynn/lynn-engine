# Qwen3.5-9B App Local-First Gate Contract · 2026-05-19

## Goal

Define the app-side gate for offering Qwen3.5-9B local-first setup without breaking first launch. The app default provider remains MIMO until local runtime smoke passes.

This document does not duplicate `scripts/local_lynn_app_model_probe.py`; it records how the app consumes that probe and when it may switch from MIMO-first to local-first.

## Release positioning

- Mac stable track: Qwen3.5-9B Q4_K_M imatrix GGUF through llama.cpp / LM Studio.
- NVIDIA track: Lynn Engine NVFP4.
- Optional 35B is opt-in for large-memory Macs and must not block 9B first-run.
- App first launch must remain usable even if no local model is installed.

## Product entrypoint

The app/installer should not ask ordinary users to paste shell commands. It
should call the client bootstrap helper and present a normal authorization UI.

Plan call:

```bash
python3 scripts/local_qwen35_9b_client_bootstrap.py plan
```

The returned JSON lists actions such as `install_runtime`, `download_model`,
`register_provider`, and `smoke_and_start`. The client shows those actions,
download size, destination path, and MIMO fallback guarantee to the user.

After the user authorizes:

```bash
python3 scripts/local_qwen35_9b_client_bootstrap.py execute --yes-user-authorized --start
```

Internally, the bootstrap delegates to the release setup script:

```bash
bash scripts/local_qwen35_9b_setup.sh --install-runtime --download --smoke --serve
```

That command:

1. installs `llama.cpp` on macOS/Homebrew if missing;
2. downloads or discovers the recommended Q4_K_M-imatrix GGUF;
3. writes `~/Models/Lynn/Qwen3.5-9B/lynn-qwen35-9b-q4km.env`;
4. writes `~/.lynn-engine/providers/qwen35-9b-q4km-imatrix-gguf.json`;
5. runs chat + tool-call + decode smoke;
6. starts the persistent OpenAI-compatible local endpoint.

The client should still treat MIMO as fallback until the smoke report passes.

## State machine

```text
MIMO_FIRST
  -> OFFER_LOCAL
  -> DOWNLOADING
  -> VERIFYING
  -> SMOKE
  -> LOCAL_FIRST

Any failure after OFFER_LOCAL:
  -> FALLBACK_MIMO
```

### States

| State | Meaning | App behavior |
|---|---|---|
| `MIMO_FIRST` | Default app state. | Use MIMO provider. Run local probe in setup/background. |
| `OFFER_LOCAL` | Probe says a local offer is reasonable. | Show local 9B offer; optionally show 35B offer if present. Do not switch provider yet. |
| `DOWNLOADING` | User accepted local setup. | Download selected artifact from approved source; keep MIMO usable. |
| `VERIFYING` | Artifact exists locally. | Verify file size and SHA256 against manifest. |
| `SMOKE` | Runtime can be launched. | Run required smoke checks from probe offer. |
| `LOCAL_FIRST` | All required smoke checks passed. | Set local provider as first provider for this model/runtime. Keep MIMO fallback configured. |
| `FALLBACK_MIMO` | Any local setup gate failed or user skipped. | Keep MIMO active; show repair/retry action. |

## Transition rules

### `MIMO_FIRST -> OFFER_LOCAL`

Allowed only when `scripts/local_lynn_app_model_probe.py` returns:

```json
{
  "local_first_allowed_after_smoke": true,
  "decision": "offer_local_models_after_app_setup",
  "offers": [ ... ]
}
```

The app must not switch provider at this point. It may only show an offer.

### `OFFER_LOCAL -> DOWNLOADING`

Allowed when the user chooses an offer.

Recommended copy for 9B:

```text
Try local Qwen3.5-9B on this Mac?

This downloads the stable Q4_K_M imatrix GGUF and runs it locally with llama.cpp.
The app will keep MIMO available and will only switch to local-first after checksum and smoke tests pass.
```

Optional copy for 35B:

```text
Optional: try the larger 35B local model

Your Mac appears to have enough memory and disk for the 35B quality path. This is opt-in and separate from the recommended 9B first-run path. If setup fails, the app keeps using MIMO.
```

### `DOWNLOADING -> VERIFYING`

Allowed after the selected file is fully downloaded and not marked `.partial`.

The app should keep MIMO active while download happens.

### `VERIFYING -> SMOKE`

Allowed only after:

- file exists;
- file size matches manifest;
- SHA256 matches manifest;
- artifact ID matches selected offer.

### `SMOKE -> LOCAL_FIRST`

Allowed only after every entry in `offer.smoke_required` passes.

For the current Mac 9B offer this is:

```json
[
  "download_manifest",
  "sha256",
  "llama_cpp_v1_models",
  "chat_32_tokens",
  "tool_call_weather"
]
```

### Any failure -> `FALLBACK_MIMO`

Fallback is not an error state for first launch. It means the app remains usable with MIMO and can offer retry instructions.

## Failure handling

| Failure | Detect in | App action | User copy |
|---|---|---|---|
| Checksum fail | `VERIFYING` | Delete or quarantine artifact; keep MIMO active. | "The local model download did not match the published checksum. Please retry the download." |
| llama-server fail | `SMOKE` | Keep MIMO active; show install/build help. | "The local runtime did not start. Install or rebuild llama.cpp with Metal, then retry." |
| Chat smoke fail | `SMOKE` | Keep MIMO active; show retry and log path. | "The local endpoint started but did not answer the smoke prompt. MIMO remains active." |
| Insufficient disk | `MIMO_FIRST` or `DOWNLOADING` | Do not offer or pause download; keep MIMO active. | "Free more disk space before downloading the local model." |
| Insufficient memory | `MIMO_FIRST` | Do not offer local 9B/35B if below threshold; keep MIMO active. | "This device may not have enough unified memory for the local model. You can continue with MIMO." |
| User cancels | any setup state | Move to `FALLBACK_MIMO`. | "Local setup skipped. You can retry from Settings." |

## Probe JSON contract

The app consumes the exact JSON emitted by `scripts/local_lynn_app_model_probe.py`.

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Currently `lynn-app-local-model-probe-v1`. |
| `platform` | object | Host capability summary. |
| `default_provider` | string | Currently `mimo`; app starts here. |
| `fallback_provider` | string | Currently `mimo`; app returns here on failure. |
| `local_first_allowed_after_smoke` | boolean | Whether any local offer is reasonable, not whether local is already active. |
| `offers` | array | Local model offers the app may show. |
| `decision` | string | UI decision hint. |
| `notes` | array of string | App behavior guardrails. |

### `platform` fields

| Field | Type | Meaning |
|---|---|---|
| `system` | string | `platform.system()`, e.g. `Darwin`. |
| `machine` | string | `platform.machine()`, e.g. `arm64`. |
| `is_macos_apple_silicon` | boolean | True for Darwin arm64/aarch64. |
| `unified_memory_gib` | number or null | Rounded total memory in GiB. |
| `model_root` | string | Directory checked for local model artifacts. |
| `free_disk_gib` | number or null | Rounded free disk in GiB at `model_root`. |

### Offer fields

| Field | Type | Meaning |
|---|---|---|
| `artifact_id` | string | Stable artifact ID, e.g. `qwen35-9b-q4km-imatrix-gguf`. |
| `model` | string | Runtime model name, e.g. `qwen35-9b-q4km-imatrix`. |
| `runtime` | string | Runtime label, e.g. `llama.cpp-metal`. |
| `priority` | string | `recommended` for 9B, `optional` for 35B. |
| `reason` | string | Human-readable why this offer is shown. |
| `download_gib` | number | Approximate download size in GiB. |
| `min_unified_memory_gib` | integer | Minimum memory threshold used for offer. |
| `min_free_disk_gib` | integer | Minimum disk threshold used for offer. |
| `smoke_required` | array of string | Checks required before local-first activation. |

### Decision values

| Value | Meaning |
|---|---|
| `offer_local_models_after_app_setup` | Show local setup offer after app setup. |
| `keep_mimo_first_no_local_offer` | Do not offer local model by default; keep MIMO first. |

### Notes consumed as guardrails

The probe emits these guardrails:

```json
[
  "Never block first app launch on local model setup.",
  "Only switch local_provider.priority=first after all smoke_required checks pass.",
  "If any local runtime check fails, keep MIMO as the active provider."
]
```

The app should treat these as product rules, not merely diagnostics.

## Offer examples

### Recommended 9B offer

```json
{
  "artifact_id": "qwen35-9b-q4km-imatrix-gguf",
  "model": "qwen35-9b-q4km-imatrix",
  "runtime": "llama.cpp-metal",
  "priority": "recommended",
  "reason": "Apple Silicon with enough unified memory and disk for the stable 9B local-agent path.",
  "download_gib": 5.3,
  "min_unified_memory_gib": 8,
  "min_free_disk_gib": 8,
  "smoke_required": [
    "download_manifest",
    "sha256",
    "llama_cpp_v1_models",
    "chat_32_tokens",
    "tool_call_weather"
  ]
}
```

### Optional 35B offer

```json
{
  "artifact_id": "qwen36-35b-a3b-q4km-imatrix-gguf",
  "model": "qwen36-35b-a3b-q4km",
  "runtime": "llama.cpp-metal",
  "priority": "optional",
  "reason": "Large-memory Mac can try the 35B quality path, but it stays opt-in.",
  "download_gib": 20.0,
  "min_unified_memory_gib": 32,
  "min_free_disk_gib": 30,
  "smoke_required": [
    "download_manifest",
    "sha256",
    "llama_cpp_v1_models",
    "chat_32_tokens",
    "tool_call_weather",
    "short_structured_smoke"
  ]
}
```

## Smoke check definitions

| Smoke key | Required behavior |
|---|---|
| `download_manifest` | Manifest exists and includes selected `artifact_id`, filename, size, sha256, and sources. |
| `sha256` | Local file SHA256 matches manifest. |
| `llama_cpp_v1_models` | `llama-server` starts and `/v1/models` returns the selected model name. |
| `chat_32_tokens` | `/v1/chat/completions` returns non-empty text with `max_tokens` near 32. |
| `tool_call_weather` | OpenAI-compatible `tools` request returns a `get_weather` function call with the requested city. |
| `short_structured_smoke` | Optional 35B structured output smoke returns parseable short JSON or equivalent app-defined structured response. |

## Provider activation contract

Local-first activation must be explicit:

```json
{
  "active_provider": "local",
  "fallback_provider": "mimo",
  "local_provider": {
    "priority": "first",
    "model": "qwen35-9b-q4km-imatrix",
    "base_url": "http://127.0.0.1:8080/v1",
    "runtime": "llama.cpp-metal",
    "artifact_id": "qwen35-9b-q4km-imatrix-gguf"
  }
}
```

Before smoke passes, provider config must remain equivalent to:

```json
{
  "active_provider": "mimo",
  "fallback_provider": "mimo",
  "local_provider": {
    "priority": "available_after_smoke"
  }
}
```

## UI copy blocks

### First-run banner

```text
Run Qwen3.5-9B locally on this Mac

Your app will keep using MIMO while we download and verify the local model. If the local runtime passes smoke checks, you can make it the first provider. If anything fails, MIMO stays active.
```

### 9B CTA

```text
Try local 9B
Stable Mac path: Qwen3.5-9B Q4_K_M imatrix GGUF with llama.cpp Metal.
```

### Optional 35B CTA

```text
Optional: try 35B
For large-memory Macs only. This is opt-in and separate from the recommended 9B setup.
```

### Fallback copy

```text
Local setup did not complete. MIMO is still active, so you can keep working. You can retry local setup from Settings after fixing the issue.
```

## Non-goals

- Do not make local setup mandatory for first launch.
- Do not switch away from MIMO before all smoke checks pass.
- Do not use 35B as the default first-run offer.
- Do not treat thinking-on 32K GPQA as a short-answer TPS benchmark.
- Do not duplicate or replace `scripts/local_lynn_app_model_probe.py`.
