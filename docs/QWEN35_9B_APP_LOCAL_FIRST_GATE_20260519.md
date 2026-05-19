# Qwen3.5-9B Desktop App Local-First Gate Contract

**Date:** 2026-05-19
**Scope:** Lynn desktop app first-run wizard for switching from MIMO cloud
default to local Qwen3.5-9B Q4_K_M (Mac stable) or optional 35B-A3B (large
unified memory only).
**Probe source:** `scripts/local_lynn_app_model_probe.py` (already exists; do
not duplicate).

---

## Design Principle

The app must stay usable from first launch even when no local model is
installed. The local-first wizard is an **upgrade path**, never a hard
prerequisite. If any step fails, the app silently falls back to MIMO and the
user can retry the wizard later.

## State Machine

```
                    ┌─────────────┐
                    │ MIMO_FIRST  │  default at first launch
                    └──────┬──────┘
                           │ user opens "Use local model" wizard
                           ▼
                    ┌─────────────┐
                    │ OFFER_LOCAL │  show probe.offers[]
                    └──────┬──────┘
                           │ user accepts an offer
                           ▼
                    ┌─────────────┐
                    │ DOWNLOADING │  pull GGUF + manifest from dl.merkyorlynn.com
                    └──────┬──────┘
                           │ download complete
                           ▼
                    ┌─────────────┐
                    │  VERIFYING  │  sha256 + manifest schema check
                    └──────┬──────┘
                           │ verify ok
                           ▼
                    ┌─────────────┐
                    │    SMOKE    │  start llama-server, run smoke_required[]
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            │ all checks pass             │ any failure
            ▼                             ▼
    ┌─────────────┐               ┌─────────────────┐
    │ LOCAL_FIRST │               │ FALLBACK_MIMO   │
    └─────────────┘               └─────────────────┘
```

## State Definitions

| State | App Behavior | Provider | User Visibility |
|-------|--------------|----------|-----------------|
| `MIMO_FIRST` | Cloud chat works | MIMO | "Local model: not configured" |
| `OFFER_LOCAL` | Wizard modal open | MIMO | Cards from `probe.offers[]` |
| `DOWNLOADING` | Background download | MIMO | Progress bar + cancel button |
| `VERIFYING` | Hash check (~5s) | MIMO | "Verifying model integrity…" |
| `SMOKE` | llama-server boot + 4 checks | MIMO | "Testing model…" with substeps |
| `LOCAL_FIRST` | Local active, MIMO fallback | local-first | "Using Qwen3.5-9B locally" |
| `FALLBACK_MIMO` | Wizard closed, log written | MIMO | Toast: "Setup didn't complete; using cloud." |

The provider field stays `MIMO` until SMOKE passes. `LOCAL_FIRST` only flips the
default after every entry in `offer.smoke_required` has returned ok.

## Probe JSON Fields the App Consumes

The app shells out to:

```bash
python3 scripts/local_lynn_app_model_probe.py --model-root ~/Models/Lynn --pretty
```

It reads these top-level fields:

| Field | Type | Use |
|-------|------|-----|
| `schema_version` | string | Must equal `lynn-app-local-model-probe-v1`; refuse otherwise |
| `platform.system` | string | Show in wizard header |
| `platform.machine` | string | Show in wizard header |
| `platform.is_macos_apple_silicon` | bool | Gate Metal-specific UI copy |
| `platform.unified_memory_gib` | number\|null | Show as "Memory: X GiB" |
| `platform.free_disk_gib` | number\|null | Show as "Free disk: X GiB"; warn if `< offer.min_free_disk_gib + 2` |
| `platform.model_root` | string | Where to download into |
| `default_provider` | string | Set as session default |
| `fallback_provider` | string | Used when local fails |
| `local_first_allowed_after_smoke` | bool | If false, hide wizard entirely |
| `decision` | string | One of `offer_local_models_after_app_setup` or `keep_mimo_first_no_local_offer` |
| `offers[]` | array | Offer cards |
| `notes[]` | array | Show as fine print |

Each `offers[i]` element:

| Field | Type | Use |
|-------|------|-----|
| `artifact_id` | string | Internal identifier, e.g. `qwen35-9b-q4km-imatrix-gguf` |
| `model` | string | Display name + API model id |
| `runtime` | string | e.g. `llama.cpp-metal` |
| `priority` | string | `recommended` or `optional` — affects card sort + label |
| `reason` | string | One-sentence card subtitle |
| `download_gib` | number | Show in card; pre-flight free-disk check |
| `min_unified_memory_gib` | int | Minimum RAM for this offer |
| `min_free_disk_gib` | int | Minimum disk for this offer |
| `smoke_required[]` | array of strings | The exact checks SMOKE must pass |

## Smoke Check Definitions

`smoke_required` strings the app must implement:

| ID | Check | Pass criterion |
|----|-------|----------------|
| `download_manifest` | Fetch + parse `manifest.json` for the artifact | Schema version + file list ok |
| `sha256` | Recompute SHA256 of GGUF, match manifest | Bytes match exactly |
| `llama_cpp_v1_models` | After server boot, `GET /v1/models` | Returns 200 with `model.id == offer.model` |
| `chat_32_tokens` | `POST /v1/chat/completions` with simple prompt, max_tokens=32 | Non-empty content, no HTTP error |
| `short_structured_smoke` | (35B only) Ask for a tiny JSON object via `response_format` | `json.loads()` succeeds |

All checks must pass. A single failure transitions to `FALLBACK_MIMO`.

## Failure Handling

| Failure | State Transition | User Toast | Log |
|---------|------------------|-----------|-----|
| Insufficient disk | OFFER_LOCAL → OFFER_LOCAL (offer disabled) | "Need X more GiB free" | `wizard.disk_short` |
| Insufficient memory | offer not shown | (silent) | `wizard.mem_short` |
| Manifest fetch 4xx/5xx | DOWNLOADING → FALLBACK_MIMO | "Couldn't reach download server" | `wizard.manifest_http` |
| Manifest schema mismatch | DOWNLOADING → FALLBACK_MIMO | "Download server returned an unsupported manifest" | `wizard.manifest_schema` |
| GGUF download interrupted | DOWNLOADING → DOWNLOADING (resume) or FALLBACK_MIMO after 3 retries | "Download paused — retrying" | `wizard.download_retry` |
| sha256 mismatch | VERIFYING → FALLBACK_MIMO | "Downloaded file is corrupted" | `wizard.sha256_fail`; offer redownload |
| llama-server fails to bind port | SMOKE → FALLBACK_MIMO | "Couldn't start local model server" | `wizard.server_bind` |
| llama-server exits during smoke | SMOKE → FALLBACK_MIMO | "Local model server crashed" | `wizard.server_crash` with last 32 log lines |
| `/v1/models` 4xx/5xx | SMOKE → FALLBACK_MIMO | "Local model didn't load" | `wizard.models_endpoint` |
| `chat_32_tokens` empty / timeout | SMOKE → FALLBACK_MIMO | "Local model didn't respond" | `wizard.chat_smoke` |
| `short_structured_smoke` invalid JSON | SMOKE → FALLBACK_MIMO (35B only) | "Local 35B couldn't produce structured output" | `wizard.structured_smoke` |
| User cancel | any → FALLBACK_MIMO | "Setup cancelled" | `wizard.user_cancel` |

The app must never delete a downloaded GGUF on failure — leave it for resume on
the next wizard run.

## Launch Copy

### Wizard entry button (settings page)

```
Use a local model
Run Qwen3.5-9B on this Mac. Stays private, works offline.
```

### Wizard step 1 — OFFER_LOCAL

> **Try Lynn local model**
>
> Lynn can run Qwen3.5-9B directly on this Mac. It stays private, works
> offline, and joins the cloud assistant as a second provider.
>
> The cloud assistant keeps working either way.

For each `offer` in `probe.offers[]`:

```
[card]
  [recommended badge if priority=recommended, otherwise "Advanced"]
  Qwen3.5-9B (Q4_K_M)
  ~5.3 GB · needs 8 GiB memory · Apple Silicon
  "Apple Silicon with enough unified memory and disk for the stable 9B
   local-agent path."
  [Install button]
```

For 35B (priority=optional):

```
[card]
  [Advanced]
  Qwen3.6-35B-A3B (Q4_K_M) — high quality, 32 GiB memory required
  ~20 GB download
  "Large-memory Mac can try the 35B quality path, but it stays opt-in."
  [Install button]
```

### Wizard step 5 — SMOKE substeps

- "Checking download" (verifying)
- "Starting local model server"
- "Asking the model to say hello"
- "Done"

### LOCAL_FIRST landing toast

```
Local model ready
Qwen3.5-9B is now your default. You can switch back to the cloud
assistant any time from Settings.
```

### FALLBACK_MIMO landing toast

```
Setup didn't finish
Lynn will keep using the cloud assistant. Open Settings → Local model
to retry.
```

## Provider Switch After LOCAL_FIRST

- Current session: `default_provider` = `local`
- Next launch: same — persistent setting
- User can manually switch back to MIMO without uninstalling
- If local server fails to start at next launch, app silently uses MIMO + shows
  the same toast as `FALLBACK_MIMO` and offers `Retry local`

## Log Schema

Every state transition writes one JSON line to `~/Library/Logs/Lynn/wizard.log`
with fields:

```json
{
  "ts": "2026-05-19T22:00:00Z",
  "from_state": "DOWNLOADING",
  "to_state": "VERIFYING",
  "artifact_id": "qwen35-9b-q4km-imatrix-gguf",
  "ok": true,
  "detail": ""
}
```

Failure transitions also include `error_code` (the `wizard.*` IDs from the
failure-handling table) and a 32-line tail of any subprocess stderr.

## What This Contract Does NOT Cover

1. The actual download server / CDN selection — owned by the dist team.
2. The MIMO provider implementation — separate contract.
3. Retraining or quality regression checks — those are R6000 territory.
4. NVIDIA/Linux/Windows wizard flow — this doc is Mac-stable only.
5. Server lifecycle (port reuse, multi-instance) — separate runtime doc.

## References

| Path | Content |
|------|---------|
| `scripts/local_lynn_app_model_probe.py` | Probe that produces the JSON consumed above |
| `scripts/local_qwen35_9b_q4km_llamacpp_server.sh` | Launcher referenced from SMOKE step |
| `scripts/local_qwen35_9b_release_qa_smoke.sh` | Manual equivalent of the in-app smoke checks |
| `docs/QWEN35_9B_INSTALL_QUICKSTART_20260519.md` | User-facing install guide |
| `docs/QWEN35_9B_RELEASE_MATRIX_20260519.md` | Release decision (Mac/NVIDIA tracks) |
| `reports/qwen35_9b/QWEN35_9B_RELEASE_EVIDENCE_INDEX_20260519.md` | Release evidence summary |
