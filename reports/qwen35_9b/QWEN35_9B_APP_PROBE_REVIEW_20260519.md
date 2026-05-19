# Qwen3.5-9B App Probe Review · 2026-05-20

## Scope

Review `scripts/local_lynn_app_model_probe.py` for Qwen3.5-9B first release readiness.
Run `py_compile`, dry probe on Mac, verify JSON fields satisfy desktop wizard requirements.

## Product Rules (from spec)

| Rule | Status |
|---|---|
| App installs first with MIMO fallback active | ✅ `default_provider: mimo`, `fallback_provider: mimo` |
| Local model only offered after hardware/disk probe | ✅ probe runs before offers are shown |
| Offer 9B Q4_K_M at Apple Silicon ≥8 GiB mem, ≥8 GiB disk | ✅ threshold correct |
| Offer 35B Q4_K_M only opt-in at ≥32 GiB mem, ≥30 GiB disk | ✅ threshold correct, `priority: optional` |
| local-first only after download + checksum + /v1/models + 32-token smoke | ✅ `smoke_required` list present per offer |
| Never block first app launch on local model | ✅ notes array, MIMO fallback always set |

## Dry Probe Result (this Mac: arm64, 24 GiB, 471 GiB free)

```json
{
  "decision": "offer_local_models_after_app_setup",
  "offers": [{"artifact_id": "qwen35-9b-q4km-imatrix-gguf", "priority": "recommended"}],
  "offer_excluded_reasons": [{"offer_id": "qwen36-35b-a3b-q4km-imatrix-gguf", "reason": "Unified memory 24.0 GiB < 32 GiB minimum."}],
  "local_first_allowed_after_smoke": true
}
```

9B offered (24 ≥ 8, 471 ≥ 8). 35B excluded (24 < 32). Correct.

## JSON Field Audit

### Fields present and correct

| Field | Purpose | Wizard needs | Status |
|---|---|---|---|
| `schema_version` | API versioning | Yes | ✅ `lynn-app-local-model-probe-v1` |
| `platform.system` | OS detection | Yes | ✅ |
| `platform.machine` | Arch detection | Yes | ✅ |
| `platform.is_macos_apple_silicon` | Gate for Metal offers | Yes | ✅ |
| `platform.unified_memory_gib` | Memory check | Yes | ✅ |
| `platform.free_disk_gib` | Disk check | Yes | ✅ |
| `platform.model_root` | Where to store models | Yes | ✅ |
| `default_provider` | Active provider on first launch | Yes | ✅ `mimo` |
| `fallback_provider` | Fallback if local fails | Yes | ✅ `mimo` |
| `local_first_allowed_after_smoke` | Wizard gate for local-first toggle | Yes | ✅ |
| `offers[].artifact_id` | Model identifier | Yes | ✅ |
| `offers[].model` | API model name | Yes | ✅ |
| `offers[].runtime` | Runtime to use | Yes | ✅ |
| `offers[].priority` | recommended / optional | Yes | ✅ |
| `offers[].reason` | User-facing explanation | Yes | ✅ |
| `offers[].download_gib` | Download size for progress bar | Yes | ✅ (fixed: 5.3→5.49) |
| `offers[].min_unified_memory_gib` | Hardware gate | Yes | ✅ |
| `offers[].min_free_disk_gib` | Disk gate | Yes | ✅ |
| `offers[].smoke_required` | Validation steps | Yes | ✅ |
| `decision` | Top-level action | Yes | ✅ |
| `notes` | Policy reminders | Nice-to-have | ✅ |

### Fields added by this review

| Field | Purpose | Wizard needs |
|---|---|---|
| `offers[].download_url_hint` | Download URL for app to initiate fetch | Yes — without this the wizard has no URL to show |
| `offer_excluded_reasons[]` | Why each non-offered model was skipped | Yes — allows wizard to show "35B needs 32 GiB" messaging |

## Improvements Made

### 1. Fix `download_gib` (5.3 → 5.49)

**Before:** `download_gib: 5.3`
**After:** `download_gib: 5.49`
**Why:** Manifest (`qwen35_9b_release_artifact_manifest.json`) declares `expected_size_gib: 5.49`. The probe should match so the wizard shows an accurate progress bar.

### 2. Add `download_url_hint` to each offer

**Before:** No URL in probe output.
**After:** `download_url_hint: "https://dl.merkyorlynn.com/models/qwen35-9b/q4_k_m/Qwen3.5-9B-Q4_K_M-imatrix.gguf"`
**Why:** The desktop wizard needs a URL to initiate the download. Without this field, the app would need to hard-code URLs elsewhere.

### 3. Add `offer_excluded_reasons` array

**Before:** Wizard could not determine *why* an offer was absent.
**After:** Array of `{"offer_id": "...", "reason": "..."}` for each non-offered model.
**Why:** Allows the wizard to display contextual messaging like "35B requires 32 GiB unified memory" instead of silently hiding the option.

## Remaining TODOs (not blocking first release)

| ID | Item | Severity | Notes |
|---|---|---|---|
| TODO-1 | HF/ModelScope download URLs are placeholders in runbook | low | `download_url_hint` currently only has `dl.merkyorlynn.com`; add HF/ModelScope when repos are ready |
| TODO-2 | 35B `download_gib` is estimated at 20.0 | low | Verify against actual 35B Q4_K_M imatrix GGUF size when available |
| TODO-3 | `download_url_hint` for 35B is a directory URL, not a file URL | low | File URL depends on exact GGUF filename for 35B |
| TODO-4 | Intel Mac exclusion reason could mention Rosetta | informational | Current: "Not Apple Silicon" — clear enough |

## Files Changed

| File | Change |
|---|---|
| `scripts/local_lynn_app_model_probe.py` | Fix download_gib, add download_url_hint, add offer_excluded_reasons |
| `reports/qwen35_9b/QWEN35_9B_APP_PROBE_REVIEW_20260519.md` | This review report |

## Verification

```
py_compile:           OK
bash -n:              N/A (Python only)
dry probe (this Mac): OK — 9B offered, 35B excluded with reason
```
