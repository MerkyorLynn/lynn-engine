# Lynn Engine P12 One-Shot Packed-Resident Server Gate (2026-05-15)

P11 proved the internal runner can release decode-covered BF16 shadow weights
after prefill while preserving exact greedy decode ids. P12 wires that lifecycle
into the OpenAI-compatible HTTP server as an explicit opt-in mode.

This is intentionally **not** the default multi-request server path. It is a
session-scoped / one-shot serving mode:

1. load the full model with BF16 shadows,
2. run the first request prefill,
3. release decode-covered BF16 shadows,
4. continue decode from packed NVFP4 aliases,
5. reject additional prefill requests until the server is restarted.

The point is to make the packed-resident memory win observable at the service
boundary without pretending that multi-request packed prefill is solved.

## Switch

```bash
export LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1
python -m server.openai_http \
  --model /root/autodl-tmp/models/lynn-27b-variable-recovery-step5000-nvfp4-final \
  --served-name Lynn-27B-NVFP4-P12-OneShot \
  --host 127.0.0.1 \
  --port 18101
```

`/health` exposes both state flags:

```json
{
  "status": "ok",
  "model": "Lynn-27B-NVFP4-P12-OneShot",
  "release_decode_shadows_after_prefill": true,
  "release_decode_shadows_consumed": false
}
```

## R6000 Gate Result

Hardware: RTX PRO 6000 Blackwell 96 GB.

Model: `lynn-27b-variable-recovery-step5000-nvfp4-final`.

Request:

```json
{
  "model": "Lynn-27B-NVFP4-P12-OneShot",
  "prompt": "用一句话解释 MoE active parameters",
  "max_tokens": 16,
  "temperature": 0
}
```

Result:

| Metric | Value |
|---|---:|
| Released tensors | 270 |
| Released bytes | 60,636,528,640 |
| Released GiB | 56.472 |
| GPU memory after release | 27,189 MiB used / 70,065 MiB free |
| First request | PASS |
| Health after first request | `release_decode_shadows_consumed=true` |
| Second request | HTTP 409 fail-loud |

Second request response:

```json
{
  "detail": "LYNN_RELEASE_DECODE_SHADOWS_AFTER_PREFILL=1 is a one-shot/session-scoped mode. BF16 shadows were already released; restart the server for another prefill request."
}
```

## Long Decode Follow-Up

We also reran the same one-shot server gate with a chat request capped at 128
new tokens. This verifies that the release path survives a non-trivial decode,
not just a short 16-token smoke.

| Metric | Value |
|---|---:|
| Completion tokens | 128 |
| Request elapsed | 8.20 s |
| Request TPS | 15.61 tok/s |
| Decode TPS | 20.23 tok/s |
| Released tensors | 270 |
| Released GiB | 56.472 |
| GPU memory after long decode | 27,189 MiB used / 70,065 MiB free |
| Second request | HTTP 409 fail-loud |

The low TPS is expected here: this gate is the release/no-graph safety path,
not the P10 graph-slot 100+ TPS path. Its purpose is memory lifecycle
correctness and fail-loud serving semantics.

## Why This Matters

Before P11/P12, the runtime could read the 20G packed NVFP4 artifact but still
kept BF16 resident shadows around for safety, so service memory looked like a
BF16-ish runtime. P12 proves the server can cross the boundary into a
packed-resident lifecycle and fail loudly rather than silently accepting an
unsafe second prefill.

This is the first service-level proof that Lynn-native NVFP4 can claim the
memory advantage, not just the artifact-size advantage.

## Current Boundary

P12 does **not** yet solve:

- multi-request packed prefill,
- mixed concurrent sessions,
- releasing every non-decode BF16 tensor,
- default production routing into this mode.

Those are the next engineering steps. The default OpenAI server path remains
the safer multi-request path until packed prefill is implemented.
